"""Run the VVC experiment over samples/*.wav and report size vs intelligibility.

For every sample and every configuration:
  * encode -> .mid, decode -> .wav
  * STOI (0..1 objective intelligibility) of decoded vs original
  * word error rate of a speech recognizer on the decoded audio: Whisper
    (multilingual, see vvc/asr.py) if installed, else the Windows en-US recognizer
Also reported: the 'oracle' (vocoder fed unquantised parameters, no MIDI),
which separates loss from the parametric model from loss from MIDI quantisation.

Usage:  python tools/evaluate.py            (writes out/results.md and out/results.json)
"""
from __future__ import annotations

import gzip
import json

import sys
from pathlib import Path

import numpy as np
from pystoi import stoi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vvc import Config, analyze, midi_to_params, params_to_midi, render_piano, synthesize  # noqa: E402
from vvc import asr, dsp  # noqa: E402
from vvc.eval_text import wer  # noqa: E402

CONFIGS = {
    "rich (36 bands, 10 ms, 0.75 dB)": dict(n_bands=36, hop_ms=10, db_step=0.75),
    "default (24 bands, 10 ms, 0.75 dB)": dict(n_bands=24, hop_ms=10, db_step=0.75),
    "compact (24 bands, 20 ms, 1.5 dB)": dict(n_bands=24, hop_ms=20, db_step=1.5),
    "small (16 bands, 20 ms, 2 dB)": dict(n_bands=16, hop_ms=20, db_step=2.0),
    "tiny (12 bands, 30 ms, 3 dB)": dict(n_bands=12, hop_ms=30, db_step=3.0),
}


def transcribe(rows: list[dict], originals: list[Path], others: list[Path]) -> dict[str, tuple[str, str]]:
    """Originals first (language auto-detected), then every derived file in its
    original's language, so a degraded decode can't be 'heard' as another language."""
    res = asr.transcribe_many(originals)
    lang_of_sample = {Path(p).stem: lang for p, (_, lang) in res.items()}
    langs = {}
    for r in rows:
        for k in ("decoded", "oracle", "piano"):
            if k in r:
                langs[r[k]] = lang_of_sample.get(r["sample"])
    res.update(asr.transcribe_many(others, langs))
    return res


def main():
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    samples = sorted((ROOT / "samples").glob("*.wav"))
    rows, originals, to_asr = [], [], []

    for wav in samples:
        ref_text = wav.with_suffix(".txt").read_text(encoding="utf-8").strip() if wav.with_suffix(".txt").exists() else ""
        stem = wav.stem
        originals.append(wav)
        for ci, (cname, kw) in enumerate(CONFIGS.items()):
            is_default = cname.startswith("default")
            cfg = Config(piano=is_default, **kw)
            x = dsp.load_wav(str(wav), cfg.sr)
            x = x / np.max(np.abs(x))
            secs = len(x) / cfg.sr
            p = analyze(x, cfg)

            tag = cname.split()[0]
            mid = out_dir / f"{stem}.{tag}.vvc.mid"
            params_to_midi(p, str(mid))
            data = mid.read_bytes()
            q = midi_to_params(str(mid))
            y = synthesize(q)
            dec = out_dir / f"{stem}.{tag}.decoded.wav"
            dsp.save_wav(str(dec), y, cfg.sr)
            to_asr.append(dec)

            row = dict(sample=stem, config=cname, ref=ref_text, secs=secs,
                       mid_bytes=len(data), gz_bytes=len(gzip.compress(data, 9)),
                       stoi=float(stoi(x, y[: len(x)], cfg.sr)), decoded=str(dec.resolve()))
            if is_default:
                # piano-free size, for a fair bitrate comparison with the other configs
                cfg_np = Config(piano=False, **kw)
                tmp = out_dir / f"{stem}.nopiano.tmp.mid"
                params_to_midi(analyze(x, cfg_np), str(tmp))
                row["mid_bytes_no_piano"] = tmp.stat().st_size
                row["gz_bytes_no_piano"] = len(gzip.compress(tmp.read_bytes(), 9))
                tmp.unlink()

                oracle = synthesize(p)
                orc = out_dir / f"{stem}.oracle.wav"
                dsp.save_wav(str(orc), oracle, cfg.sr)
                row["stoi_oracle"] = float(stoi(x, oracle[: len(x)], cfg.sr))
                row["oracle"] = str(orc.resolve())
                to_asr.append(orc)

                pia = out_dir / f"{stem}.piano.wav"
                py = render_piano(q)
                dsp.save_wav(str(pia), py, cfg.sr)
                row["stoi_piano"] = float(stoi(x, py[: len(x)], cfg.sr))
                row["piano"] = str(pia.resolve())
                to_asr.append(pia)
            rows.append(row)
            print(f"{stem:15s} {cname:38s} {len(data):7d} B  STOI {row['stoi']:.3f}", flush=True)

    print(f"transcribing {len(originals) + len(to_asr)} files with {asr.backend()} ...", flush=True)
    heard = transcribe(rows, originals, to_asr)
    for row in rows:
        orig = str((ROOT / "samples" / f"{row['sample']}.wav").resolve())
        row["asr_backend"] = asr.backend()
        row["asr_original"], row["lang"] = heard.get(orig, ("", ""))
        row["wer_original"] = wer(row["ref"], row["asr_original"])
        row["asr_decoded"] = heard.get(row["decoded"], ("", ""))[0]
        row["wer_decoded"] = wer(row["ref"], row["asr_decoded"])
        row["drift"] = wer(row["asr_original"], row["asr_decoded"])
        for k in ("oracle", "piano"):
            if k in row:
                row[f"asr_{k}"] = heard.get(row[k], ("", ""))[0]
                row[f"wer_{k}"] = wer(row["ref"], row[f"asr_{k}"])

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, out_dir / "results.md")
    print((out_dir / "results.md").read_text(encoding="utf-8"))


def write_markdown(rows, path: Path):
    total_secs = sum(r["secs"] for r in rows if r["config"].startswith("default"))
    pcm_kbps = 16000 * 16 / 1000
    lines = ["# VVC results", "",
             f"{len({r['sample'] for r in rows})} samples, {total_secs:.1f} s of speech in total. "
             f"Source audio is 16 kHz 16-bit PCM = {pcm_kbps:.0f} kbit/s. "
             f"Speech recognizer: {rows[0].get('asr_backend', 'windows-en-US')}.", "",
             "## Size vs intelligibility (averaged over samples)", "",
             "| Configuration | .mid kbit/s | gzipped kbit/s | STOI | Recognizer WER | Transcript drift vs original |",
             "|---|---|---|---|---|---|"]
    for cname in CONFIGS:
        rs = [r for r in rows if r["config"] == cname]
        secs = sum(r["secs"] for r in rs)
        key = "mid_bytes_no_piano" if cname.startswith("default") else "mid_bytes"
        gkey = "gz_bytes_no_piano" if cname.startswith("default") else "gz_bytes"
        kb = sum(r[key] for r in rs) * 8 / secs / 1000
        gz = sum(r[gkey] for r in rs) * 8 / secs / 1000
        lines.append(f"| {cname} | {kb:.1f} | {gz:.1f} | {np.mean([r['stoi'] for r in rs]):.3f} | "
                     f"{np.mean([r['wer_decoded'] for r in rs]):.0%} | {np.mean([r['drift'] for r in rs]):.0%} |")
    d = [r for r in rows if r["config"].startswith("default")]
    lines += ["", "Sizes exclude the optional speaking-piano track. WER is against the reference text; "
              "*drift* is the word error between the decoded audio's transcript and the original audio's "
              "transcript, which removes the recognizer's own mistakes (e.g. \"730\" for \"seven thirty\").", "",
              "## Reference points (default configuration)", "",
              "| Signal | STOI | Recognizer WER |", "|---|---|---|",
              f"| Original recording | 1.000 | {np.mean([r['wer_original'] for r in d]):.0%} |",
              f"| Oracle: vocoder with unquantised parameters (no MIDI) | "
              f"{np.mean([r['stoi_oracle'] for r in d]):.3f} | {np.mean([r['wer_oracle'] for r in d]):.0%} |",
              f"| Decoded from VVC .mid | {np.mean([r['stoi'] for r in d]):.3f} | "
              f"{np.mean([r['wer_decoded'] for r in d]):.0%} |",
              f"| Speaking-piano track only | {np.mean([r['stoi_piano'] for r in d]):.3f} | "
              f"{np.mean([r['wer_piano'] for r in d]):.0%} |",
              "", "## Transcripts", ""]
    for r in d:
        lines += [f"**{r['sample']}** [{r.get('lang', '?')}] — reference: \"{r['ref']}\"", "",
                  f"- original: {r['asr_original']}",
                  f"- decoded (default): {r['asr_decoded']}"]
        for cname in CONFIGS:
            if cname.startswith("default"):
                continue
            rr = next(x for x in rows if x["sample"] == r["sample"] and x["config"] == cname)
            lines.append(f"- decoded ({cname.split()[0]}): {rr['asr_decoded']}")
        lines += [f"- speaking piano: {r['asr_piano']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
