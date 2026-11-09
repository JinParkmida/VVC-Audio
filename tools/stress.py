"""Push the codec until the language breaks.

From the 'compact' baseline (24 bands, 20 ms, 1.5 dB) one dimension at a time is
pushed towards the extreme, and a few pitch variants test where the message lives.
Every version goes through the real .mid round-trip and is scored on:

  * gzipped size (kbit/s)
  * STOI             - objective intelligibility (0..1)
  * spectral dist.   - acoustic similarity: RMS difference of 40-band log-mel spectra (dB)
  * Whisper drift    - word error of Whisper's transcript vs its transcript of the original

Decoded audio is kept in out/stress/ for listening.
Usage:  python tools/stress.py        (needs out/results.json from tools/evaluate.py)
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from pystoi import stoi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vvc import Config, analyze, asr, dsp, midi_to_params, params_to_midi, synthesize  # noqa: E402
from vvc.eval_text import wer  # noqa: E402

OUT = ROOT / "out" / "stress"
BASE = dict(n_bands=24, hop_ms=20, db_step=1.5)

SWEEPS = {
    "bands": [24, 16, 12, 8, 6, 4, 3, 2],
    "frame ms": [10, 20, 30, 40, 60, 80, 120, 200],
    "dB/step": [0.75, 1.5, 3, 6, 10, 16, 24],
}
KEY = {"bands": "n_bands", "frame ms": "hop_ms", "dB/step": "db_step"}

VARIANTS = {
    "pitch snapped to semitones (notes, no pitch bend)":
        lambda p: replace(p, f0=np.where(p.voiced, 440 * 2 ** (np.round(12 * np.log2(np.maximum(p.f0, 1) / 440)) / 12), 0.0)),
    "monotone (one fixed pitch)":
        lambda p: replace(p, f0=np.where(p.voiced, np.median(p.f0[p.voiced]) if p.voiced.any() else 100, 0.0)),
    "whispered (no pitch at all, noise only)":
        lambda p: replace(p, voiced=np.zeros_like(p.voiced), ap=np.ones_like(p.ap)),
    "always buzzing (voicing info thrown away)":
        lambda p: _all_voiced(p),
}


def _all_voiced(p):
    v = p.voiced
    f0 = np.interp(np.arange(len(v)), np.nonzero(v)[0], p.f0[v]) if v.any() else np.full(len(v), 100.0)
    return replace(p, voiced=np.ones_like(v), f0=f0, ap=np.full_like(p.ap, 0.1))


def mel_db(x, sr=16000):
    X = dsp.stft(x, 512, 160)
    fb = dsp.mel_filterbank(40, 512, sr, 60, 7800)
    return 10 * np.log10(np.abs(X) ** 2 @ fb.T + 1e-10)


def spectral_distance(x, y):
    a, b = mel_db(x), mel_db(y[: len(x)])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    a = np.maximum(a, a.max() - 60)          # ignore differences in near-silence
    b = np.maximum(b, b.max() - 60)
    b = b - np.median(b - a)                 # remove overall gain offset
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((ROOT / "out" / "results.json").read_text(encoding="utf-8"))
    ref = {r["sample"]: (r["asr_original"], r.get("lang")) for r in prior if r["config"].startswith("default")}
    clips = sorted((ROOT / "samples").glob("*.wav"))
    clips = [c for c in clips if c.stem in ref]
    audio = {c.stem: (lambda x: x / np.max(np.abs(x)))(dsp.load_wav(str(c), 16000)) for c in clips}

    jobs = []   # (group, label, kwargs, variant)
    for dim, values in SWEEPS.items():
        for v in values:
            jobs.append((dim, f"{v:g}", {**BASE, KEY[dim]: v}, None))
    for name in VARIANTS:
        jobs.append(("pitch", name, BASE, name))

    rows = []
    t0 = time.time()
    for group, label, kw, variant in jobs:
        for stem, x in audio.items():
            cfg = Config(piano=False, **kw)
            p = analyze(x, cfg)
            tag = f"{group}-{label}".replace(" ", "_").replace("/", "per").replace("(", "").replace(")", "").replace(",", "")[:40]
            mid = OUT / f"{stem}.{tag}.mid"
            params_to_midi(p, str(mid))
            q = midi_to_params(str(mid))
            if variant:
                q = VARIANTS[variant](q)
            y = synthesize(q)
            wav = OUT / f"{stem}.{tag}.wav"
            dsp.save_wav(str(wav), y, cfg.sr)
            secs = len(x) / cfg.sr
            text, _ = asr.transcribe(wav, ref[stem][1])
            rows.append(dict(group=group, label=label, sample=stem, lang=ref[stem][1],
                             gz_kbps=len(gzip.compress(mid.read_bytes(), 9)) * 8 / secs / 1000,
                             stoi=float(stoi(x, y[: len(x)], cfg.sr)),
                             spec_db=spectral_distance(x, y),
                             heard=text, drift=wer(ref[stem][0], text), wav=str(wav)))
        g = [r for r in rows if r["group"] == group and r["label"] == label]
        print(f"[{time.time() - t0:5.0f}s] {group:8s} {label:48s} {np.mean([r['gz_kbps'] for r in g]):5.1f} kbit/s  "
              f"STOI {np.mean([r['stoi'] for r in g]):.3f}  dist {np.mean([r['spec_db'] for r in g]):4.1f} dB  "
              f"drift {np.mean([r['drift'] for r in g]):4.0%}", flush=True)

    (ROOT / "out" / "stress.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    write_md(rows, ROOT / "out" / "stress.md")


def write_md(rows, path):
    L = ["# VVC stress test", "",
         f"Baseline: compact ({BASE['n_bands']} bands, {BASE['hop_ms']} ms, {BASE['db_step']} dB/step). "
         f"One setting changed at a time; averages over {len({r['sample'] for r in rows})} clips. "
         f"Recognizer: {asr.backend()}. Drift = word error vs the recognizer's transcript of the original.", ""]
    for group in [*SWEEPS, "pitch"]:
        L += [f"## {group}", "", f"| {group} | gzipped kbit/s | STOI | spectral distance (dB) | Whisper drift | your clip heard as |",
              "|---|---|---|---|---|---|"]
        for label in dict.fromkeys(r["label"] for r in rows if r["group"] == group):
            g = [r for r in rows if r["group"] == group and r["label"] == label]
            mine = next((r["heard"] for r in g if r["sample"].startswith("rec_")), "")
            L.append(f"| {label} | {np.mean([r['gz_kbps'] for r in g]):.1f} | {np.mean([r['stoi'] for r in g]):.3f} | "
                     f"{np.mean([r['spec_db'] for r in g]):.1f} | {np.mean([r['drift'] for r in g]):.0%} | {mine} |")
        L.append("")
    L += ["## Where the two judges disagree", "",
          "Per clip: STOI says intelligible but Whisper lost words, or STOI says poor but Whisper got every word.", "",
          "| setting | clip | STOI | Whisper drift | heard |", "|---|---|---|---|---|"]
    for r in rows:
        if (r["stoi"] >= 0.80 and r["drift"] >= 0.3) or (r["stoi"] < 0.70 and r["drift"] == 0):
            L.append(f"| {r['group']} {r['label']} | {r['sample']} | {r['stoi']:.3f} | {r['drift']:.0%} | {r['heard']} |")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
