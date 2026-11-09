"""Build a blind listening test: fresh Danish sentences, each degraded by exactly one
VVC condition, presented in random order under random file names.

  * 24 sentences the listener has not heard, spoken by the Windows voice "Helle".
  * 12 conditions x 2 sentences; each sentence is heard once, in one condition only,
    so nobody can learn a sentence in a clear version and then "hear" it in a bad one.
  * Whisper medium transcribes every stimulus in advance (with its own confidence),
    so the listener's answers can be compared with the machine's on identical audio.

Nothing is destroyed. The answer key, the clean recordings, the exact .mid behind every clip
and copies of the clips go to a read-only archive, out/listening/sealed/<run>/, with SHA-256
checksums. The test screen only reads out/listening/playlist.json (clip ids and audio files)
and opens the key after the last answer. Don't open the sealed folder before taking the test.
Run again with --seed N to deal the sentences to conditions differently (e.g. for a second
listener). Every run gets its own sealed folder; earlier runs are never overwritten.

Usage:  python tools/make_listening_test.py [--seed 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import shutil
import stat
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vvc import Config, analyze, asr, dsp, midi_to_params, params_to_midi, render_piano, synthesize  # noqa: E402

OUT = ROOT / "out" / "listening"
VOICE = "Microsoft Helle"

SENTENCES = [
    "Bussen kører først igen om en time.",
    "Min søster har købt en rød cykel i går.",
    "Der står nogle kopper kaffe på køkkenbordet.",
    "Hunden løb ud i haven og gravede et hul.",
    "Vi skal huske at vande blomsterne i morgen.",
    "Toget til Aarhus er desværre forsinket i dag.",
    "Han glemte sin paraply på biblioteket.",
    "Kan du række mig saltet og peberet?",
    "Børnene byggede en snemand foran skolen.",
    "Lejligheden ved siden af er blevet solgt.",
    "Hun spiller violin hver tirsdag aften.",
    "Der er kommet et brev fra banken.",
    "Vinden blæste taget af det gamle skur.",
    "Jeg har aldrig smagt så god en æblekage.",
    "Mødet er flyttet til næste fredag.",
    "Katten sover altid i vindueskarmen.",
    "Vi tog færgen over til Bornholm sidste sommer.",
    "Butikken lukker tidligt på grund af sne.",
    "Lægen sagde, at jeg skulle drikke mere vand.",
    "Min bedstefar samler på gamle frimærker.",
    "Fodboldkampen blev aflyst på grund af regn.",
    "Der ligger en blå nøgle under måtten.",
    "Naboen maler sit hus gult i weekenden.",
    "Jeg fandt en mønt i lommen på min jakke.",
]

COMPACT = dict(n_bands=24, hop_ms=20, db_step=1.5)
# name -> (description, codec kwargs or None for the clean original, special mode)
CONDITIONS = {
    "original":  ("unprocessed TTS (control)", None, None),
    "default":   ("default: 24 bands, 10 ms", dict(n_bands=24, hop_ms=10, db_step=0.75), None),
    "compact":   ("compact: 24 bands, 20 ms", COMPACT, None),
    "bands8":    ("8 bands", {**COMPACT, "n_bands": 8}, None),
    "bands6":    ("6 bands", {**COMPACT, "n_bands": 6}, None),
    "bands4":    ("4 bands", {**COMPACT, "n_bands": 4}, None),
    "bands3":    ("3 bands", {**COMPACT, "n_bands": 3}, None),
    "frame80":   ("80 ms frames", {**COMPACT, "hop_ms": 80}, None),
    "frame120":  ("120 ms frames", {**COMPACT, "hop_ms": 120}, None),
    "frame200":  ("200 ms frames", {**COMPACT, "hop_ms": 200}, None),
    "whispered": ("whispered: compact, no pitch", COMPACT, "whisper"),
    "piano":     ("speaking piano only", dict(n_bands=24, hop_ms=10, db_step=0.75), "piano"),
}


def tts(sentences: list[str], folder: Path) -> list[Path]:
    """Speak every sentence with the Danish WinRT voice (tools/tts_winrt.ps1)."""
    folder.mkdir(parents=True, exist_ok=True)
    lst = folder / "sentences.txt"
    lst.write_text("\n".join(sentences), encoding="utf-8")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-File", str(ROOT / "tools" / "tts_winrt.ps1"),
                        "-ListFile", str(lst), "-OutDir", str(folder), "-Lang", "da-DK"],
                       capture_output=True, text=True)
    lst.unlink()
    if r.returncode != 0 or "Helle" not in r.stdout:
        raise SystemExit(f"Danish TTS failed:\n{r.stdout}\n{r.stderr}")
    print(" ", r.stdout.strip())
    return [folder / f"s{i:02d}.wav" for i in range(len(sentences))]


def render(x: np.ndarray, cond: str, mid: Path) -> np.ndarray:
    """Degrade x by one condition. The exact MIDI instructions are written to `mid`."""
    _, kw, mode = CONDITIONS[cond]
    if kw is None:
        return x
    cfg = Config(piano=(mode == "piano"), **kw)
    params_to_midi(analyze(x, cfg), str(mid))
    q = midi_to_params(str(mid))
    if mode == "whisper":
        q = replace(q, voiced=np.zeros_like(q.voiced), ap=np.ones_like(q.ap))
    return render_piano(q) if mode == "piano" else synthesize(q)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(folder: Path) -> None:
    """Make every file in the sealed archive read-only, so it can't be edited or
    deleted by accident (it can still be read, and unlocked deliberately)."""
    for f in folder.rglob("*"):
        if f.is_file():
            os.chmod(f, stat.S_IREAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    run_id = f"{datetime.now():%Y%m%d_%H%M%S}_seed{a.seed}"
    sealed = OUT / "sealed" / run_id
    clean_dir, mid_dir, copy_dir = sealed / "clean", sealed / "mid", sealed / "stim"
    stim_dir = OUT / "stim"
    for d in (clean_dir, mid_dir, copy_dir, stim_dir):
        d.mkdir(parents=True, exist_ok=True)
    for f in stim_dir.glob("*.wav"):        # the previous run's clips; archived in its sealed folder
        f.unlink()

    print(f"speaking {len(SENTENCES)} sentences with {VOICE} ...", flush=True)
    sources = tts(SENTENCES, clean_dir)

    rng = random.Random(a.seed)
    conds = list(CONDITIONS) * (len(SENTENCES) // len(CONDITIONS))
    rng.shuffle(conds)                                   # which sentence gets which condition
    trials = []
    for i, (sent, src, cond) in enumerate(zip(SENTENCES, sources, conds)):
        sid = secrets.token_hex(3)
        clean = clean_dir / f"{sid}.wav"
        src.rename(clean)                                # clean recording, named by clip id
        x = dsp.load_wav(str(clean), 16000)
        x /= np.max(np.abs(x))
        mid = mid_dir / f"{sid}.mid"
        y = render(x, cond, mid)
        path = stim_dir / f"{sid}.wav"
        dsp.save_wav(str(path), y, 16000)
        shutil.copy2(path, copy_dir / path.name)
        w = asr.transcribe_detailed(path, "da")
        trials.append(dict(id=sid, sentence=sent, condition=cond, description=CONDITIONS[cond][0],
                           file=str(path.relative_to(ROOT)), whisper=w["text"],
                           whisper_confidence=w["confidence"], whisper_no_speech=w["no_speech"],
                           sha256_stimulus=sha256(path), sha256_clean=sha256(clean),
                           sha256_mid=sha256(mid) if mid.exists() else None))
        print(f"  {i + 1:2d}/{len(SENTENCES)} prepared", flush=True)   # deliberately no spoilers
    rng.shuffle(trials)                                  # presentation order

    key = {"run_id": run_id, "seed": a.seed, "language": "da", "voice": VOICE,
           "recognizer": asr.backend(), "created": datetime.now().isoformat(timespec="seconds"),
           "conditions": {k: v[0] for k, v in CONDITIONS.items()},
           "condition_settings": {k: v[1] for k, v in CONDITIONS.items()},
           "trials": trials}
    (sealed / "key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    (sealed / "README.txt").write_text(
        "Sealed answer key for one VVC listening-test run. Don't open before taking the test.\n"
        "key.json: sentence, condition and Whisper result per clip (in presentation order), plus SHA-256\n"
        "  checksums of every file. clean/: the unprocessed TTS recordings. mid/: the exact VVC\n"
        "  instructions each clip was decoded from. stim/: copies of the clips the listener heard.\n"
        "Files are read-only on purpose. They are never deleted by tools/make_listening_test.py.\n",
        encoding="utf-8")
    seal(sealed)

    # the only file the test screen reads before the end: clip ids and audio, no answers
    (OUT / "playlist.json").write_text(json.dumps(
        {"run_id": run_id, "key": str((sealed / "key.json").relative_to(ROOT)),
         "clips": [{"id": t["id"], "file": t["file"]} for t in trials]}, indent=1), encoding="utf-8")
    print(f"ready: {len(trials)} clips, run {run_id}. Answer key sealed in "
          f"{sealed.relative_to(ROOT)}\nStart the test with:  python -m vvc listen")


if __name__ == "__main__":
    main()
