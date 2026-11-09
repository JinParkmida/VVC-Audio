"""Speech recognition for judging intelligibility.

Uses OpenAI Whisper (multilingual, runs locally) when installed, otherwise the
offline Windows dictation recognizer (English only).

Model size is set with the VVC_WHISPER_MODEL environment variable
(tiny / base / small / medium / large-v3; default: medium. On the Danish test
clip "small" misheard even the original recording; medium got it right).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from . import dsp

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = os.environ.get("VVC_WHISPER_MODEL", "medium")
_model = None

try:
    import whisper  # type: ignore
except ImportError:
    whisper = None


def backend() -> str:
    return f"whisper-{MODEL_NAME}" if whisper is not None else "windows-en-US"


def _load():
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_NAME)      # downloads to ~/.cache/whisper on first use
    return _model


def is_loaded() -> bool:
    return whisper is None or _model is not None


def detect_language(path: str | Path) -> str | None:
    if whisper is None:
        return "en"
    audio = whisper.pad_or_trim(dsp.load_wav(str(path), 16000).astype("float32"))
    model = _load()
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)


def transcribe(path: str | Path, language: str | None = None) -> tuple[str, str]:
    """Return (text, language code). Pass the original clip's language when
    transcribing decoded audio, so detection can't drift to another language."""
    if whisper is None:
        return _windows([path])[0], "en"
    audio = dsp.load_wav(str(path), 16000).astype("float32")
    r = _load().transcribe(audio, language=language, fp16=False, temperature=0.0,
                           condition_on_previous_text=False)
    return r["text"].strip(), r.get("language", language or "")


def transcribe_detailed(path: str | Path, language: str | None = None) -> dict:
    """Whisper transcript plus its own confidence: `confidence` is the geometric-mean
    token probability (exp of the mean avg_logprob over segments), `no_speech` the
    probability it thinks there is no speech at all."""
    if whisper is None:
        text, lang = transcribe(path, language)
        return {"text": text, "language": lang, "confidence": None, "no_speech": None}
    import numpy as np
    audio = dsp.load_wav(str(path), 16000).astype("float32")
    r = _load().transcribe(audio, language=language, fp16=False, temperature=0.0,
                           condition_on_previous_text=False)
    segs = r.get("segments") or []
    lp = [s["avg_logprob"] for s in segs]
    return {"text": r["text"].strip(), "language": r.get("language", language or ""),
            "confidence": float(np.exp(np.mean(lp))) if lp else 0.0,
            "no_speech": float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0}


def transcribe_many(paths: list[Path], languages: dict[str, str] | None = None) -> dict[str, tuple[str, str]]:
    """Transcribe several files; `languages` maps resolved path -> language code."""
    languages = languages or {}
    if whisper is None:
        texts = _windows(paths)
        return {str(Path(p).resolve()): (t, "en") for p, t in zip(paths, texts)}
    return {str(Path(p).resolve()): transcribe(p, languages.get(str(Path(p).resolve())))
            for p in paths}


def _windows(paths) -> list[str]:
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(ROOT / "tools" / "asr.ps1"), *map(str, paths)],
        capture_output=True, text=True, timeout=1800,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    got = {}
    for line in out.stdout.splitlines():
        if "\t" in line:
            p, t = line.split("\t", 1)
            got[str(Path(p).resolve())] = t.strip()
    return [got.get(str(Path(p).resolve()), "") for p in paths]
