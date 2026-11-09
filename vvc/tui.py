"""Terminal UI for VVC.   Run:  python -m vvc   (or python -m vvc tui)"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Vertical
from textual.widgets import (Button, Footer, Header, Input, Label, ListItem, ListView,
                             RichLog, Select, Sparkline, Static, Switch)

from . import asr, dsp
from .codec import Config, analyze, midi_to_params, params_to_midi, render_piano, synthesize
from .listening import ListeningScreen
from .smf import read_smf

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "out"

PRESETS = {
    "rich": (36, 10, 0.75),
    "default": (24, 10, 0.75),
    "compact": (24, 20, 1.5),
    "small": (16, 20, 2.0),
    "tiny": (12, 30, 3.0),
}

try:
    import sounddevice as sd
except Exception:            # optional: recording only
    sd = None
try:
    import winsound
except ImportError:
    winsound = None


def input_devices() -> list[tuple[str, int]]:
    """Input devices on the same host API as the system default input (avoids
    listing every device four times on Windows)."""
    if sd is None:
        return []
    try:
        default = sd.query_devices(kind="input")
        api = default["hostapi"]
        return [(d["name"], i) for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0 and d["hostapi"] == api]
    except Exception:
        return []


def fmt_bytes(n: int) -> str:
    return f"{n / 1024:.1f} KB" if n >= 1024 else f"{n} B"


class FileItem(ListItem):
    def __init__(self, path: Path):
        self.path = path
        icon = {".wav": "♪", ".mid": "♫"}.get(path.suffix, "·")
        where = path.parent.name
        super().__init__(Label(f"{icon} [dim]{where}/[/]{path.name}"))


class VVCApp(App):
    TITLE = "VVC — Virtual Voice Complaint"
    SUB_TITLE = "speech ⇄ MIDI synthesis instructions"
    CSS = """
    Screen { layout: horizontal; }
    #left { width: 42%; min-width: 36; }
    #right { width: 1fr; }
    .box { border: round $primary 60%; border-title-color: $accent; padding: 0 1; }
    #files { height: 1fr; }
    #info { height: auto; min-height: 7; }
    #viz { height: auto; }
    #viz Sparkline { height: 2; margin-bottom: 0; }
    #viz Label { color: $text-muted; }
    #settings { layout: grid; grid-size: 4; grid-columns: 10 1fr 10 1fr; grid-gutter: 0 1;
                height: auto; }
    #settings Label { padding-top: 1; }
    #settings Input, #settings Select { width: 1fr; }
    #record { layout: grid; grid-size: 4; grid-columns: 10 1fr 10 1fr; grid-gutter: 0 1;
              height: auto; }
    #record Label { padding-top: 1; }
    #actions { layout: grid; grid-size: 4; grid-gutter: 0 1; height: auto; padding: 0 1; }
    #actions Button { width: 1fr; }
    #log { height: 1fr; min-height: 6; }
    """
    BINDINGS = [
        Binding("t", "roundtrip", "Round-trip"),
        Binding("e", "encode", "Encode"),
        Binding("d", "decode", "Decode"),
        Binding("p", "play", "Play"),
        Binding("s", "stop", "Stop"),
        Binding("r", "record", "Record"),
        Binding("a", "transcribe", "Transcribe"),
        Binding("l", "listening", "Listening test"),
        Binding("f5", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.selected: Path | None = None
        self.busy = False
        self.languages: dict[str, str] = {}      # sample path -> detected language

    def check_action(self, action: str, parameters) -> bool | None:
        # main-screen shortcuts are off while the listening test is open
        if isinstance(self.screen, ListeningScreen) and action in {b.action for b in self.BINDINGS}:
            return False
        return True

    def action_listening(self) -> None:
        self.action_stop()
        self.push_screen(ListeningScreen())

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="left"):
            files = ListView(id="files", classes="box")
            files.border_title = "Files  (samples/ · out/)"
            yield files
            info = Static("Select a file.", id="info", classes="box")
            info.border_title = "Details"
            yield info
        with Vertical(id="right"):
            with Grid(id="settings", classes="box") as g:
                g.border_title = "Encoder settings"
                yield Label("Preset")
                yield Select([(k, k) for k in PRESETS], value="default", allow_blank=False, id="preset")
                yield Label("Piano trk")
                yield Switch(value=True, id="piano")
                yield Label("Bands")
                yield Input("24", id="bands", type="integer")
                yield Label("Frame ms")
                yield Input("10", id="hop", type="integer")
                yield Label("dB/step")
                yield Input("0.75", id="dbstep", type="number")
            with Grid(id="record", classes="box") as g:
                g.border_title = "Microphone"
                devs = input_devices()
                yield Label("Device")
                if devs:
                    yield Select(devs, value=devs[0][1], allow_blank=False, id="device")
                else:
                    yield Label("[dim]no input devices (pip install sounddevice)[/]")
                yield Label("Seconds")
                yield Input("5", id="secs", type="number")
                yield Label("Words")
                yield Input(placeholder="what you will say (optional, saved as .txt)", id="words")
            with Grid(id="actions"):
                yield Button("● Record", id="b-record", variant="error")
                yield Button("▶ Play", id="b-play", variant="success")
                yield Button("■ Stop", id="b-stop")
                yield Button("⇄ Round-trip", id="b-roundtrip", variant="primary")
                yield Button("Encode → .mid", id="b-encode")
                yield Button("Decode → .wav", id="b-decode")
                yield Button("Transcribe", id="b-transcribe")
                yield Button("Refresh", id="b-refresh")
            with Vertical(id="viz", classes="box") as v:
                v.border_title = "Instructions"
                yield Label("pitch (f0) — note + pitch bend", id="l-pitch")
                yield Sparkline([0], id="sp-pitch")
                yield Label("loudness — sum of band CCs", id="l-loud")
                yield Sparkline([0], id="sp-loud")
            log = RichLog(id="log", classes="box", markup=True, wrap=True)
            log.border_title = "Log"
            yield log
        yield Footer()

    async def on_mount(self) -> None:
        SAMPLES.mkdir(exist_ok=True)
        OUT.mkdir(exist_ok=True)
        await self.action_refresh()
        self.log_msg("Pick a [b]WAV[/] and press [b]t[/] to round-trip it through MIDI, "
                     "or [b]r[/] to record your voice.")
        if winsound is None:
            self.log_msg("[yellow]Playback uses winsound and only works on Windows.[/]")

    # ------------------------------------------------------------------ helpers
    def log_msg(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    def tlog(self, msg: str) -> None:
        """Log from a worker thread."""
        self.call_from_thread(self.log_msg, msg)

    def config(self) -> Config | None:
        try:
            bands = int(self.query_one("#bands", Input).value)
            hop = int(self.query_one("#hop", Input).value)
            db = float(self.query_one("#dbstep", Input).value)
        except ValueError:
            self.log_msg("[red]Settings must be numbers.[/]")
            return None
        if not (4 <= bands <= 96 and 5 <= hop <= 100 and 0.1 <= db <= 10):
            self.log_msg("[red]Out of range: bands 4–96, frame 5–100 ms, dB/step 0.1–10.[/]")
            return None
        return Config(n_bands=bands, hop_ms=hop, db_step=db,
                      piano=self.query_one("#piano", Switch).value)

    def need(self, *suffixes: str) -> Path | None:
        p = self.selected
        if p is None or p.suffix not in suffixes:
            self.log_msg(f"[yellow]Select a {' or '.join(suffixes)} file first.[/]")
            return None
        return p

    def start(self) -> bool:
        if self.busy:
            self.log_msg("[yellow]Busy — wait for the current job.[/]")
            return False
        self.busy = True
        return True

    def done(self) -> None:
        self.busy = False

    # ------------------------------------------------------------------ files
    async def action_refresh(self, select: Path | None = None) -> None:
        lv = self.query_one("#files", ListView)
        paths = sorted(SAMPLES.glob("*.wav")) + sorted(
            [*OUT.glob("*.mid"), *OUT.glob("*.wav")], key=lambda p: p.name)
        await lv.clear()                      # async: must finish before the index is set
        await lv.extend(FileItem(p) for p in paths)
        target = select or self.selected
        if target is not None:
            for i, p in enumerate(paths):
                if p == target:
                    lv.index = i
                    break

    @on(ListView.Highlighted, "#files")
    def file_highlighted(self, event: ListView.Highlighted) -> None:
        if isinstance(event.item, FileItem):
            self.selected = event.item.path
            self.describe(event.item.path)

    @work(thread=True, exclusive=True, group="describe")
    def describe(self, path: Path) -> None:
        try:
            text, f0, loud = self.inspect(path)
        except Exception as ex:
            text, f0, loud = f"[red]{ex}[/]", None, None
        self.call_from_thread(self.show_details, path, text, f0, loud)

    def show_details(self, path: Path, text: str, f0, loud) -> None:
        if path != self.selected:
            return
        self.query_one("#info", Static).update(text)
        if f0 is not None:
            v = f0[f0 > 0]
            base = 0.85 * v.min() if len(v) else 0.0   # show intonation, not the offset from 0 Hz
            self.query_one("#sp-pitch", Sparkline).data = list(np.where(f0 > 0, f0 - base, 0.0))
            self.query_one("#sp-loud", Sparkline).data = list(loud)
            voiced = f0[f0 > 0]
            rng = f"{voiced.min():.0f}–{voiced.max():.0f} Hz" if len(voiced) else "unvoiced"
            self.query_one("#l-pitch", Label).update(f"pitch (f0) — note + pitch bend   [dim]{rng}[/]")

    def inspect(self, path: Path):
        size = path.stat().st_size
        if path.suffix == ".wav":
            from scipy.io import wavfile
            sr, x = wavfile.read(path)
            secs = len(x) / sr
            lines = [f"[b]{path.name}[/]", f"{secs:.2f} s · {sr} Hz · {fmt_bytes(size)} · "
                     f"{size * 8 / secs / 1000:.0f} kbit/s"]
            txt = path.with_suffix(".txt")
            if txt.exists():
                lines.append(f"words: [i]{txt.read_text(encoding='utf-8', errors='replace').strip()}[/]")
            cfg = Config(piano=False)
            p = analyze(dsp.load_wav(str(path), cfg.sr), cfg)
        else:
            ppq, tracks = read_smf(str(path))
            p = midi_to_params(str(path))
            secs = p.n_samples / p.cfg.sr
            names = []
            for tr in tracks:
                name = next((d.decode(errors="replace") for _, k, t, d in tr if k == "meta" and t == 3), "?")
                n_midi = sum(1 for e in tr if e[1] == "midi")
                names.append(f"  {name}: " + (f"{n_midi} MIDI events" if n_midi else "settings (text + tempo)"))
            c = p.cfg
            lines = [f"[b]{path.name}[/]",
                     f"{secs:.2f} s · {fmt_bytes(size)} · [b]{size * 8 / secs / 1000:.1f} kbit/s[/]",
                     f"{c.n_bands} bands · {c.hop_ms} ms frames · {c.db_step:g} dB/step · "
                     f"{100 * p.voiced.mean():.0f}% voiced",
                     *names]
        f0 = np.where(p.voiced, p.f0, 0.0)
        loud = 10 * np.log10(np.sum(10 ** (p.band_db / 10), axis=1) + 1e-12)
        loud = np.clip(loud - (loud.max() - 60), 0, None)
        k = max(1, len(f0) // 120)                      # downsample for the sparkline
        n = len(f0) // k * k
        return "\n".join(lines), f0[:n].reshape(-1, k).max(1), loud[:n].reshape(-1, k).mean(1)

    # ------------------------------------------------------------------ settings
    @on(Select.Changed, "#preset")
    def preset_changed(self, event: Select.Changed) -> None:
        bands, hop, db = PRESETS[event.value]
        self.query_one("#bands", Input).value = str(bands)
        self.query_one("#hop", Input).value = str(hop)
        self.query_one("#dbstep", Input).value = f"{db:g}"

    # ------------------------------------------------------------------ buttons
    @on(Button.Pressed)
    async def button(self, event: Button.Pressed) -> None:
        await self.run_action((event.button.id or "").removeprefix("b-"))

    # ------------------------------------------------------------------ actions
    def action_play(self) -> None:
        p = self.need(".wav")
        if p is None:
            return
        if winsound is None:
            self.log_msg("[red]Playback needs Windows (winsound).[/]")
            return
        winsound.PlaySound(str(p), winsound.SND_FILENAME | winsound.SND_ASYNC)
        self.log_msg(f"▶ playing {p.name}")

    def action_stop(self) -> None:
        if winsound is not None:
            winsound.PlaySound(None, 0)
        if sd is not None:
            sd.stop()

    def action_encode(self) -> None:
        p, cfg = self.need(".wav"), None
        if p is not None and (cfg := self.config()) is not None and self.start():
            self.run_encode(p, cfg, decode=False)

    def action_roundtrip(self) -> None:
        p, cfg = self.need(".wav"), None
        if p is not None and (cfg := self.config()) is not None and self.start():
            self.run_encode(p, cfg, decode=True)

    def action_decode(self) -> None:
        p = self.need(".mid")
        if p is not None and self.start():
            self.run_decode(p)

    @work(thread=True)
    def run_encode(self, wav: Path, cfg: Config, decode: bool) -> None:
        try:
            t0 = time.perf_counter()
            x = dsp.load_wav(str(wav), cfg.sr)
            x = x / (np.max(np.abs(x)) or 1.0)
            secs = len(x) / cfg.sr
            mid = OUT / f"{wav.stem}.vvc.mid"
            info = params_to_midi(analyze(x, cfg), str(mid))
            self.tlog(f"[b]encoded[/] {wav.name} → {mid.name}: {fmt_bytes(info['bytes'])}, "
                      f"[b]{info['bytes'] * 8 / secs / 1000:.1f} kbit/s[/] "
                      f"(WAV {fmt_bytes(wav.stat().st_size)}, "
                      f"{wav.stat().st_size / info['bytes']:.1f}× larger)")
            last = mid
            if decode:
                last = self._decode(mid, reference=x)
            self.tlog(f"[dim]done in {time.perf_counter() - t0:.1f} s[/]")
            self.call_from_thread(self.action_refresh, last)
            if decode and winsound is not None:
                winsound.PlaySound(str(last), winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.tlog(f"▶ playing {last.name}  ([b]s[/] to stop)")
        except Exception as ex:
            self.tlog(f"[red]encode failed: {ex}[/]")
        finally:
            self.call_from_thread(self.done)

    @work(thread=True)
    def run_decode(self, mid: Path) -> None:
        try:
            out = self._decode(mid)
            self.call_from_thread(self.action_refresh, out)
        except Exception as ex:
            self.tlog(f"[red]decode failed: {ex}[/]")
        finally:
            self.call_from_thread(self.done)

    def _decode(self, mid: Path, reference: np.ndarray | None = None) -> Path:
        p = midi_to_params(str(mid))
        y = synthesize(p)
        stem = mid.name.removesuffix(".vvc.mid").removesuffix(".mid")
        out = OUT / f"{stem}.decoded.wav"
        dsp.save_wav(str(out), y, p.cfg.sr)
        msg = f"[b]decoded[/] {mid.name} → {out.name}"
        if p.piano_notes:
            pia = OUT / f"{stem}.piano.wav"
            dsp.save_wav(str(pia), render_piano(p), p.cfg.sr)
            msg += f" (+ {pia.name})"
        self.tlog(msg)
        if reference is not None:
            try:
                from pystoi import stoi
                n = min(len(reference), len(y))
                score = stoi(reference[:n], y[:n], p.cfg.sr)
                verdict = "excellent" if score > .93 else "good" if score > .88 else "fair" if score > .8 else "poor"
                self.tlog(f"intelligibility (STOI): [b]{score:.3f}[/] — {verdict}")
            except ImportError:
                pass
        return out

    # ------------------------------------------------------------------ record
    def action_record(self) -> None:
        if sd is None:
            self.log_msg("[red]Recording needs: pip install sounddevice[/]")
            return
        try:
            secs = float(self.query_one("#secs", Input).value)
            dev = self.query_one("#device", Select).value
        except Exception:
            self.log_msg("[red]Pick a device and a number of seconds.[/]")
            return
        if not 0.5 <= secs <= 60:
            self.log_msg("[red]Seconds must be 0.5–60.[/]")
            return
        if self.start():
            self.run_record(secs, dev, self.query_one("#words", Input).value.strip())

    @work(thread=True)
    def run_record(self, secs: float, device: int, words: str) -> None:
        try:
            sr = int(sd.query_devices(device)["default_samplerate"])
            for i in (3, 2, 1):
                self.tlog(f"[b]recording in {i}…[/]")
                time.sleep(0.6)
            self.tlog(f"[b red]● REC[/] speak now — {secs:g} s")
            audio = sd.rec(int(secs * sr), samplerate=sr, channels=1, device=device, dtype="float32")
            sd.wait()
            x = audio[:, 0].astype(np.float64)
            peak = float(np.max(np.abs(x)))
            if peak < 1e-4:
                self.tlog("[red]Recording is silent — check the device / microphone permissions.[/]")
                return
            name = datetime.now().strftime("rec_%Y%m%d_%H%M%S")
            path = SAMPLES / f"{name}.wav"
            from scipy.io import wavfile
            wavfile.write(path, sr, (x / peak * 0.89 * 32767).astype(np.int16))
            if words:
                path.with_suffix(".txt").write_text(words, encoding="utf-8")
            level = 20 * np.log10(peak)
            self.tlog(f"saved {path.name} (peak {level:.0f} dBFS"
                      f"{', [yellow]quiet — move closer[/]' if level < -30 else ''}). "
                      f"Press [b]t[/] to round-trip it.")
            self.call_from_thread(self.action_refresh, path)
        except Exception as ex:
            self.tlog(f"[red]recording failed: {ex}[/]")
        finally:
            self.call_from_thread(self.done)

    # ------------------------------------------------------------------ ASR
    def action_transcribe(self) -> None:
        p = self.need(".wav")
        if p is not None and self.start():
            self.run_transcribe(p)

    @staticmethod
    def source_sample(path: Path) -> Path | None:
        """The samples/*.wav a file in out/ was made from (longest matching stem)."""
        if path.parent == SAMPLES:
            return path
        stems = sorted((p for p in SAMPLES.glob("*.wav") if path.name.startswith(p.stem + ".")),
                       key=lambda p: len(p.stem), reverse=True)
        return stems[0] if stems else None

    @work(thread=True)
    def run_transcribe(self, wav: Path) -> None:
        try:
            if asr.whisper is None and os.name != "nt":
                self.tlog("[red]Install openai-whisper to transcribe on this system.[/]")
                return
            if not asr.is_loaded():
                self.tlog(f"loading {asr.backend()} (first time takes ~20 s) …")
            src = self.source_sample(wav)
            lang = None
            if src is not None:
                if str(src) not in self.languages:
                    self.languages[str(src)] = asr.detect_language(src)
                lang = self.languages[str(src)]
            self.tlog(f"transcribing {wav.name} with {asr.backend()}"
                      f"{f' as [b]{lang}[/] (language of {src.name})' if lang and src != wav else ''} …")
            text, lang = asr.transcribe(wav, lang)
            if src == wav and lang:
                self.languages[str(src)] = lang
            self.tlog(f"heard ({lang}): [b]{text or '(nothing)'}[/]")
            ref = src.with_suffix(".txt") if src is not None else None
            if ref is not None and ref.exists() and text:
                from .eval_text import wer
                reference = ref.read_text(encoding="utf-8", errors="replace").strip()
                self.tlog(f"word error vs {ref.name}: [b]{wer(reference, text):.0%}[/]  "
                          f"[dim](reference: {reference})[/]")
        except Exception as ex:
            self.tlog(f"[red]transcription failed: {ex}[/]")
        finally:
            self.call_from_thread(self.done)


def main() -> None:
    VVCApp().run()


if __name__ == "__main__":
    main()
