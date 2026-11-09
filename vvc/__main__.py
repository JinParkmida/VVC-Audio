"""Command line:  python -m vvc                       (terminal UI)
                  python -m vvc listen                (blind listening test)
                  python -m vvc encode speech.wav speech.vvc.mid
                  python -m vvc decode speech.vvc.mid decoded.wav [--piano piano.wav]"""
import argparse
import os

from . import dsp
from .codec import Config, analyze, midi_to_params, params_to_midi, render_piano, synthesize


def main():
    ap = argparse.ArgumentParser(prog="vvc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("tui", help="open the terminal UI (default)")
    sub.add_parser("listen", help="take the blind listening test")
    e = sub.add_parser("encode", help="WAV -> VVC (.mid)")
    e.add_argument("wav")
    e.add_argument("out")
    e.add_argument("--bands", type=int, default=24, help="mel bands (12 per MIDI channel)")
    e.add_argument("--hop-ms", type=int, default=10, help="frame rate in ms")
    e.add_argument("--db-step", type=float, default=0.75, help="dB per CC step")
    e.add_argument("--no-piano", action="store_true", help="omit the speaking-piano track")
    d = sub.add_parser("decode", help="VVC (.mid) -> WAV")
    d.add_argument("mid")
    d.add_argument("out")
    d.add_argument("--piano", help="also render the speaking-piano track to this WAV")
    a = ap.parse_args()

    if a.cmd == "listen":
        from .listening import main as listen_main
        listen_main()
        return

    if a.cmd in (None, "tui"):
        from .tui import main as tui_main
        tui_main()
        return

    for out in (getattr(a, "out", None), getattr(a, "piano", None)):
        if out:                                   # out/ doesn't exist in a fresh clone
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    if a.cmd == "encode":
        cfg = Config(n_bands=a.bands, hop_ms=a.hop_ms, db_step=a.db_step, piano=not a.no_piano)
        x = dsp.load_wav(a.wav, cfg.sr)
        info = params_to_midi(analyze(x, cfg), a.out)
        secs = len(x) / cfg.sr
        print(f"{a.out}: {info['bytes']} bytes, {info['bytes'] * 8 / secs / 1000:.1f} kbit/s "
              f"for {secs:.2f} s of speech (WAV was {os.path.getsize(a.wav)} bytes)")
        for name, n in info["events"].items():
            print(f"  {name}: {n} events")
    else:
        p = midi_to_params(a.mid)
        dsp.save_wav(a.out, synthesize(p), p.cfg.sr)
        print(f"wrote {a.out}")
        if a.piano:
            dsp.save_wav(a.piano, render_piano(p), p.cfg.sr)
            print(f"wrote {a.piano}")


if __name__ == "__main__":
    main()
