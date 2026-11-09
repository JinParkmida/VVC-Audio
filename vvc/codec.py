"""VVC codec: speech  <->  Standard MIDI File of synthesis instructions.

Pipeline
    analyze()        waveform  -> Params (float, unquantised)
    params_to_midi() Params    -> .mid (quantised to 7/14-bit MIDI values)
    midi_to_params() .mid      -> Params
    synthesize()     Params    -> waveform  (harmonic+noise channel vocoder)

The split lets us measure how much is lost in the *model* (analyze ->
synthesize) versus in the *MIDI quantisation* (via the .mid round-trip).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .smf import Track, read_smf, write_smf

PPQ = 500
TEMPO_US = 500_000            # 120 bpm at PPQ 500  ->  1 tick = 1 ms
VOICE_CH = 0                  # pitch / voicing / loudness
BAND_CH0 = 1                  # spectrum CCs start here, 12 bands per channel
BAND_CC0 = 20                 # CC 20-31 are "undefined" in General MIDI
BANDS_PER_CH = 12
AP_CC = 20                    # aperiodicity on the voice channel
EXPR_CC = 11                  # expression = loudness, for DAW audition only
PIANO_CH = 15
GM_CHOIR = 52                 # "Choir Aahs" — a voice-like default patch
GM_PIANO = 0


@dataclass
class Config:
    sr: int = 16000
    hop_ms: int = 10
    nfft: int = 0                # 0 = automatic, see __post_init__
    n_bands: int = 24
    fmin: float = 80.0
    fmax: float = 7600.0
    db_step: float = 0.75        # band level resolution; 127 steps -> ~95 dB range
    bend_range: int = 12         # pitch-bend sensitivity in semitones
    f0_min: float = 60.0
    f0_max: float = 400.0
    piano: bool = True           # also write a DAW-playable "speaking piano" track
    piano_voices: int = 6
    piano_hop_frames: int = 2

    def __post_init__(self):
        # 0 = automatic: at least 32 ms, and at least two frame steps, so analysis
        # windows always overlap by >= 50 % (fixed 512 left gaps from 32 ms frames up).
        if self.nfft == 0:
            self.nfft = max(512, 2 * self.hop)

    @property
    def hop(self) -> int:
        return self.sr * self.hop_ms // 1000

    def header(self, n_frames: int, n_samples: int, db_top: float) -> str:
        return (f"VVC/1 sr={self.sr} hop_ms={self.hop_ms} nfft={self.nfft} "
                f"bands={self.n_bands} fmin={self.fmin:g} fmax={self.fmax:g} "
                f"db_step={self.db_step:g} db_top={db_top:g} bend={self.bend_range} "
                f"frames={n_frames} samples={n_samples}")


@dataclass
class Params:
    cfg: Config
    n_samples: int
    f0: np.ndarray               # Hz per frame, 0 where unvoiced
    voiced: np.ndarray           # bool per frame
    ap: np.ndarray               # aperiodicity 0..1 per frame (1 = noise)
    band_db: np.ndarray          # (frames, bands) band energy in dB
    loud_db: np.ndarray          # frame energy in dB (DAW expression only)
    piano_notes: list = field(default_factory=list)   # (start_frame, end_frame, note, vel)

    @property
    def n_frames(self) -> int:
        return len(self.f0)


# ======================================================================== analysis
def analyze(x: np.ndarray, cfg: Config) -> Params:
    x = x / (np.max(np.abs(x)) or 1.0)
    hop = cfg.hop
    n = dsp.frame_count(len(x), hop)

    X = dsp.stft(x, cfg.nfft, hop, n)
    P = np.abs(X) ** 2
    fb = dsp.mel_filterbank(cfg.n_bands, cfg.nfft, cfg.sr, cfg.fmin, cfg.fmax)
    band_db = 10 * np.log10(P @ fb.T + 1e-12)
    loud_db = 10 * np.log10(P.sum(axis=1) + 1e-12)

    f0, ap = dsp.yin(x, cfg.sr, hop, n, cfg.f0_min, cfg.f0_max)
    gate = loud_db > loud_db.max() - 45
    voiced = (ap < 0.3) & gate & (f0 >= cfg.f0_min) & (f0 <= cfg.f0_max)
    f0, voiced = dsp.clean_voicing(f0, voiced)
    ap = np.where(voiced, ap, 1.0)

    piano = _piano_notes(x, cfg, n) if cfg.piano else []
    return Params(cfg, len(x), f0, voiced, ap, band_db, loud_db, piano)


def _piano_notes(x: np.ndarray, cfg: Config, n_frames: int):
    """Strongest spectral peaks, rounded to semitones: the classic 'speaking piano'."""
    nfft = 2048
    step = cfg.piano_hop_frames
    X = dsp.stft(x, nfft, cfg.hop, n_frames)[::step]
    mag_db = 20 * np.log10(np.abs(X) + 1e-9)
    freqs = np.fft.rfftfreq(nfft, 1 / cfg.sr)
    lo, hi = np.searchsorted(freqs, [80, 4200])
    gmax = mag_db.max()

    active: dict[int, tuple[int, int]] = {}      # note -> (start_frame, vel)
    notes = []
    for j, frame in enumerate(mag_db):
        seg = frame[lo:hi]
        pk = np.nonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
        pk = pk[(seg[pk] > gmax - 50) & (seg[pk] > seg.max() - 30)]
        pk = pk[np.argsort(seg[pk])[::-1]]
        chosen: dict[int, int] = {}
        for k in pk:
            note = int(round(69 + 12 * np.log2(freqs[lo + k] / 440)))
            if note not in chosen:
                chosen[note] = int(np.clip((seg[k] - (gmax - 50)) / 50 * 127, 1, 127))
            if len(chosen) == cfg.piano_voices:
                break
        f = j * step
        for note in list(active):
            if note not in chosen:
                s, v = active.pop(note)
                notes.append((s, f, note, v))
        for note, v in chosen.items():
            if note not in active:
                active[note] = (f, v)
    end = n_frames
    for note, (s, v) in active.items():
        notes.append((s, end, note, v))
    return sorted(notes)


# ======================================================================== MIDI mapping
def params_to_midi(p: Params, path: str) -> dict:
    cfg = p.cfg
    db_top = float(np.ceil(p.band_db.max()))
    floor = db_top - 127 * cfg.db_step
    t = lambda i: i * cfg.hop_ms     # frame -> tick (1 tick = 1 ms)

    head = Track("VVC header")
    head.tempo(0, TEMPO_US)
    head.text(0, cfg.header(p.n_frames, p.n_samples, db_top))

    voice = Track("Voice: pitch + voicing")
    voice.program(0, VOICE_CH, GM_CHOIR)
    for num, val in ((101, 0), (100, 0), (6, cfg.bend_range), (38, 0), (101, 127), (100, 127)):
        voice.cc(0, VOICE_CH, num, val)          # RPN 0: pitch-bend sensitivity
    loud_top = p.loud_db.max()
    cur_note, last_bend, last_ap, last_ex = None, None, None, None
    for i in range(p.n_frames):
        ex = int(np.clip((p.loud_db[i] - (loud_top - 60)) / 60 * 127, 0, 127))
        if ex != last_ex:
            voice.cc(t(i), VOICE_CH, EXPR_CC, ex)
            last_ex = ex
        if p.voiced[i]:
            ap = int(round(np.sqrt(p.ap[i]) * 127))
            if ap != last_ap:
                voice.cc(t(i), VOICE_CH, AP_CC, ap)
                last_ap = ap
            m = 69 + 12 * np.log2(p.f0[i] / 440)
            retrigger = cur_note is None or abs(m - cur_note) > cfg.bend_range - 0.5
            if retrigger and cur_note is not None:
                voice.note_off(t(i), VOICE_CH, cur_note)
            if retrigger:
                cur_note = int(round(m))
            bend = int(round(8192 + (m - cur_note) / cfg.bend_range * 8192))
            if bend != last_bend or retrigger:
                voice.pitch_bend(t(i), VOICE_CH, bend)
                last_bend = bend
            if retrigger:
                voice.note_on(t(i), VOICE_CH, cur_note, max(1, ex))
        elif cur_note is not None:
            voice.note_off(t(i), VOICE_CH, cur_note)
            cur_note = None
    if cur_note is not None:
        voice.note_off(t(p.n_frames), VOICE_CH, cur_note)

    spec = Track("Spectrum: mel band levels")
    q = np.clip(np.round((p.band_db - floor) / cfg.db_step), 0, 127).astype(int)
    last = np.full(cfg.n_bands, -1)
    for i in range(p.n_frames):
        for b in np.nonzero(q[i] != last)[0]:
            spec.cc(t(i), BAND_CH0 + b // BANDS_PER_CH, BAND_CC0 + b % BANDS_PER_CH, int(q[i, b]))
        last = q[i]

    tracks = [head, voice, spec]
    if p.piano_notes:
        piano = Track("Speaking piano")
        piano.program(0, PIANO_CH, GM_PIANO)
        # note-offs before note-ons at the same tick
        evs = [(e, 0, n, 0) for (s, e, n, v) in p.piano_notes] + \
              [(s, 1, n, v) for (s, e, n, v) in p.piano_notes]
        for f, kind, n, v in sorted(evs):
            if kind:
                piano.note_on(t(f), PIANO_CH, n, v)
            else:
                piano.note_off(t(f), PIANO_CH, n)
        tracks.append(piano)

    size = write_smf(path, tracks, PPQ)
    counts = {tr.events[0][2][3:].decode() if tr.events else "?": len(tr.events) for tr in tracks}
    return {"bytes": size, "events": counts}


def midi_to_params(path: str) -> Params:
    ppq, tracks = read_smf(path)
    tempo, header = TEMPO_US, None
    evs = []
    for tr in tracks:
        for tick, kind, st, data in tr:
            if kind == "meta" and st == 0x51:
                tempo = int.from_bytes(data, "big")
            elif kind == "meta" and st == 0x01 and data.startswith(b"VVC/"):
                header = data.decode()
            elif kind == "midi":
                evs.append((tick, st, data))
    if header is None:
        raise ValueError("no VVC header text event — not a VVC file")
    kv = dict(tok.split("=") for tok in header.split()[1:])
    cfg = Config(sr=int(kv["sr"]), hop_ms=int(kv["hop_ms"]), nfft=int(kv["nfft"]),
                 n_bands=int(kv["bands"]), fmin=float(kv["fmin"]), fmax=float(kv["fmax"]),
                 db_step=float(kv["db_step"]), bend_range=int(kv["bend"]))
    n_frames, n_samples = int(kv["frames"]), int(kv["samples"])
    floor = float(kv["db_top"]) - 127 * cfg.db_step
    ms_per_tick = tempo / 1000 / ppq
    frame_of = lambda tick: min(n_frames - 1, int(round(tick * ms_per_tick / cfg.hop_ms)))

    band_q = np.full((n_frames, cfg.n_bands), np.nan)
    note_at = np.full(n_frames, np.nan)       # state changes, forward-filled below
    bend_at = np.full(n_frames, np.nan)
    ap_at = np.full(n_frames, np.nan)
    ex_at = np.full(n_frames, np.nan)
    piano, piano_on = [], {}
    for tick, st, data in sorted(evs, key=lambda e: e[0]):
        f, typ, ch = frame_of(tick), st & 0xF0, st & 0x0F
        if ch == VOICE_CH:
            if typ == 0x90 and data[1] > 0:
                note_at[f] = data[0]
            elif typ in (0x80, 0x90):
                note_at[f] = -1
            elif typ == 0xE0:
                bend_at[f] = data[0] | (data[1] << 7)
            elif typ == 0xB0 and data[0] == AP_CC:
                ap_at[f] = data[1]
            elif typ == 0xB0 and data[0] == EXPR_CC:
                ex_at[f] = data[1]
        elif ch == PIANO_CH:
            if typ == 0x90 and data[1] > 0:
                piano_on[data[0]] = (f, data[1])
            elif typ in (0x80, 0x90) and data[0] in piano_on:
                s, v = piano_on.pop(data[0])
                piano.append((s, f, data[0], v))
        elif typ == 0xB0 and BAND_CC0 <= data[0] < BAND_CC0 + BANDS_PER_CH:
            b = (ch - BAND_CH0) * BANDS_PER_CH + data[0] - BAND_CC0
            if 0 <= b < cfg.n_bands:
                band_q[f, b] = data[1]
    for s_note, (s, v) in piano_on.items():
        piano.append((s, n_frames, s_note, v))

    def ffill(a, init):
        a = a.copy()
        if np.isnan(a[0]).any() if a.ndim > 1 else np.isnan(a[0]):
            a[0] = np.where(np.isnan(a[0]), init, a[0])
        for i in range(1, len(a)):
            m = np.isnan(a[i])
            a[i] = np.where(m, a[i - 1], a[i])
        return a

    band_q = ffill(band_q, 0)
    note = ffill(note_at, -1)
    bend = ffill(bend_at, 8192)
    ap_q = ffill(ap_at, 127)
    ex = ffill(ex_at, 0)

    voiced = note >= 0
    m = note + (bend - 8192) / 8192 * cfg.bend_range
    f0 = np.where(voiced, 440 * 2 ** ((m - 69) / 12), 0.0)
    ap = np.where(voiced, (ap_q / 127) ** 2, 1.0)
    return Params(cfg, n_samples, f0, voiced, ap, floor + band_q * cfg.db_step,
                  ex / 127 * 60, sorted(piano))


# ======================================================================== synthesis
def synthesize(p: Params, seed: int = 0) -> np.ndarray:
    cfg = p.cfg
    hop, sr, n = cfg.hop, cfg.sr, p.n_samples
    rng = np.random.default_rng(seed)
    frame_t = np.arange(p.n_frames) * hop
    s_t = np.arange(n)

    # --- excitation: band-limited harmonic series + white noise
    vf = p.voiced.astype(float)
    f0_fill = p.f0.copy()
    if p.voiced.any():                      # bridge unvoiced gaps for smooth interpolation
        vi = np.nonzero(p.voiced)[0]
        f0_fill = np.interp(np.arange(p.n_frames), vi, p.f0[vi])
    else:
        f0_fill[:] = 100.0
    f0_s = np.interp(s_t, frame_t, f0_fill)
    vgain = np.interp(s_t, frame_t, vf)
    phase = np.cumsum(2 * np.pi * f0_s / sr)
    harm = np.zeros(n)
    count = np.zeros(n)
    for k in range(1, int(0.5 * sr / cfg.f0_min) + 1):
        ok = k * f0_s < 0.47 * sr
        if not ok.any():
            break
        harm += np.cos(k * phase) * ok
        count += ok
    harm *= vgain / np.sqrt(np.maximum(count, 1) / 2)
    noise = rng.standard_normal(n)

    H = dsp.stft(harm, cfg.nfft, hop, p.n_frames)
    N = dsp.stft(noise, cfg.nfft, hop, p.n_frames)

    # --- per-bin noise fraction: frame aperiodicity, rising towards high frequencies
    freqs = np.fft.rfftfreq(cfg.nfft, 1 / sr)
    tilt = np.clip((freqs - 3000) / 5000, 0, 1) * 0.6
    a = p.ap[:, None] + (1 - p.ap[:, None]) * tilt[None, :]
    a = np.where(p.voiced[:, None], a, 1.0)
    Ex = np.sqrt(1 - a) * H + np.sqrt(a) * N

    # --- channel vocoder: scale each band of the excitation to the target level
    fb = dsp.mel_filterbank(cfg.n_bands, cfg.nfft, sr, cfg.fmin, cfg.fmax)
    W = dsp.band_interp_weights(fb)
    Eb = np.abs(Ex) ** 2 @ fb.T
    log_g = 0.5 * (p.band_db / 10 * np.log(10) - np.log(Eb + 1e-12))
    log_g = np.minimum(log_g, np.percentile(log_g, 99.5))
    Y = Ex * np.exp(log_g @ W)
    return dsp.istft(Y, cfg.nfft, hop, n)


def render_piano(p: Params, seed: int = 0) -> np.ndarray:
    """Simple additive 'piano' so the speaking-piano layer can be heard without a DAW."""
    cfg = p.cfg
    sr, hop = cfg.sr, cfg.hop
    y = np.zeros(p.n_samples + sr)
    for s, e, note, vel in p.piano_notes:
        f = 440 * 2 ** ((note - 69) / 12)
        start, dur = s * hop, max(1, (e - s) * hop)
        tail = int(0.04 * sr)
        tt = np.arange(dur + tail) / sr
        env = np.exp(-tt / 0.6) * np.minimum(1, tt / 0.004)
        env[dur:] *= np.linspace(1, 0, tail)
        tone = sum(a * np.sin(2 * np.pi * f * h * tt) for h, a in ((1, 1), (2, .5), (3, .25), (4, .12))
                   if f * h < 0.45 * sr)
        seg = (vel / 127) ** 2 * env * tone
        end = min(len(y), start + len(seg))
        y[start:end] += seg[: end - start]
    return y[: p.n_samples]
