# The VVC file format (version 1)

A VVC file is a Standard MIDI File (SMF, type 1) that stores speech as synthesis instructions:
when a voice is on, what pitch it has, how noisy it is, and how loud each part of the spectrum
is over time. It holds no audio samples. Any MIDI tool can open it; a VVC decoder turns it back
into speech.

This document describes exactly what the reference encoder in `vvc/codec.py` writes and what the
decoder expects.

## Timing

| Field | Value |
|---|---|
| Division | 500 ticks per quarter note |
| Tempo | 500 000 µs per quarter note (120 bpm) |
| Resulting tick | 1 ms |

Speech is analysed in frames of `hop_ms` milliseconds (10 ms by default). Frame *i* starts at tick
`i × hop_ms`. The decoder reads the tempo and division from the file and maps every event to the
nearest frame, so a file re-saved with a different division or tempo still decodes correctly.

## Tracks

| # | Track name | MIDI channel | Content |
|---|---|---|---|
| 0 | `VVC header` | none | Tempo, and the settings as a text event |
| 1 | `Voice: pitch + voicing` | 1 | Notes, pitch bend, aperiodicity, loudness |
| 2 | `Spectrum: mel band levels` | 2, 3, … | One controller lane per frequency band |
| 3 | `Speaking piano` (optional) | 16 | Spectral peaks as piano notes, for DAW playback only |

Events use running status, and note-offs are written as note-on with velocity 0. Controllers and
pitch bend are only written when their value changes; the decoder holds each value until the next
change.

### Track 0: header

A text meta event (type `0x01`) at tick 0 makes the file self-describing:

```
VVC/1 sr=16000 hop_ms=10 nfft=512 bands=24 fmin=80 fmax=7600 db_step=0.75 db_top=31 bend=12 frames=338 samples=54000
```

| Key | Meaning |
|---|---|
| `sr` | Sample rate of the analysed and decoded audio, in Hz |
| `hop_ms` | Frame length in milliseconds |
| `nfft` | Analysis window length in samples (at least 512 and at least two frames) |
| `bands` | Number of mel bands in track 2 |
| `fmin`, `fmax` | Frequency range covered by the mel bands, in Hz |
| `db_step` | Decibels per controller step in track 2 |
| `db_top` | Level in dB that controller value 127 stands for |
| `bend` | Pitch-bend range in semitones |
| `frames`, `samples` | Length of the original in frames and in samples |

A file without a text event starting with `VVC/` is not a VVC file. The decoder refuses it.

### Track 1: voice (channel 1)

| Message | Meaning |
|---|---|
| Program change 52 | "Choir Aahs", so a DAW plays the pitch track with a voice-like sound |
| RPN 0 (CC 101 = 0, CC 100 = 0, CC 6 = `bend`, CC 38 = 0) | Sets the pitch-bend range, then the RPN is closed (CC 101/100 = 127) |
| Note on | A voiced stretch of speech starts. The note is the pitch at that moment rounded to the nearest semitone; velocity is the loudness |
| Note off | Voicing stops (an unvoiced consonant, or silence) |
| Pitch bend (14 bit) | Exact pitch relative to the held note: `value = 8192 + (m − note) / bend × 8192`, where `m = 69 + 12·log2(f0 / 440)` |
| CC 20 | Aperiodicity (how noisy the voice is), `round(√a × 127)` for `a` between 0 (pure tone) and 1 (pure noise) |
| CC 11 | Overall loudness over a 60 dB range. Written for DAW playback; the decoder ignores it |

When the pitch moves more than `bend − 0.5` semitones away from the held note, the encoder ends that
note and starts a new one on the same tick. With the default range of ±12 semitones the pitch
resolution is about 0.3 cents.

### Track 2: spectrum (channels 2 and up)

The spectrum is described by `bands` triangular filters, evenly spaced on the mel scale between `fmin`
and `fmax`. Each band is one controller lane:

```
channel    = 2 + band ÷ 12          (integer division)
controller = 20 + band mod 12       (CC 20–31, undefined in General MIDI)
```

24 bands therefore use CC 20–31 on channels 2 and 3. The value is the band's energy in a
logarithmic scale:

```
floor = db_top − 127 × db_step
value = clamp(round((level_dB − floor) / db_step), 0, 127)
```

With the default step of 0.75 dB the 127 steps span about 95 dB.

### Track 3: speaking piano (channel 16, optional)

Every second frame the encoder finds the six strongest spectral peaks between 80 and 4200 Hz,
rounds each to the nearest semitone, and holds a note for as long as the peak lasts. Velocity follows
the peak's level. The track is program 0 (acoustic grand piano), so a DAW plays it as a "speaking
piano". The decoder ignores it, apart from the separate piano render (`vvc decode --piano`).

## Decoding

1. Read the header, tempo and division. Map every event to its frame.
2. Hold each value until it changes. A frame is voiced while a note is held on channel 1. Its pitch is
   `f0 = 440 × 2^((note + (bend − 8192) / 8192 × range − 69) / 12)`, and its aperiodicity is
   `(CC20 / 127)²`. Unvoiced frames count as pure noise.
3. Band levels are `floor + value × db_step`.
4. Build an excitation signal: a band-limited series of harmonics at `f0`, mixed with white noise
   according to the aperiodicity (with more noise towards high frequencies).
5. Run a channel vocoder. For every frame, measure the excitation's energy in each mel band, and
   scale the band so it matches the stored level. Gains are interpolated across frequency on a log
   scale.
6. Resynthesise with an inverse STFT (Hann window, `nfft` samples, overlap-add).

## Size

The spectrum track dominates the file. For the presets used in the experiments, averaged over 25 s
of speech:

| Preset | Bands | Frame | dB/step | `.mid` | gzipped |
|---|---|---|---|---|---|
| rich | 36 | 10 ms | 0.75 | 70.6 kbit/s | 34.8 kbit/s |
| default | 24 | 10 ms | 0.75 | 48.5 kbit/s | 24.5 kbit/s |
| compact | 24 | 20 ms | 1.5 | 25.2 kbit/s | 12.0 kbit/s |
| small | 16 | 20 ms | 2 | 17.3 kbit/s | 8.2 kbit/s |
| tiny | 12 | 30 ms | 3 | 8.8 kbit/s | 4.4 kbit/s |

For comparison, 16 kHz 16-bit PCM is 256 kbit/s.

## Compatibility notes

- DAWs that re-save MIDI files sometimes drop text events or merge tracks. The decoder needs the
  header text, so keep the original file if you plan to decode it again.
- Files written before the analysis window became automatic store `nfft=512` and decode exactly as
  before.
