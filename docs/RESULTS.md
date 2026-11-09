# Experiments and results

All numbers come from running the scripts in `tools/` on the material described below. Rerunning
them reproduces the tables, apart from small differences in Whisper's output.

## Material and measures

**Speech.** Five clips, 25.4 s in total:

- four English sentences from two Windows text-to-speech voices (Microsoft David and Zira), made with
  `tools/make_samples.ps1`;
- one real microphone recording in Danish (5 s, one male speaker, about 102 Hz median pitch). Its words:
  *"Hej mit navn det er ‹name›, og jeg tester denne lydfil."* The speaker's name is removed here, and
  the recording is not part of the repository.

All audio is processed at 16 kHz mono. As a 16-bit WAV that is 256 kbit/s.

**Measures.**

| Measure | What it tells you |
|---|---|
| Size | Bitrate of the `.mid` file, raw and gzipped |
| STOI | Short-Time Objective Intelligibility, 0 to 1, comparing decoded audio with the original. Above about 0.9 speech is easy to follow |
| Spectral distance | RMS difference between 40-band log-mel spectra of the original and the decoded audio, in dB. Lower means the two sound more alike |
| Transcript drift | Word error between what Whisper hears in the decoded audio and what it hears in the original. This cancels Whisper's own mistakes, which it makes on both |

The recognizer is OpenAI Whisper *medium*, running locally. Decoded clips are transcribed in the
language Whisper detected for the original. The smaller *small* model misheard even the original
Danish recording, so it was not used for scoring.

## 1. Does speech survive the round trip?

`tools/evaluate.py` encodes every clip with five presets, decodes it, and scores the result.

| Preset | Bands | Frame | dB/step | `.mid` kbit/s | gzipped kbit/s | STOI | Drift |
|---|---|---|---|---|---|---|---|
| rich | 36 | 10 ms | 0.75 | 70.6 | 34.8 | 0.956 | 2 % |
| **default** | 24 | 10 ms | 0.75 | 48.5 | 24.5 | 0.947 | 0 % |
| compact | 24 | 20 ms | 1.5 | 25.2 | 12.0 | 0.926 | 2 % |
| small | 16 | 20 ms | 2 | 17.3 | 8.2 | 0.910 | 2 % |
| tiny | 12 | 30 ms | 3 | 8.8 | 4.4 | 0.862 | 7 % |

Sizes exclude the optional speaking-piano track.

At the default setting Whisper's transcript of every decoded clip is identical to its transcript of
the original. Every 2 % entry is the same single word in the Danish clip:

| Version | Whisper heard |
|---|---|
| Spoken | Hej mit navn det er ‹name›, og jeg tester denne lydfil. |
| Original recording | Hej, mit navn er ‹name›, og jeg tester den løde fil. |
| default | Hej, mit navn er ‹name›, og jeg tester den løde fil. |
| rich, compact, small | Hej, mit navn er ‹name›, og jeg tester den lydede fil. |
| tiny | Hej, mit navn er ‹name› og jeg tæster denne lydfide. |
| speaking-piano track only | (nothing) |

**MIDI is not where quality is lost.** Feeding the decoder the unrounded analysis values, without going
through MIDI at all, gives the same STOI as decoding the `.mid` file (0.947 for both). The 7-bit
controller values and 14-bit pitch bend are fine enough. What the listener loses comes from the
parametric model itself: a synthetic harmonic-plus-noise source shaped by a 24-band envelope. That is
also why the decoded voice sounds like a classic vocoder.

| Signal (default preset) | STOI | Whisper word error vs the spoken text |
|---|---|---|
| Original recording | 1.000 | 12 % |
| Decoder fed unrounded values (no MIDI) | 0.947 | 12 % |
| Decoded from the `.mid` file | 0.947 | 12 % |
| Speaking-piano track alone | 0.467 | 93 % |

The 12 % on the originals is Whisper's own error rate on this material (it writes "730" for "seven
thirty", and "den løde fil" for "denne lydfil").

## 2. Where does the language break?

`tools/stress.py` starts from *compact* and pushes one setting at a time towards the extreme. The last
column shows what Whisper made of the Danish clip.

### Frequency bands

| Bands | gzipped kbit/s | STOI | Spectral distance | Drift | Danish clip heard as |
|---|---|---|---|---|---|
| 24 | 12.0 | 0.926 | 3.8 dB | 2 % | Hej, mit navn er ‹name›, og jeg tester den lydede fil. |
| 16 | 8.9 | 0.911 | 4.5 dB | 2 % | Hej, mit navn er ‹name›, og jeg tester den lydede fil. |
| 12 | 7.0 | 0.895 | 4.9 dB | 7 % | Hej, mit navn er ‹name› og jeg tilstatter din lydfie. |
| 8 | 5.2 | 0.856 | 5.9 dB | 11 % | Hej, min navn er ‹name› og jeg tager stadig en gløde tid. |
| 6 | 4.3 | 0.823 | 7.1 dB | 26 % | Hej, jeg er nu med dig og du har ret skridt støt ind i døden. |
| 4 | 3.4 | 0.770 | 8.6 dB | 56 % | (nothing) |
| 3 | 2.9 | 0.735 | 10.0 dB | 275 % | (nothing) |
| 2 | 2.5 | 0.711 | 12.0 dB | 174 % | (nothing) |

Drift above 100 % means Whisper wrote more words than were spoken: it invented text.

### Frame length

| Frame | gzipped kbit/s | STOI | Spectral distance | Drift | Danish clip heard as |
|---|---|---|---|---|---|
| 10 ms | 21.1 | 0.947 | 3.3 dB | 0 % | Hej, mit navn er ‹name›, og jeg tester den løde fil. |
| 20 ms | 12.0 | 0.926 | 3.8 dB | 2 % | Hej, mit navn er ‹name›, og jeg tester den lydede fil. |
| 30 ms | 8.6 | 0.894 | 4.8 dB | 0 % | Hej, mit navn er ‹name›, og jeg tester den løde fil. |
| 40 ms | 6.8 | 0.855 | 5.9 dB | 3 % | Hej, min navn er ‹name›, og jeg tester den løde fil. |
| 60 ms | 5.0 | 0.783 | 7.3 dB | 15 % | Hej, min navn er ‹name› og jeg ses til den nede fil. |
| 80 ms | 4.0 | 0.716 | 8.6 dB | 26 % | Hej, min navn er ‹name› og jeg er sædstavende med filth. |
| 120 ms | 2.9 | 0.622 | 10.4 dB | 104 % | Ej, jeg er ikke med på asus, og jeg skal selv finde en givet fild. |
| 200 ms | 2.1 | 0.527 | 11.8 dB | 98 % | Er det en mand, der har sådan forvisset en ny uddelt? |

### Level precision

| dB per step | gzipped kbit/s | STOI | Spectral distance | Drift |
|---|---|---|---|---|
| 0.75 | 13.5 | 0.927 | 3.8 dB | 0 % |
| 1.5 | 12.0 | 0.926 | 3.8 dB | 2 % |
| 3 | 9.6 | 0.923 | 3.8 dB | 2 % |
| 6 | 7.3 | 0.911 | 3.9 dB | 0 % |
| 10 | 5.8 | 0.884 | 4.1 dB | 4 % |
| 16 | 4.7 | 0.842 | 4.3 dB | 2 % |
| 24 | 4.0 | 0.792 | 5.1 dB | 14 % |

### Pitch and voicing

Same compact settings, with the decoded instructions altered before synthesis:

| Variant | STOI | Spectral distance | Drift |
|---|---|---|---|
| Pitch snapped to semitones (plain MIDI notes, no pitch bend) | 0.923 | 3.9 dB | 0 % |
| Monotone (one fixed pitch for the whole clip) | 0.892 | 4.2 dB | 2 % |
| Whispered (no pitch at all, noise only) | 0.854 | 4.2 dB | 2 % |
| Always buzzing (voiced/unvoiced information thrown away) | 0.931 | 3.9 dB | 0 % |

### All three settings pushed together

| Setting | gzipped kbit/s | STOI | Drift | Worst clip |
|---|---|---|---|---|
| 16 bands, 30 ms, 6 dB | 4.4 | 0.864 | 7 % | Danish, 27 % |
| 16 bands, 40 ms, 6 dB | 3.6 | 0.828 | 7 % | Danish, 27 % |
| 24 bands, 40 ms, 10 dB | 3.7 | 0.816 | 3 % | Danish, 9 % |
| 12 bands, 30 ms, 10 dB | 3.1 | 0.813 | 26 % | Danish, 118 % |

### What the stress test shows

1. **The words are carried by the spectral envelope.** Flattening the melody, removing the voice
   entirely or discarding the voiced/unvoiced decisions barely changes what Whisper hears. Pitch and
   voicing carry who is speaking and how, and almost none of what is said.
2. **Plain MIDI notes are enough for the pitch.** Snapping the pitch to semitones, which is what a
   piano without pitch bend would play, costs nothing. The speaking-piano track fails for a different
   reason: it has no spectral envelope.
3. **Frequency detail matters most, level precision least.** Band levels survive being rounded to
   16 dB steps. Below 8 bands the words fall apart quickly. Time resolution sits in between: 40 ms
   frames are fine and 80 ms are not.
4. **The real Danish recording breaks first.** At 12 bands it has already lost words (36 % drift on
   that clip) while STOI still rates it 0.87 and the English clips are intact. STOI does not know which
   language is spoken. Whisper has less Danish training data, and real speech is harder than TTS.
5. **Whisper fails confidently.** As the audio degrades it keeps producing fluent sentences, with
   less and less connection to what was said. It does not answer "unclear". Whether people behave
   differently is what the listening test is for.

The best combined setting found, 24 bands with 40 ms frames and 10 dB steps, keeps the words at
**3.7 kbit/s**: 3 % drift on average and 9 % on the hardest clip. Cutting bands instead of precision
costs more. At 16 bands the Danish clip already drifts 27 %.

## 3. A bug the stress test found

The first version used a fixed 32 ms analysis window. From 40 ms frames upwards the windows no longer
overlapped, and intelligibility collapsed for reasons unrelated to the idea itself. The window now
grows with the frame (at least two frames long):

| Frame | Bands | STOI, fixed 32 ms window | STOI, automatic window |
|---|---|---|---|
| 10 ms | 24 | 0.947 | 0.947 |
| 20 ms | 24 | 0.925 | 0.927 |
| 30 ms | 12 | 0.842 | 0.865 |
| 40 ms | 24 | 0.337 | 0.856 |
| 60 ms | 24 | 0.266 | 0.784 |
| 100 ms | 24 | 0.223 | 0.670 |

All results above use the automatic window.

## 4. Do people and Whisper break at the same point?

`tools/make_listening_test.py` builds a blind test, and `python -m vvc listen` runs it.

- **Sentences.** 24 short everyday Danish sentences, spoken by the Windows voice *Microsoft Helle*.
  Helle only works through the WinRT speech API (`tools/tts_winrt.ps1`). The older System.Speech API
  lists her but silently falls back to an English voice.
- **Conditions.** Twelve, with two sentences each: untouched, default, compact, 8, 6, 4 and 3 bands,
  80, 120 and 200 ms frames, whispered, and the speaking piano alone. Each sentence is heard once, in
  one condition, in random order behind a random file name.
- **Answers.** The listener types what they heard and rates how sure they are from 1 to 5. Replays and
  response times are recorded.
- **Whisper on the same audio.** Whisper transcribes every clip when the test is built and records its
  confidence (average token probability). On the current set it gets 100 % of the words in the
  untouched clips and 93 % at the default setting. It reaches 0 % at 4 and 3 bands, and at 3 bands it
  reports 92 % confidence while being completely wrong.
- **Scoring.** Words correct for the listener and for Whisper, per condition, with lenient matching so
  small typos count. The report lists clips only one of them understood, and cases where either was
  sure (4 or 5 out of 5, or above 50 % confidence) but got less than half the words.
- **Nothing is destroyed.** The test screen reads only a playlist of clip ids. The answer key, the clean
  recordings, the exact `.mid` file behind every clip and copies of the clips are kept read-only in
  `out/listening/sealed/<run>/`, with SHA-256 checksums that are verified before scoring.

Human results are not in yet. A good test needs listeners who have not seen the sentence list in
`tools/make_listening_test.py`.

## Reproducing

```bash
powershell -ExecutionPolicy Bypass -File tools/make_samples.ps1   # English test sentences (Windows)
python tools/evaluate.py                                          # section 1 -> out/results.md
python tools/stress.py                                            # section 2 -> out/stress.md
python tools/make_listening_test.py                               # section 4, then: python -m vvc listen
```

`tools/stress.py` reads Whisper's transcripts of the originals from `out/results.json`, so run
`evaluate.py` first.
