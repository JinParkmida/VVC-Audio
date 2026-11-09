# Samples

Put speech to test here: `name.wav` (any sample rate, mono or stereo) and, optionally, `name.txt`
with the exact words spoken. With a `.txt` file the evaluation can report word error rates.
Audio in this folder is ignored by git, so recordings stay on your machine.

Generate the four English test sentences used in the experiments (Windows only):

```powershell
powershell -ExecutionPolicy Bypass -File tools\make_samples.ps1
```

Or record your own voice from the terminal app (`python -m vvc`, then `r`). Recordings are saved
here as `rec_<time>.wav`, with whatever you type in the *Words* field as the `.txt`.
