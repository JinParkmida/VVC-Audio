# Generate test speech with the built-in Windows voices (16 kHz mono WAV).
param([string]$OutDir = "$PSScriptRoot\..\samples")
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$items = @(
  @{ voice = "Microsoft David Desktop"; name = "david_fox";   text = "The quick brown fox jumps over the lazy dog." },
  @{ voice = "Microsoft Zira Desktop";  name = "zira_premise"; text = "Can we turn human speech into musical instructions and then reliably reconstruct the original message?" },
  @{ voice = "Microsoft David Desktop"; name = "david_harvard"; text = "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background." },
  @{ voice = "Microsoft Zira Desktop";  name = "zira_numbers"; text = "Please call me back at seven thirty, the meeting moved to room nine." }
)
New-Item -ItemType Directory -Force $OutDir | Out-Null
foreach ($it in $items) {
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice($it.voice)
  $path = Join-Path $OutDir ($it.name + ".wav")
  $s.SetOutputToWaveFile($path, $fmt)
  $s.Speak($it.text)
  $s.Dispose()
  Set-Content -Path (Join-Path $OutDir ($it.name + ".txt")) -Value $it.text -NoNewline
  Write-Output $path
}
