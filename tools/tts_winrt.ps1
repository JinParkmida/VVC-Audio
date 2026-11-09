# Speak each line of a UTF-8 text file with a Windows "OneCore" voice via the WinRT
# speech API, writing <OutDir>\s00.wav, s01.wav, ...
# Needed for voices like "Microsoft Helle" (da-DK): they are listed by the older
# System.Speech API but cannot be selected there (it silently falls back to English).
# Must run in Windows PowerShell 5.1 (powershell.exe), which can load WinRT types.
param([Parameter(Mandatory)][string]$ListFile, [Parameter(Mandatory)][string]$OutDir, [string]$Lang = "da-DK")
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
  $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); $task.Wait(); $task.Result }

$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.Language -eq $Lang } | Select-Object -First 1
if (-not $voice) { throw "No WinRT voice installed for $Lang" }
$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$synth.Voice = $voice
$lines = [IO.File]::ReadAllLines($ListFile, [Text.Encoding]::UTF8)
for ($i = 0; $i -lt $lines.Count; $i++) {
  $stream = Await ($synth.SynthesizeTextToStreamAsync($lines[$i])) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
  $reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
  $null = Await ($reader.LoadAsync([uint32]$stream.Size)) ([uint32])
  $bytes = New-Object byte[] $stream.Size
  $reader.ReadBytes($bytes)
  [IO.File]::WriteAllBytes((Join-Path $OutDir ('s{0:D2}.wav' -f $i)), $bytes)
}
"voice: " + $voice.DisplayName + " (" + $voice.Language + "), " + $lines.Count + " lines"
