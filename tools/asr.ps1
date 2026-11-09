# Transcribe WAV files with the offline Windows en-US dictation recognizer.
# Prints one line per file:  <path>\t<transcript>
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Files)
Add-Type -AssemblyName System.Speech
foreach ($f in $Files) {
  $rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine([System.Globalization.CultureInfo]::new("en-US"))
  $rec.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
  $rec.SetInputToWaveFile((Resolve-Path $f).Path)
  $words = @()
  while ($true) {
    try { $r = $rec.Recognize() } catch { break }
    if ($null -eq $r) { break }
    $words += $r.Text
  }
  $rec.Dispose()
  Write-Output ("{0}`t{1}" -f $f, ($words -join " "))
}
