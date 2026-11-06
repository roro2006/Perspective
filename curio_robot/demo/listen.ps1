# Windows' built-in, offline speech recognizer on a fixed phrase list (T5-33, demo/voice.py).
# Prints one line per recognition: HEARD<TAB>confidence<TAB>text. -WaveFile reads a recording
# instead of the microphone (used by the tests and the self-check).
param(
    [Parameter(Mandatory = $true)][string]$PhraseFile,
    [string]$WaveFile = ""
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$culture = New-Object System.Globalization.CultureInfo("en-US")
$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine($culture)
$choices = New-Object System.Speech.Recognition.Choices
foreach ($line in [System.IO.File]::ReadAllLines($PhraseFile)) {
    if ($line.Trim()) { [void]$choices.Add($line.Trim()) }
}
$gb = New-Object System.Speech.Recognition.GrammarBuilder
$gb.Culture = $culture
$gb.Append($choices)
$rec.LoadGrammar((New-Object System.Speech.Recognition.Grammar($gb)))
if ($WaveFile) { $rec.SetInputToWaveFile($WaveFile) } else { $rec.SetInputToDefaultAudioDevice() }
$rec.EndSilenceTimeout = [TimeSpan]::FromMilliseconds(500)
[Console]::Out.WriteLine("READY")
[Console]::Out.Flush()
while ($true) {
    try {
        $r = $rec.Recognize()
    } catch {
        break  # end of the wave file, or the device went away
    }
    if ($r -ne $null) {
        [Console]::Out.WriteLine(("HEARD`t{0:F2}`t{1}" -f $r.Confidence, $r.Text))
        [Console]::Out.Flush()
    } elseif ($WaveFile) {
        break
    }
}
