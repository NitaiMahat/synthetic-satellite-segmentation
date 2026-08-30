$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runDir = Join-Path $projectRoot "models\baseline_v1_complete"
$checkpoint = Join-Path $runDir "best_checkpoint.pt"

Set-Location $projectRoot
& $python -u "experiments\train_baseline.py" --epochs 15 --batch-size 1 --image-size 512 --learning-rate 1e-4 --base-channels 16 --seed 42 --output-dir $runDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python "evaluation\visualize_baseline.py" --checkpoint-path $checkpoint --output-path "evaluation\baseline_v1_complete_validation_examples.png"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python "evaluation\write_baseline_note.py" --run-dir $runDir --output-path "evaluation\baseline_v1_complete_note.md"
exit $LASTEXITCODE
