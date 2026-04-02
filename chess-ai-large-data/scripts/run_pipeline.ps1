param(
  [string]$Month = "2025-01",
  [int]$MaxGames = 500000,
  [switch]$SkipDownload,
  [switch]$SkipConvert,
  [switch]$SkipPrecompute,
  [string]$TrainConfig = "configs/train_30min_fast.yaml"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $projectRoot "src"

& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $projectRoot "requirements.txt")

# RTX 50-series needs a newer CUDA build than many default wheels.
& $py -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

if (-not $SkipDownload) {
  & $py -m chess_ai.data.download_lichess --month $Month --out data/raw
}

$inputFile = "data/raw/lichess_db_standard_rated_$Month.pgn.zst"
if (-not $SkipConvert) {
  & $py -m chess_ai.data.pgn_to_parquet --input $inputFile --output data/processed/train --max-games $MaxGames
}

if (-not $SkipPrecompute) {
  & $py -m chess_ai.data.precompute_planes_parquet --input-dir data/processed/train --output-dir data/processed/train_planes --workers 8 --overwrite
  if (Test-Path "data/processed/valid") {
    & $py -m chess_ai.data.precompute_planes_parquet --input-dir data/processed/valid --output-dir data/processed/valid_planes --workers 8 --overwrite
  }
}

& $py -m chess_ai.train --config $TrainConfig
