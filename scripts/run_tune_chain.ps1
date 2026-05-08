# US-022..US-025 (Option C hybrid) — fast Optuna tune + top-3 full-convergence validation.
#
# Pipeline (sequential, fail-soft):
#   Stage 1 — fast tune × 4 CNN pairs (5 ep / 2k train / 1k val):
#       Skips a pair if its SQLite study already has >= 20 completed trials.
#   Stage 2 — top-3 full-convergence validation × 4 CNN pairs
#       (60 ep / 10k train / 5k val / patience=10):
#       Skips a pair if best_hparams JSON has validated_at_full_convergence=true.
#       Per-trial cache at artifacts/validation/{m}_{d}_rank{N}.json.
#
# Encoding contract:
#   - PYTHONIOENCODING=utf-8 forces python's stdout/stderr to UTF-8.
#   - Per-pair python invocation goes through `cmd /c ... >> log 2>&1` so PS
#     never wraps the I/O (avoids the UTF-16 / ErrorRecord issues from `*>>`).
#   - Markers written via Out-File -Encoding utf8.
#   - Result: log file is pure UTF-8, grep-clean, no "python.exe :" prefixes.
#
# Output: .venv-gpu/tune_chain.log (gitignored).

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repo  = "J:\Object-Classification-in-Low-Resolution-THz-Imagery--"
$py    = Join-Path $repo ".venv-gpu\Scripts\python.exe"
$tune  = Join-Path $repo "tune_all.py"
$valid = Join-Path $repo "scripts\validate_top3.py"
$log   = Join-Path $repo ".venv-gpu\tune_chain.log"

function Write-Marker {
    param([string]$Text)
    "$Text" | Out-File -Append -Encoding utf8 -FilePath $log
}

function Get-CompletedTrials {
    param([string]$StudyName)
    $code = "import optuna; s=optuna.load_study(study_name='$StudyName', storage='sqlite:///artifacts/optuna_thz.db'); print(sum(1 for t in s.trials if t.state.is_finished()))"
    $out = & $py -c $code 2>$null
    if ($LASTEXITCODE -eq 0 -and $out -match '^\s*(\d+)\s*$') {
        return [int]$matches[1]
    }
    return 0
}

function Invoke-Logged {
    param([string]$ScriptPath, [string[]]$ArgList)
    $cmdline = "`"$py`" `"$ScriptPath`" $($ArgList -join ' ') >> `"$log`" 2>&1"
    & cmd /c $cmdline
    return $LASTEXITCODE
}

Write-Marker "=== TUNE-CHAIN START $(Get-Date -Format 'o') ==="
Write-Marker "=== Stage 1: fast Optuna tune (5ep / 2k train / 1k val) ==="

$pairs = @(
    @{ us = "US-022"; model = "resnet50";    dataset = "cifar10" },
    @{ us = "US-023"; model = "densenet121"; dataset = "cifar10" },
    @{ us = "US-024"; model = "resnet50";    dataset = "mnist"   },
    @{ us = "US-025"; model = "densenet121"; dataset = "mnist"   }
)

foreach ($p in $pairs) {
    $tag = "$($p.model) x $($p.dataset)"
    $study = "$($p.model)_$($p.dataset)_L3"
    Write-Marker "=== $($p.us) [$tag] start $(Get-Date -Format 'o') ==="

    $existing = Get-CompletedTrials -StudyName $study
    if ($existing -ge 20) {
        Write-Marker "=== $($p.us) SKIP: study has $existing completed trials (>=20) ==="
        continue
    }

    $rc = Invoke-Logged -ScriptPath $tune -ArgList @("--n-trials","20","--model",$p.model,"--dataset",$p.dataset)
    Write-Marker "=== $($p.us) done exit=$rc at $(Get-Date -Format 'o') ==="
}

Write-Marker "=== Stage 2: top-3 full-convergence validation (60ep / 10k train / 5k val / patience=10) ==="

foreach ($p in $pairs) {
    $tag = "$($p.model) x $($p.dataset)"
    Write-Marker "=== VALIDATE [$tag] start $(Get-Date -Format 'o') ==="
    $rc = Invoke-Logged -ScriptPath $valid -ArgList @("--model",$p.model,"--dataset",$p.dataset)
    Write-Marker "=== VALIDATE [$tag] done exit=$rc at $(Get-Date -Format 'o') ==="
}

Write-Marker "=== TUNE-CHAIN DONE $(Get-Date -Format 'o') ==="
