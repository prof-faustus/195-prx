# Resume the d=5 theta sweep if it is neither already running nor already complete.
# Safe to run repeatedly (e.g. from a logon scheduled task): the workers resume from
# their JSON checkpoints with restored RNG state, so re-launching only continues the run.
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot           # project root (parent of experiments/)
$py   = Join-Path $proj ".venv\Scripts\python.exe"
$log  = Join-Path $proj "results\d5sweep_run.log"

# 1. already running? (any python with this run's tag in its command line)
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -match 'd5sweep' }
if ($running) { Write-Output "$(Get-Date -Format s)  d5sweep already running (PID $($running.ProcessId -join ',')); skip."; exit 0 }

# 2. already complete? (all four per-GPU checkpoints report _meta.done = true)
$allDone = $true
foreach ($g in 0..3) {
    $c = Join-Path $proj "results\d5_ckpt_d5sweep_gpu$g.json"
    if (-not (Test-Path $c)) { $allDone = $false; break }
    $j = Get-Content $c -Raw | ConvertFrom-Json
    if (-not $j._meta.done) { $allDone = $false; break }
}
if ($allDone) { Write-Output "$(Get-Date -Format s)  d5sweep already complete; nothing to resume."; exit 0 }

# 3. resume: relaunch the EXACT same orchestrator command (checkpoints make it idempotent)
$thetas = & $py -c "print(' '.join(f'{0.05+0.01*i:.2f}' for i in range(46)))"
$argline = "-u `"$proj\experiments\run_d5_multigpu.py`" --gpus 0 1 2 3 --distance 5 " +
           "--thetas $thetas --coh-shots 8000 --pauli-shots 120000 --tag d5sweep --plot " +
           "--overlay `"$proj\results\coherent_scaling_d5_multigpu_d3sweep.csv`""
Write-Output "$(Get-Date -Format s)  resuming d5sweep..."
Start-Process -FilePath $py -ArgumentList $argline -WorkingDirectory $proj -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError ($log + ".err")
Write-Output "$(Get-Date -Format s)  d5sweep resume launched (log: $log)."
