# Damaten HexAI exchange system

Damaten turns the HexAI engine into a Windows/Colab training loop:

- Windows plays self-play games forever.
- Windows uploads completed self-play parts to GitHub every day at 08:30.
- Colab pulls those parts, trains the policy/value model on GPU, and pushes
  `models/hex_model.nn`.
- Windows pulls `models/hex_model.nn` every day at 12:00 and uses it for later
  self-play.
- Right after that daily pull, Windows plays the freshly trained model against
  the previous one and records the result, so model strength is tracked over
  time instead of guessed.

The GitHub repository is the exchange point. Large `.tsv` and `.nn` files are
tracked with Git LFS.

> **Current mode: local-only training (Plan B).** The self-play data stream
> outgrew GitHub's free 1 GB LFS quota, so training now runs entirely on the
> Windows machine with PyTorch on CPU — no data leaves the box. The two daily
> GitHub jobs (`Damaten Push Results 0830`, `Damaten Pull Model 1200`) are
> disabled and the Colab path below is kept only for reference. See
> [Local-only training](#local-only-training-plan-b) before the GitHub sections.

## Local-only training (Plan B)

```text
self-play (forever)  ->  hex_selfplay.tsv (local)
                          |
   windows\Train-DamatenLocal.ps1 (run every few days):
     1. pause self-play (free the CPU cores)
     2. train the HEXCNN_V1 conv-ResNet on recent data, warm-started from the
        current model            (colab\colab_train_cnn.py --device cpu)
     3. SPRT gate: candidate vs current model
     4. promote the candidate only if it is measurably stronger
     5. resume self-play
```

This replaces the Colab GPU trainer **and** both daily GitHub jobs. The engine,
self-play, and training all run on one machine; GitHub is optional and only ever
needs code + the ~1.2 MB model, never the `.tsv` data.

### One-time setup (already done on this machine)

```powershell
# per-user Python (no admin / no UAC), then CPU PyTorch
python-3.11.9-amd64.exe /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=0
python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
```

### Run a training cycle

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\Train-DamatenLocal.ps1 `
  -Config .\config\damaten.local.json
```

It pauses self-play, trains a candidate, runs the SPRT promotion gate, promotes
on PASS, and resumes self-play. Flags: `-NoPause` (train alongside self-play),
`-NoPromote` (train a candidate without adopting it), `-Limit` / `-Epochs` /
`-GateIters`.

Settings in `config/damaten.local.json`:

```text
PythonPath          path to python.exe that has torch installed
TrainScript         colab\colab_train_cnn.py (already CPU-capable)
TrainLimit          most-recent positions to train on (default 300000)
TrainEpochs         epochs per run (default 6)
TrainPauseSelfplay  true to free all CPU cores during training
GateIters           MCTS iters for the SPRT promotion gate (default 400)
CandidateModelPath  where the freshly trained candidate is written
```

On a 4-core CPU a run takes roughly 1-3 hours, so run it every few days rather
than daily. Each run warm-starts from the current model, preserving the lineage.
A first local run beat the previous Colab-trained model 30-10 (+191 Elo).

## Repository layout

```text
src/main.cpp                         HexAI C++17 engine
windows/*.ps1                        Windows automation scripts
colab/github_gpu_train_and_push.py   Colab GPU trainer and GitHub pusher
models/hex_model.nn                  latest trained model
selfplay/windows/...                 uploaded Windows self-play parts
logs/windows/...                     Windows automation logs
manifests/windows/...                upload state summaries
```

## 1. First GitHub setup

Run these once in PowerShell from this repository folder:

```powershell
git lfs install
git lfs track "*.nn" "*.tsv" "*.zip"
git add .
git commit -m "initial Damaten HexAI automation"
git push -u origin main
```

If `git push` asks you to sign in, authenticate with GitHub. If it fails with
permission denied, use a GitHub Personal Access Token with repository write
permission.

## 2. Install Windows automation

From this repository folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\Install-DamatenAutomation.ps1
```

This creates:

```text
Damaten SelfPlay Forever      starts at logon and runs self-play forever
Damaten Push Results 0830     daily 08:30 local time, pushes completed parts
Damaten Pull Model 1200       daily 12:00 local time, pulls trained model
```

Start self-play immediately:

```powershell
Start-ScheduledTask -TaskName "Damaten SelfPlay Forever"
```

Run the daily jobs manually:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\Push-DamatenResults.ps1
powershell -ExecutionPolicy Bypass -File .\windows\Pull-DamatenModel.ps1
```

Local settings are saved to:

```text
config/damaten.local.json
```

That file is ignored by Git because it contains machine-specific paths.

## New-vs-old strength check (daily)

The 12:00 pull (`windows\Pull-DamatenModel.ps1`) does more than swap the model:

1. Before overwriting, it copies the model currently in use to
   `hex_model.prev.nn` (the "old" model).
2. After installing the freshly pulled model, if the file actually changed
   (different SHA-256), it runs `windows\Eval-DamatenModels.ps1`, which plays
   the new model (side A) against the old model (side B) with colors alternating
   each game.
3. It logs a one-line verdict and appends a row to `eval_history.csv`.

The log line looks like:

```text
eval result: NEW 61-39 OLD  winrate=61.0% (95% CI 51.0-70.2%)  Elo=+78 [+7, +149]  => NEW stronger (95% CI excludes 50%)
```

`eval_history.csv` columns: `timestamp, games, iters, seed, new_model,
old_model, new_wins, old_wins, winrate, ci_low, ci_high, elo, elo_low,
elo_high, new_as_black, new_as_white, verdict, label`.

The verdict uses a Wilson 95% confidence interval on the win rate:

- `NEW stronger` — the whole interval is above 50%.
- `REGRESSION: new weaker` — the whole interval is below 50% (consider not
  promoting that model).
- `inconclusive` — the interval still spans 50%; play more games.

### Tuning the check

Settings live in `config/damaten.local.json` (runtime defaults are filled in if
a key is missing, so existing configs keep working):

```text
EvalEnabled       true to run the daily check, false to skip it
EvalGames         games per check (default 100; 40 is too few for small gains)
EvalIters         MCTS iterations per move during the check (defaults to Iters)
PrevModelPath     where the previous model is stashed
EvalHistoryPath   CSV history of every check
```

**How many games?** 40 games only resolves large gaps. Rough guide for a 95%
confident, 80%-powered verdict: ~30 games to confirm a 75% win rate, ~200 for
60%, ~780 for 55%. Raise `EvalGames` (or run the script repeatedly and let the
history accumulate) when you care about small improvements.

Run it manually any time:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\Eval-DamatenModels.ps1 `
  -Config .\config\damaten.local.json `
  -NewModel C:\path\to\new.nn -OldModel C:\path\to\old.nn -Games 200
```

### SPRT mode (play until decided)

Fixed-N always plays the same number of games. SPRT (the sequential test used
by computer-chess frameworks) instead plays in batches and stops as soon as the
evidence is conclusive, so easy decisions finish fast and close calls keep
playing. Switch the daily check over by setting `"EvalMode": "sprt"` in
`config/damaten.local.json`.

It tests two hypotheses about the new model's Elo gain over the old one:

- `H0`: gain = `SprtElo0` (default 0 — no real improvement)
- `H1`: gain = `SprtElo1` (default 30 — a meaningful improvement)

After each batch it updates the log-likelihood ratio (LLR) and stops when it
crosses a boundary set by `SprtAlpha`/`SprtBeta` (both default 0.05):

```text
sprt result: PASS: new model accepted as stronger (>= +30 Elo region)  (games=152, NEW 89-63 OLD, LLR=2.98)
```

Outcomes: **PASS** (accept H1, promote), **FAIL** (accept H0, the gain is below
`SprtElo1`), or **INCONCLUSIVE** (hit `SprtMaxGames` first). Per-batch progress
is appended to `sprt_history.csv`. The standalone exit codes are `0` PASS, `3`
FAIL, `4` inconclusive.

SPRT settings in `config/damaten.local.json`:

```text
SprtElo0 / SprtElo1   the two Elo hypotheses (H1 must exceed H0)
SprtAlpha / SprtBeta  false-positive / false-negative rates (default 0.05)
SprtBatch             games per batch before re-checking (default 8, keep even)
SprtMaxGames          hard cap so a run always terminates (default 400)
SprtHistoryPath       CSV of every batch
```

Wider bounds (e.g. `SprtElo1 = 50`) decide in fewer games; tight bounds
(`SprtElo1 = 10`) are stricter but can need thousands of games. Run it directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\Run-DamatenSprt.ps1 `
  -Config .\config\damaten.local.json `
  -NewModel C:\path\to\new.nn -OldModel C:\path\to\old.nn -Elo1 30 -MaxGames 1000
```

> Note: a match runs single-threaded inside the 12:00 job and shares CPU with
> self-play. At full `Iters` a long SPRT/fixed run can take hours; lower
> `EvalIters` (e.g. 400-600) to play more games per hour, or cap with
> `SprtMaxGames` / `EvalGames`. The model swap itself is never delayed — the new
> model is already in place before the check starts.

## 3. Colab GPU training

In Colab, choose:

```text
Runtime > Change runtime type > Hardware accelerator > GPU
```

Then mount Drive and run the Colab trainer. The trainer uses `GITHUB_TOKEN` for
push access.

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["GITHUB_TOKEN"] = "ghp_xxx"
```

```bash
python /content/drive/MyDrive/DamatenTraining/Damaten/colab/github_gpu_train_and_push.py \
  --work-root /content/drive/MyDrive/DamatenTraining \
  --repo-url https://github.com/Koushien552/Damaten.git \
  --branch main \
  --n 9 \
  --epochs 6 \
  --lr 0.003 \
  --batch-size 2048
```

The pushed model appears at:

```text
models/hex_model.nn
```

Windows receives it at 12:00.

## Notes

- GitHub Actions does not provide Colab-class GPU by default. GPU training is
  intended to run in Colab, while GitHub stores the exchanged data.
- Do not push the merged `hex_selfplay.tsv` as a normal Git file. Windows
  uploads small completed `selfplay_parts/*.tsv` files instead.
- Keep `--iters 1200` if you want to preserve the current self-play search
  quality.
