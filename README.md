# Damaten HexAI exchange system

Damaten turns the HexAI engine into a Windows/Colab training loop:

- Windows plays self-play games forever.
- Windows uploads completed self-play parts to GitHub every day at 08:30.
- Colab pulls those parts, trains the policy/value model on GPU, and pushes
  `models/hex_model.nn`.
- Windows pulls `models/hex_model.nn` every day at 12:00 and uses it for later
  self-play.

The GitHub repository is the exchange point. Large `.tsv` and `.nn` files are
tracked with Git LFS.

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
