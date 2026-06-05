# Damaten Colab GPU trainer

Use this side after Windows has pushed self-play parts to GitHub.

This version does not use `drive.mount()`. It works in Colab's local runtime
directory and uses GitHub for persistence.

Set a GitHub token with repo write permission. In Colab you can use Secrets,
or temporarily:

```python
import os
os.environ["GITHUB_TOKEN"] = "ghp_xxx"
```

Then run (CNN ResNet — the default):

```bash
git clone https://github.com/Koushien552/Damaten.git /content/DamatenTraining/Damaten

python /content/DamatenTraining/Damaten/colab/github_gpu_train_and_push.py \
  --work-root /content/DamatenTraining \
  --repo-url https://github.com/Koushien552/Damaten.git \
  --branch main \
  --n 9 \
  --arch cnn \
  --channels 32 --blocks 4 --symmetry full
```

To keep training the old MLP instead, pass `--arch mlp`.

The script pushes:

```text
models/hex_model.nn          (HEXCNN_V1 when --arch cnn, HEXNN_V1 when --arch mlp)
models/training_state.json
```

Windows receives `models/hex_model.nn` at 12:00.

## Architecture switch (mlp -> cnn): deploy order matters

`models/hex_model.nn` is a single file whose format the engine auto-detects from
its header (`HEXNN_V1` = MLP, `HEXCNN_V1` = CNN). When you switch `--arch` the
exchanged file changes format, so:

1. **Rebuild and deploy the Windows engine first** from the current `src/main.cpp`
   (run `..\build.ps1`). Only this build can load `HEXCNN_V1`; an older Windows
   binary would fail to load the new model and silently fall back to heuristics.
2. The first `--arch cnn` run ignores the existing MLP model and trains a fresh
   CNN (format mismatch is detected; resume only happens within the same format).
3. CNN inference is heavier than the MLP. Windows self-play will be slower per
   move — consider lowering `--iters` for self-play (a strong CNN policy needs
   fewer simulations). Start with a small net (`--channels 16 --blocks 3`) if
   self-play throughput matters, then scale up.

## Verify the CNN export matches the C++ reader (do this once)

```python
import torch; from pathlib import Path
import colab_train_cnn as T
m = T.load_hexcnn(Path('models/hex_model.nn')).eval()
x = torch.tensor([T.make_planes([0]*81, 1, 9)], dtype=torch.float32).view(1, 3, 9, 9)
with torch.no_grad():
    logits, v = m(x)
print('value', float(v))
print('logits[0:9]', [round(t, 5) for t in logits[0][:9].tolist()])
```

```powershell
# On Windows, same empty board / Black to move:
.\build\hexai.exe eval --n 9 --board ("." * 81) --player 1 --model models\hex_model.nn
```

The `value` and `logits` must agree to ~1e-4. If they do, the weight layout is
correct end to end.
