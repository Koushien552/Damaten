# Damaten Colab GPU trainer

Use this side after Windows has pushed self-play parts to GitHub.

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set a GitHub token with repo write permission. In Colab you can use Secrets,
or temporarily:

```python
import os
os.environ["GITHUB_TOKEN"] = "ghp_xxx"
```

Then run:

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

The script pushes:

```text
models/hex_model.nn
models/training_state.json
```

Windows receives `models/hex_model.nn` at 12:00.
