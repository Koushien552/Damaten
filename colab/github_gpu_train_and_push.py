#!/usr/bin/env python3
"""Colab GPU trainer that uses GitHub as the exchange point.

Flow:
1. Pull Koushien552/Damaten from GitHub.
2. Merge newly uploaded Windows self-play TSV parts into a Drive-local TSV.
3. Train the HexAI TinyNet with PyTorch/CUDA.
4. Commit and push models/hex_model.nn back to GitHub.

Adds:
- JST-based daily run limit
- At most 2 successful trainings per day
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")
TRAIN_STATE_REL = Path("models/training_state.json")


def utc_now() -> str:
    return datetime.now(ZoneInfo("UTC")).isoformat(timespec="seconds")


def jst_today() -> str:
    return datetime.now(JST).date().isoformat()


def authed_url(url: str, token: str) -> str:
    if not token or not url.startswith("https://github.com/"):
        return url
    return "https://x-access-token:" + token + "@" + url.removeprefix("https://")


def run(cmd: list[str], cwd: Path | None = None, secret: str = "") -> None:
    shown = " ".join(cmd)
    if secret:
        shown = shown.replace(secret, "***")
    print("$", shown, flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {shown}")


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_train_state(repo_dir: Path) -> dict:
    return load_json(repo_dir / TRAIN_STATE_REL, {"date": "", "count": 0})


def clone_or_pull(args: argparse.Namespace, token: str) -> Path:
    repo_dir = Path(args.work_root) / "Damaten"
    url = authed_url(args.repo_url, token)

    if (repo_dir / ".git").exists():
        run(["git", "remote", "set-url", "origin", url], cwd=repo_dir, secret=token)
        run(["git", "fetch", "origin", args.branch], cwd=repo_dir, secret=token)
        run(["git", "checkout", args.branch], cwd=repo_dir, secret=token)
        run(["git", "pull", "--rebase", "--autostash", "origin", args.branch], cwd=repo_dir, secret=token)
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--branch", args.branch, url, str(repo_dir)], secret=token)

    run(["git", "lfs", "install"], cwd=repo_dir, secret=token)
    run(["git", "lfs", "pull"], cwd=repo_dir, secret=token)
    run(["git", "config", "user.name", args.git_user_name], cwd=repo_dir, secret=token)
    run(["git", "config", "user.email", args.git_user_email], cwd=repo_dir, secret=token)
    return repo_dir


def merge_new_selfplay_parts(repo_dir: Path, data_path: Path, manifest_path: Path) -> tuple[int, int]:
    manifest = load_json(manifest_path, {"files": []})
    seen = set(manifest.get("files", []))

    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists() or data_path.stat().st_size == 0:
        data_path.write_text("# HEXSELFPLAY_V1\n# n\tplayer\tboard\tpolicy\twinner\tvalue\n", encoding="utf-8")

    new_files = 0
    added_lines = 0

    with data_path.open("a", encoding="utf-8") as out:
        for part in sorted((repo_dir / "selfplay").glob("**/*.tsv")):
            if not part.is_file():
                continue
            rel = part.relative_to(repo_dir).as_posix()
            key = f"{rel}:{part.stat().st_size}"
            if key in seen:
                continue
            with part.open("r", encoding="utf-8", errors="replace") as inp:
                for line in inp:
                    if not line.strip() or line.startswith("#"):
                        continue
                    out.write(line)
                    added_lines += 1
            seen.add(key)
            new_files += 1

    manifest["files"] = sorted(seen)
    manifest["updated_at_utc"] = utc_now()
    save_json(manifest_path, manifest)
    return new_files, added_lines


def train_model(args: argparse.Namespace, repo_dir: Path, data_path: Path, model_in: Path, model_out: Path) -> None:
    trainer = repo_dir / "colab" / "colab_train_torch.py"
    if not trainer.exists():
        raise FileNotFoundError(f"trainer not found: {trainer}")

    cmd = [
        sys.executable,
        str(trainer),
        "--n",
        str(args.n),
        "--data",
        str(data_path),
        "--model-out",
        str(model_out),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--batch-size",
        str(args.batch_size),
        "--optimizer",
        args.optimizer,
    ]
    if model_in.exists():
        cmd += ["--model-in", str(model_in)]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]
    if args.recent:
        cmd += ["--recent"]

    run(cmd)


def push_model(
    args: argparse.Namespace,
    repo_dir: Path,
    local_model: Path,
    token: str,
    merged_files: int,
    merged_lines: int,
    prev_state: dict,
) -> None:
    repo_model = repo_dir / "models" / "hex_model.nn"
    repo_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_model, repo_model)

    today = jst_today()
    prev_count = int(prev_state.get("count", 0)) if prev_state.get("date") == today else 0

    state = {
        "updated_at_utc": utc_now(),
        "date": today,
        "count": prev_count + 1,
        "trainer": "colab-pytorch-cuda",
        "n": args.n,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "merged_new_files": merged_files,
        "merged_new_positions": merged_lines,
        "model_path": "models/hex_model.nn",
    }
    save_json(repo_dir / TRAIN_STATE_REL, state)

    run(["git", "lfs", "track", "*.nn", "*.tsv", "*.zip"], cwd=repo_dir, secret=token)
    run(["git", "add", ".gitattributes", "models/hex_model.nn", "models/training_state.json"], cwd=repo_dir, secret=token)

    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_dir), text=True)
    if not status.strip():
        print("No model changes to push.", flush=True)
        return

    run(["git", "commit", "-m", "colab gpu trained model"], cwd=repo_dir, secret=token)
    run(["git", "push", "origin", args.branch], cwd=repo_dir, secret=token)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HexAI on Colab GPU and push model to GitHub")
    p.add_argument("--repo-url", default="https://github.com/Koushien552/Damaten.git")
    p.add_argument("--branch", default="main")
    p.add_argument("--work-root", default="/content/drive/MyDrive/DamatenTraining")
    p.add_argument("--github-token-env", default="GITHUB_TOKEN")
    p.add_argument("--git-user-name", default="Damaten Colab GPU")
    p.add_argument("--git-user-email", default="damaten-colab-gpu@local")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--recent", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    token = os.environ.get(args.github_token_env, "")
    if not token:
        print(
            f"WARNING: {args.github_token_env} is not set. Pull may work, push will fail without GitHub credentials.",
            flush=True,
        )

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    repo_dir = clone_or_pull(args, token)

    # 1日2回までの制限
    state = load_train_state(repo_dir)
    today = jst_today()
    if state.get("date") == today and int(state.get("count", 0)) >= 2:
        print(f"skip: already trained {state.get('count', 0)} times today ({today} JST)", flush=True)
        return 0

    data_path = work_root / "hex_selfplay.tsv"
    manifest_path = work_root / "merged_manifest.json"
    merged_files, merged_lines = merge_new_selfplay_parts(repo_dir, data_path, manifest_path)
    print(f"merged_new_files={merged_files} merged_new_positions={merged_lines}", flush=True)

    local_model = work_root / "hex_model.nn"
    repo_model = repo_dir / "models" / "hex_model.nn"
    if repo_model.exists():
        shutil.copy2(repo_model, local_model)

    next_model = work_root / "hex_model_next.nn"
    train_model(args, repo_dir, data_path, local_model, next_model)
    os.replace(next_model, local_model)

    push_model(args, repo_dir, local_model, token, merged_files, merged_lines, state)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
