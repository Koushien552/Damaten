#!/usr/bin/env python3
"""GPU trainer for HexAI's convolutional residual network (HEXCNN_V1).

This is the CNN counterpart of ``colab_train_torch.py``. The self-play engine is
still the portable C++ program, but this trainer uses PyTorch/CUDA to train an
AlphaZero-style conv ResNet and writes the exact ``HEXCNN_V1`` text format that
the C++ engine loads (``ConvNet`` in ``src/main.cpp``).

Architecture (must stay in lock-step with the C++ ``ConvNet``):
  input planes [0]=my stones, [1]=opp stones, [2]=color (1.0 if Black to move)
  trunk:  Conv2d(in_planes->C, 3x3, pad 1) + ReLU
  B blocks: y = ReLU(x + Conv2d(ReLU(Conv2d(x))))     (no BatchNorm)
  policy: Conv2d(C->2, 1x1) + ReLU -> flatten -> Linear(2*nn -> nn)
  value:  Conv2d(C->1, 1x1) + ReLU -> flatten -> Linear(nn -> H) + ReLU
                                              -> Linear(H -> 1) + tanh

Data augmentation uses the two Hex board symmetries (order-4 group):
  * 180-degree rotation                (same player, same colors)
  * transpose + colour/role swap       (mirror across the main diagonal)
Both are exact symmetries of Hex, so they multiply the training data ~4x.

Cross-check against the C++ reader with:
  hexai eval --n 9 --board <81 chars> --player 1 --model hex_cnn.nn
and compare the printed value/logits with this model's output on the same input.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


BLACK = 1
WHITE = 2


def other_player(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def parse_board(text: str, nn_: int):
    if len(text) != nn_:
        return None
    board = []
    for ch in text:
        if ch in ".0":
            board.append(0)
        elif ch in "Bb1":
            board.append(BLACK)
        elif ch in "Ww2":
            board.append(WHITE)
        else:
            return None
    return board


def parse_policy(text: str, nn_: int):
    policy = [0.0] * nn_
    for item in text.split(","):
        if not item or ":" not in item:
            continue
        left, right = item.split(":", 1)
        try:
            idx = int(left)
            prob = float(right)
        except ValueError:
            continue
        if 0 <= idx < nn_:
            policy[idx] = prob
    total = sum(policy)
    if total > 0.0:
        policy = [x / total for x in policy]
    return policy


def make_planes(board, player: int, n: int):
    """Return a [3, n, n] float list of planes matching the C++ make_planes."""
    nn_ = n * n
    opp = other_player(player)
    mine = [0.0] * nn_
    theirs = [0.0] * nn_
    for i, cell in enumerate(board):
        if cell == player:
            mine[i] = 1.0
        elif cell == opp:
            theirs[i] = 1.0
    color = 1.0 if player == BLACK else 0.0
    colorp = [color] * nn_
    return [mine, theirs, colorp]


def tail_data_lines(path: Path, limit: int, block_size: int = 1024 * 1024):
    data = bytearray()
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        line_count = 0
        while pos > 0 and line_count <= limit:
            read_size = min(block_size, pos)
            pos -= read_size
            f.seek(pos)
            chunk = f.read(read_size)
            data[:0] = chunk
            line_count = data.count(b"\n")
    lines = data.decode("utf-8", errors="replace").splitlines()
    lines = [line for line in lines if line and not line.startswith("#")]
    return lines[-limit:]


def load_examples(path: Path, n: int, limit: int = 0, recent: bool = False):
    """Load self-play rows into compact float32 numpy arrays.

    Returns (X[N,3,n,n], P[N,nn], V[N,1]). Filling pre-allocated numpy arrays
    (instead of Python lists of lists) keeps memory around ~1.4 GB for ~1M
    positions instead of ~10 GB, which is what made Colab OOM-kill the trainer.
    """
    nn_ = n * n
    if recent and limit > 0:
        rows = tail_data_lines(path, limit)
    else:
        rows = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                rows.append(line.rstrip("\n"))
                if limit > 0 and len(rows) >= limit:
                    break

    cap = len(rows)
    X = np.zeros((cap, 3, n, n), dtype=np.float32)
    P = np.zeros((cap, nn_), dtype=np.float32)
    V = np.zeros((cap, 1), dtype=np.float32)
    valid = 0
    for line in rows:
        cols = line.split("\t")
        if len(cols) < 6:
            continue
        try:
            row_n = int(cols[0])
            player = int(cols[1])
            value = float(cols[5])
        except ValueError:
            continue
        if row_n != n:
            continue
        board = parse_board(cols[2], nn_)
        if board is None:
            continue
        b = np.asarray(board, dtype=np.int16)
        opp = other_player(player)
        X[valid, 0] = (b == player).astype(np.float32).reshape(n, n)
        X[valid, 1] = (b == opp).astype(np.float32).reshape(n, n)
        if player == BLACK:
            X[valid, 2] = 1.0
        P[valid] = parse_policy(cols[3], nn_)
        V[valid, 0] = value
        valid += 1

    if valid == 0:
        raise RuntimeError(f"no examples loaded from {path}")
    del rows
    return X[:valid], P[:valid], V[:valid]


class HexDataset(Dataset):
    """Serves (x, policy, value) with on-the-fly Hex symmetry augmentation, so the
    4x expanded set never has to be materialised in memory at once.

    Symmetries (all value-preserving): identity, 180-degree rotation, transpose
    with colour/role swap (which flips the constant colour plane), and their
    composition. A random one is applied per item for training samples.
    """

    def __init__(self, X, P, V, idx, n: int, symmetry: str, train: bool):
        self.X = X
        self.P = P
        self.V = V
        self.idx = idx
        self.n = n
        self.symmetry = symmetry
        self.train = train

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = int(self.idx[k])
        x = torch.from_numpy(self.X[i].copy())                       # [3,n,n]
        p = torch.from_numpy(self.P[i].copy()).view(self.n, self.n)  # [n,n]
        v = torch.from_numpy(self.V[i].copy())                       # [1]
        if self.train and self.symmetry != "none":
            choices = ("id", "rot") if self.symmetry == "rot180" else ("id", "rot", "tr", "rot_tr")
            t = random.choice(choices)
            if t in ("rot", "rot_tr"):
                x = torch.flip(x, dims=[1, 2])
                p = torch.flip(p, dims=[0, 1])
            if t in ("tr", "rot_tr"):
                x = x.transpose(1, 2).contiguous()
                x[2] = 1.0 - x[2]            # colour plane flips under role swap
                p = p.t().contiguous()
        return x, p.reshape(-1), v


class HexResNet(nn.Module):
    def __init__(self, n: int, channels: int, blocks: int, in_planes: int = 3, val_hidden: int = 64) -> None:
        super().__init__()
        self.n = n
        self.nn = n * n
        self.C = channels
        self.B = blocks
        self.in_planes = in_planes
        self.val_hidden = val_hidden
        self.conv_in = nn.Conv2d(in_planes, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            nn.ModuleDict({
                "c1": nn.Conv2d(channels, channels, 3, padding=1),
                "c2": nn.Conv2d(channels, channels, 3, padding=1),
            })
            for _ in range(blocks)
        )
        self.pconv = nn.Conv2d(channels, 2, 1)
        self.pfc = nn.Linear(2 * self.nn, self.nn)
        self.vconv = nn.Conv2d(channels, 1, 1)
        self.vfc1 = nn.Linear(self.nn, val_hidden)
        self.vfc2 = nn.Linear(val_hidden, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv_in(x))
        for blk in self.blocks:
            h = F.relu(blk["c1"](x))
            h2 = blk["c2"](h)
            x = F.relu(x + h2)
        p = F.relu(self.pconv(x))
        logits = self.pfc(p.flatten(1))
        v = F.relu(self.vconv(x))
        v = F.relu(self.vfc1(v.flatten(1)))
        v = torch.tanh(self.vfc2(v))
        return logits, v


def flat(t: torch.Tensor):
    return t.detach().cpu().contiguous().view(-1).tolist()


def fmt_vec(vals):
    return str(len(vals)) + " " + " ".join(f"{float(x):.9g}" for x in vals)


def save_hexcnn(model: HexResNet, path: Path) -> None:
    # NB: do not move the model to CPU here -- flat() already copies each tensor
    # to CPU, and this is called mid-training (best-checkpoint) so the model must
    # stay on its current device.
    lines = [f"HEXCNN_V1 {model.n} {model.C} {model.B} {model.in_planes} {model.val_hidden}"]
    lines.append(fmt_vec(flat(model.conv_in.weight)))
    lines.append(fmt_vec(flat(model.conv_in.bias)))
    for blk in model.blocks:
        lines.append(fmt_vec(flat(blk["c1"].weight)))
        lines.append(fmt_vec(flat(blk["c1"].bias)))
        lines.append(fmt_vec(flat(blk["c2"].weight)))
        lines.append(fmt_vec(flat(blk["c2"].bias)))
    lines.append(fmt_vec(flat(model.pconv.weight)))
    lines.append(fmt_vec(flat(model.pconv.bias)))
    lines.append(fmt_vec(flat(model.pfc.weight)))
    lines.append(fmt_vec(flat(model.pfc.bias)))
    lines.append(fmt_vec(flat(model.vconv.weight)))
    lines.append(fmt_vec(flat(model.vconv.bias)))
    lines.append(fmt_vec(flat(model.vfc1.weight)))
    lines.append(fmt_vec(flat(model.vfc1.bias)))
    lines.append(fmt_vec(flat(model.vfc2.weight)))
    lines.append(f"{float(model.vfc2.bias.item()):.9g}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def read_vec(tokens, idx):
    size = int(tokens[idx]); idx += 1
    vals = [float(x) for x in tokens[idx:idx + size]]
    idx += size
    return vals, idx


def load_hexcnn(path: Path) -> HexResNet:
    text = path.read_text(encoding="utf-8").split()
    if text[0] != "HEXCNN_V1":
        raise RuntimeError(f"not a HEXCNN_V1 model: {path}")
    n, C, B, in_planes, val_hidden = (int(x) for x in text[1:6])
    model = HexResNet(n, C, B, in_planes, val_hidden)
    idx = 6

    def load_into(param, idx):
        vals, idx = read_vec(text, idx)
        t = torch.tensor(vals, dtype=torch.float32).view_as(param)
        with torch.no_grad():
            param.copy_(t)
        return idx

    idx = load_into(model.conv_in.weight, idx)
    idx = load_into(model.conv_in.bias, idx)
    for blk in model.blocks:
        idx = load_into(blk["c1"].weight, idx)
        idx = load_into(blk["c1"].bias, idx)
        idx = load_into(blk["c2"].weight, idx)
        idx = load_into(blk["c2"].bias, idx)
    idx = load_into(model.pconv.weight, idx)
    idx = load_into(model.pconv.bias, idx)
    idx = load_into(model.pfc.weight, idx)
    idx = load_into(model.pfc.bias, idx)
    idx = load_into(model.vconv.weight, idx)
    idx = load_into(model.vconv.bias, idx)
    idx = load_into(model.vfc1.weight, idx)
    idx = load_into(model.vfc1.bias, idx)
    idx = load_into(model.vfc2.weight, idx)
    # vfc2 bias is the final lone scalar (not size-prefixed)
    with torch.no_grad():
        model.vfc2.bias.copy_(torch.tensor([float(text[idx])], dtype=torch.float32))
    return model


def parse_args(argv):
    p = argparse.ArgumentParser(description="Train HexAI conv ResNet (HEXCNN_V1) with PyTorch/CUDA")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--data", required=True)
    p.add_argument("--model-in", default="")
    p.add_argument("--model-out", required=True)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--recent", action="store_true")
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--val-hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=20240601)
    p.add_argument("--value-weight", type=float, default=1.0)
    p.add_argument("--lr-decay", type=float, default=0.98)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--symmetry", choices=["none", "rot180", "full"], default="full")
    p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Training-quality controls.
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fraction of base positions held out for validation (split before augmentation)")
    p.add_argument("--schedule", choices=["cosine", "step"], default="cosine")
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--min-lr-frac", type=float, default=0.05, help="cosine floor as a fraction of --lr")
    p.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 disables)")
    p.add_argument("--save-best", type=int, default=1, help="1 = keep the best-by-val-loss checkpoint")
    return p.parse_args(argv)


def lr_for_epoch(args, epoch: int) -> float:
    """1-indexed epoch -> learning rate."""
    if epoch <= args.warmup_epochs and args.warmup_epochs > 0:
        return args.lr * epoch / max(1, args.warmup_epochs)
    if args.schedule == "step":
        return args.lr * (args.lr_decay ** (epoch - 1))
    # cosine over the post-warmup epochs
    t = epoch - args.warmup_epochs
    total = max(1, args.epochs - args.warmup_epochs)
    floor = args.lr * args.min_lr_frac
    cos = 0.5 * (1.0 + math.cos(math.pi * min(t, total) / total))
    return floor + (args.lr - floor) * cos


def evaluate_val(model, loader, value_weight, device):
    """Validation loss + policy top-1 accuracy + value MAE over a DataLoader."""
    model.eval()
    n = 0
    loss_sum = pol_loss = val_loss = 0.0
    correct = 0
    val_abs = 0.0
    with torch.no_grad():
        for xb, pb, vb in loader:
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            logits, v = model(xb)
            pl = -(pb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            vl = F.mse_loss(v, vb)
            bs = xb.shape[0]
            loss_sum += float(pl + value_weight * vl) * bs
            pol_loss += float(pl) * bs
            val_loss += float(vl) * bs
            correct += int((logits.argmax(dim=1) == pb.argmax(dim=1)).sum())
            val_abs += float((v - vb).abs().sum())
            n += bs
    n = max(1, n)
    return {
        "loss": loss_sum / n,
        "policy": pol_loss / n,
        "value": val_loss / n,
        "acc": correct / n,
        "vmae": val_abs / n,
    }


def main(argv) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    data_path = Path(args.data)
    model_in = Path(args.model_in) if args.model_in else None
    model_out = Path(args.model_out)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    X, P, V = load_examples(data_path, args.n, args.limit, args.recent)
    N = X.shape[0]
    print(f"loaded examples={N}", flush=True)

    # Split indices for train/val. Augmentation is applied on the fly per item,
    # so val samples (identity only) never share an augmented copy with train.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_val = int(N * args.val_frac) if args.val_frac > 0 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"train={len(train_idx)} (on-the-fly {args.symmetry}) val={len(val_idx)}", flush=True)

    pin = device.type == "cuda"
    train_ds = HexDataset(X, P, V, train_idx, args.n, args.symmetry, train=True)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=pin, drop_last=False)
    val_loader = None
    if n_val > 0:
        val_ds = HexDataset(X, P, V, val_idx, args.n, "none", train=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=2, pin_memory=pin)

    if model_in and model_in.exists():
        model = load_hexcnn(model_in)
        print(f"loaded model: {model_in} (C={model.C} B={model.B})", flush=True)
    else:
        model = HexResNet(args.n, args.channels, args.blocks, 3, args.val_hidden)
        print(f"initialized model: n={args.n} C={args.channels} B={args.blocks}", flush=True)
    model.to(device)

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    saved_best = False
    for epoch in range(1, args.epochs + 1):
        lr = lr_for_epoch(args, epoch)
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = ploss_sum = vloss_sum = 0.0
        count = 0
        for xb, pb, vb in loader:
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            logits, v = model(xb)
            policy_loss = -(pb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(v, vb)
            loss = policy_loss + args.value_weight * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            bs = xb.shape[0]
            loss_sum += float(loss.detach()) * bs
            ploss_sum += float(policy_loss.detach()) * bs
            vloss_sum += float(value_loss.detach()) * bs
            count += bs

        msg = (f"epoch {epoch}/{args.epochs} lr={lr:.6g} "
               f"train_loss={loss_sum / max(1, count):.5f} "
               f"(policy={ploss_sum / max(1, count):.5f} value={vloss_sum / max(1, count):.5f})")
        if val_loader is not None:
            m = evaluate_val(model, val_loader, args.value_weight, device)
            msg += (f" | val_loss={m['loss']:.5f} policy={m['policy']:.5f} "
                    f"value={m['value']:.5f} top1={m['acc']*100:.1f}% vmae={m['vmae']:.3f}")
            if args.save_best and m["loss"] < best_val:
                best_val = m["loss"]
                save_hexcnn(model, model_out)
                saved_best = True
                msg += " *best"
        print(msg, flush=True)

    # If we never saved a best checkpoint (no val or save-best off), save final.
    if not saved_best:
        save_hexcnn(model, model_out)
    print(f"saved model: {model_out}" + (f" (best val_loss={best_val:.5f})" if saved_best else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
