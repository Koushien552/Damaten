#!/usr/bin/env python3
"""GPU trainer for HexAI's C++ TinyNet model format.

The self-play engine is still the portable C++ program, but this trainer uses
PyTorch/CUDA for the policy/value network update and writes the exact
HEXNN_V1 text format that the C++ engine can load.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


BLACK = 1
WHITE = 2


def other_player(player: int) -> int:
    return WHITE if player == BLACK else BLACK


def parse_board(text: str, nn_: int) -> list[int] | None:
    if len(text) != nn_:
        return None
    board: list[int] = []
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


def parse_policy(text: str, nn_: int) -> list[float]:
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


def make_features(board: list[int], player: int, n: int) -> list[float]:
    nn_ = n * n
    opp = other_player(player)
    x = [0.0] * (2 * nn_ + 1)
    for i, cell in enumerate(board):
        if cell == player:
            x[i] = 1.0
        elif cell == opp:
            x[nn_ + i] = 1.0
    x[2 * nn_] = 1.0 if player == BLACK else -1.0
    return x


def load_examples(path: Path, n: int, limit: int = 0, recent: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nn_ = n * n
    rows: list[str]
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

    xs: list[list[float]] = []
    ps: list[list[float]] = []
    vs: list[float] = []
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
        xs.append(make_features(board, player, n))
        ps.append(parse_policy(cols[3], nn_))
        vs.append(value)

    if not xs:
        raise RuntimeError(f"no examples loaded from {path}")

    x = torch.tensor(xs, dtype=torch.float32)
    policy = torch.tensor(ps, dtype=torch.float32)
    value = torch.tensor(vs, dtype=torch.float32).view(-1, 1)
    return x, policy, value


def tail_data_lines(path: Path, limit: int, block_size: int = 1024 * 1024) -> list[str]:
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


class TinyNetTorch(nn.Module):
    def __init__(self, n: int, h1: int = 128, h2: int = 64) -> None:
        super().__init__()
        self.n = n
        self.nn = n * n
        self.input = 2 * self.nn + 1
        self.h1 = h1
        self.h2 = h2
        self.fc1 = nn.Linear(self.input, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.policy = nn.Linear(h2, self.nn)
        self.value = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = torch.tanh(self.fc1(x))
        a2 = torch.tanh(self.fc2(a1))
        logits = self.policy(a2)
        value = torch.tanh(self.value(a2))
        return logits, value


def read_vector(parts: list[str]) -> list[float]:
    if not parts:
        raise RuntimeError("empty vector line")
    size = int(parts[0])
    values = [float(x) for x in parts[1:]]
    if len(values) != size:
        raise RuntimeError(f"vector size mismatch: declared={size} actual={len(values)}")
    return values


def load_hexnn(path: Path) -> TinyNetTorch:
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().split()
        if len(header) != 4 or header[0] != "HEXNN_V1":
            raise RuntimeError(f"not a HEXNN_V1 model: {path}")
        n, h1, h2 = map(int, header[1:])
        w1 = read_vector(f.readline().split())
        b1 = read_vector(f.readline().split())
        w2 = read_vector(f.readline().split())
        b2 = read_vector(f.readline().split())
        wp = read_vector(f.readline().split())
        bp = read_vector(f.readline().split())
        wv = read_vector(f.readline().split())
        bv = float(f.readline().strip())

    model = TinyNetTorch(n, h1, h2)
    with torch.no_grad():
        model.fc1.weight.copy_(torch.tensor(w1, dtype=torch.float32).view(h1, model.input))
        model.fc1.bias.copy_(torch.tensor(b1, dtype=torch.float32))
        model.fc2.weight.copy_(torch.tensor(w2, dtype=torch.float32).view(h2, h1))
        model.fc2.bias.copy_(torch.tensor(b2, dtype=torch.float32))
        model.policy.weight.copy_(torch.tensor(wp, dtype=torch.float32).view(model.nn, h2))
        model.policy.bias.copy_(torch.tensor(bp, dtype=torch.float32))
        model.value.weight.copy_(torch.tensor(wv, dtype=torch.float32).view(1, h2))
        model.value.bias.copy_(torch.tensor([bv], dtype=torch.float32))
    return model


def init_model(n: int, h1: int, h2: int, seed: int) -> TinyNetTorch:
    model = TinyNetTorch(n, h1, h2)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        s1 = math.sqrt(2.0 / max(1, model.input))
        s2 = math.sqrt(2.0 / max(1, h1))
        s3 = math.sqrt(2.0 / max(1, h2))
        model.fc1.weight.copy_(torch.randn(model.fc1.weight.shape, generator=gen) * s1)
        model.fc1.bias.zero_()
        model.fc2.weight.copy_(torch.randn(model.fc2.weight.shape, generator=gen) * s2)
        model.fc2.bias.zero_()
        model.policy.weight.copy_(torch.randn(model.policy.weight.shape, generator=gen) * s3)
        model.policy.bias.zero_()
        model.value.weight.copy_(torch.randn(model.value.weight.shape, generator=gen) * s3)
        model.value.bias.zero_()
    return model


def flatten_tensor(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().contiguous().view(-1).tolist()


def write_vec(f, values: list[float]) -> None:
    f.write(str(len(values)))
    for x in values:
        f.write(f" {float(x):.17g}")
    f.write("\n")


def save_hexnn(model: TinyNetTorch, path: Path) -> None:
    model = model.cpu()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"HEXNN_V1 {model.n} {model.h1} {model.h2}\n")
        write_vec(f, flatten_tensor(model.fc1.weight))
        write_vec(f, flatten_tensor(model.fc1.bias))
        write_vec(f, flatten_tensor(model.fc2.weight))
        write_vec(f, flatten_tensor(model.fc2.bias))
        write_vec(f, flatten_tensor(model.policy.weight))
        write_vec(f, flatten_tensor(model.policy.bias))
        write_vec(f, flatten_tensor(model.value.weight.view(-1)))
        f.write(f"{float(model.value.bias.item()):.17g}\n")
    os.replace(tmp, path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HexAI TinyNet with PyTorch/CUDA")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--data", required=True)
    p.add_argument("--model-in", default="")
    p.add_argument("--model-out", required=True)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--recent", action="store_true")
    p.add_argument("--h1", type=int, default=128)
    p.add_argument("--h2", type=int, default=64)
    p.add_argument("--seed", type=int, default=20240531)
    p.add_argument("--value-weight", type=float, default=0.35)
    p.add_argument("--lr-decay", type=float, default=0.97)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    data_path = Path(args.data)
    model_in = Path(args.model_in) if args.model_in else None
    model_out = Path(args.model_out)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    x, target_policy, target_value = load_examples(data_path, args.n, args.limit, args.recent)
    print(f"loaded examples={len(x)} input={x.shape[1]} policy={target_policy.shape[1]}", flush=True)

    dataset = TensorDataset(x, target_policy, target_value)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )

    if model_in and model_in.exists():
        model = load_hexnn(model_in)
        if model.n != args.n:
            raise RuntimeError(f"model board size mismatch: model={model.n} requested={args.n}")
        print(f"loaded model: {model_in}", flush=True)
    else:
        model = init_model(args.n, args.h1, args.h2, args.seed)
        print(f"initialized model: n={args.n} h1={args.h1} h2={args.h2}", flush=True)

    model.to(device)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    lr = args.lr
    for epoch in range(1, args.epochs + 1):
        for group in opt.param_groups:
            group["lr"] = lr
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, pb, vb in loader:
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            logits, value = model(xb)
            policy_loss = -(pb * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value, vb)
            loss = policy_loss + args.value_weight * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bs = xb.shape[0]
            loss_sum += float(loss.detach().cpu()) * bs
            count += bs
        print(f"epoch {epoch}/{args.epochs} loss={loss_sum / max(1, count):.8f} lr={lr:.8g}", flush=True)
        lr *= args.lr_decay

    model_out.parent.mkdir(parents=True, exist_ok=True)
    save_hexnn(model, model_out)
    print(f"saved model: {model_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
