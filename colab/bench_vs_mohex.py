#!/usr/bin/env python3
"""Benchmark the Damaten HexAI model against MoHex (benzene) on Colab/Linux.

HexAI (our engine, per-move CLI) plays MoHex (HTP) under EQUAL wall-clock time
per move, from balanced random openings replayed with colours swapped so Hex's
first-player advantage cancels. It sweeps MoHex's time budget, finds the budget
where the match is ~50% (the equal-strength point), then plays a larger precise
match there. Every game is checkpointed to the CSV so a Colab disconnect resumes.

Two different engines are bridged, so on start it AUTO-DETECTS the board
convention (which colour connects which edges, and whether coordinates are
transposed) by giving MoHex one-move-to-win positions and checking it takes the
win. If it cannot reconcile the conventions it aborts loudly.

Build both engines first (see Benchmark_vs_MoHex notes), then e.g.:
  # shakedown: verify the bridge + play one game, no sweep
  python bench_vs_mohex.py --mohex ./benzene-vanilla-cmake/build/src/mohex/mohex \
      --hexai ./hexai --model models/hex_model.nn --verify-only
  # full run
  python bench_vs_mohex.py --mohex .../mohex --hexai ./hexai \
      --model models/hex_model.nn --budgets 0.1,0.5,2.0 \
      --coarse-games 20 --final-games 100 --openings 4 --out bench_results.csv
"""
import argparse, csv, math, os, random, subprocess, sys, time

EMPTY, BLACK, WHITE = 0, 1, 2
# Hex adjacency (acute corners top-left / bottom-right). Black connects top<->
# bottom (rows), White connects left<->right (cols) -- same as HexAI.
DR = (-1, -1, 0, 0, 1, 1)
DC = (0, 1, -1, 1, -1, 0)


def has_won(board, player, n):
    seen = [False] * (n * n)
    stack = []
    if player == BLACK:
        for c in range(n):
            if board[c] == BLACK:
                seen[c] = True; stack.append(c)
    else:
        for r in range(n):
            i = r * n
            if board[i] == WHITE:
                seen[i] = True; stack.append(i)
    while stack:
        v = stack.pop(); rv, cv = divmod(v, n)
        if player == BLACK and rv == n - 1:
            return True
        if player == WHITE and cv == n - 1:
            return True
        for d in range(6):
            nr, nc = rv + DR[d], cv + DC[d]
            if 0 <= nr < n and 0 <= nc < n:
                ni = nr * n + nc
                if board[ni] == player and not seen[ni]:
                    seen[ni] = True; stack.append(ni)
    return False


def legal_moves(board):
    return [i for i, x in enumerate(board) if x == EMPTY]


# ---- coordinate mapping (configurable to absorb MoHex's convention) ----
class Mapping:
    def __init__(self, transpose=False, swap_colors=False):
        self.transpose = transpose
        self.swap_colors = swap_colors

    def to_htp(self, idx, n):
        r, c = divmod(idx, n)
        if self.transpose:
            r, c = c, r
        return "%c%d" % (chr(ord('a') + c), r + 1)

    def from_htp(self, s, n):
        s = s.strip().lower()
        c = ord(s[0]) - ord('a')
        r = int(s[1:]) - 1
        if self.transpose:
            r, c = c, r
        return r * n + c

    def color(self, our_color):
        if self.swap_colors:
            our_color = WHITE if our_color == BLACK else BLACK
        return "black" if our_color == BLACK else "white"


# ---- MoHex over HTP ----
class MoHex:
    def __init__(self, exe):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def cmd(self, c):
        self.p.stdin.write(c + "\n"); self.p.stdin.flush()
        status = None; extra = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("MoHex process exited")
            s = line.rstrip("\n")
            if status is None:
                if s.startswith("=") or s.startswith("?"):
                    status = s
                continue                      # skip any banner before the response
            if s == "":
                break
            extra.append(s)
        ok = status.startswith("=")
        payload = status[1:].strip()
        if extra:
            payload = (payload + "\n" + "\n".join(extra)).strip()
        if not ok:
            raise RuntimeError("MoHex error for '%s': %s" % (c, payload))
        return payload

    def new_game(self, n, max_time):
        self.cmd("boardsize %d" % n)
        self.cmd("clear_board")
        self.cmd("param_mohex num_threads 1")
        self.cmd("param_mohex use_parallel_solver 0")
        self.cmd("param_mohex max_time %g" % max_time)

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass


def hexai_move(hexai, n, board, cur, last, model, iters):
    bs = "".join(str(x) for x in board)
    out = subprocess.run([hexai, "move", "--n", str(n), "--board", bs,
                          "--player", str(cur), "--last", str(last),
                          "--model", model, "--iters", str(iters)],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError("HexAI move failed: " + (out.stderr.strip() or "exit %d" % out.returncode))
    return int(out.stdout.split()[0])


# ---- convention auto-detection ----
def _winning_setup(n, player):
    """Board one move from a win for `player`, plus the winning index."""
    board = [EMPTY] * (n * n)
    if player == BLACK:                       # full column 0 except the last row
        for r in range(n - 1):
            board[r * n] = BLACK
        return board, (n - 1) * n             # play (n-1,0) to finish top->bottom
    else:                                      # full row 0 except the last col
        for c in range(n - 1):
            board[c] = WHITE
        return board, (n - 1)                  # play (0,n-1) to finish left->right


def detect_mapping(mohex, n):
    for transpose in (False, True):
        for swap in (False, True):
            m = Mapping(transpose, swap)
            ok = True
            for player in (BLACK, WHITE):
                board, win_idx = _winning_setup(n, player)
                mohex.new_game(n, 1.0)
                for idx, v in enumerate(board):
                    if v != EMPTY:
                        mohex.cmd("play %s %s" % (m.color(v), m.to_htp(idx, n)))
                got = mohex.cmd("genmove %s" % m.color(player)).lower()
                if got in ("resign", "pass") or m.from_htp(got, n) != win_idx:
                    ok = False; break
            if ok:
                return m
    return None


def random_opening(n, plies, rng):
    board = [EMPTY] * (n * n); cur = BLACK; moves = []
    for _ in range(plies):
        empty = [i for i, x in enumerate(board) if x == EMPTY]
        if not empty:
            break
        mv = rng.choice(empty)
        moves.append(mv); board[mv] = cur
        cur = WHITE if cur == BLACK else BLACK
    return moves


def play_game(mohex, hexai, model, n, mapping, opening, hexai_color, budget, iters):
    mohex.new_game(n, budget)
    board = [EMPTY] * (n * n); cur = BLACK; last = -1
    for mv in opening:                          # forced opening, told to MoHex
        mohex.cmd("play %s %s" % (mapping.color(cur), mapping.to_htp(mv, n)))
        board[mv] = cur; last = mv; cur = WHITE if cur == BLACK else BLACK
    while True:
        if not legal_moves(board):
            break
        if cur == hexai_color:
            mv = hexai_move(hexai, n, board, cur, last, model, iters)
            mohex.cmd("play %s %s" % (mapping.color(cur), mapping.to_htp(mv, n)))
        else:
            got = mohex.cmd("genmove %s" % mapping.color(cur)).lower()
            if got in ("resign",):
                return WHITE if cur == BLACK else BLACK   # MoHex resigned
            mv = mapping.from_htp(got, n)
        if mv < 0 or board[mv] != EMPTY:
            raise RuntimeError("illegal move %r by %d" % (got if cur != hexai_color else mv, cur))
        board[mv] = cur; last = mv
        if has_won(board, cur, n):
            return cur
        cur = WHITE if cur == BLACK else BLACK
    return BLACK if has_won(board, BLACK, n) else WHITE


def winrate_to_elo(p):
    p = min(0.999, max(0.001, p))
    return -400.0 * math.log10(1.0 / p - 1.0)


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                done[(row["budget"], int(row["game"]))] = row
    return done


def run_match(mohex, hexai, model, n, mapping, budget, iters, games, openings, rng, out, done):
    """Color-swapped pairs sharing one opening. Returns (hexai_wins, total)."""
    hx = tot = 0
    new_file = not os.path.exists(out)
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["budget", "game", "hexai_color", "winner_is_hexai"]); fh.flush()
    bkey = "%g" % budget
    for g in range(games):
        key = (bkey, g)
        if key in done:
            r = done[key]; tot += 1; hx += int(r["winner_is_hexai"]); continue
        if g % 2 == 0 or not hasattr(run_match, "opening"):
            run_match.opening = random_opening(n, openings, rng)
        hexai_color = BLACK if g % 2 == 0 else WHITE   # swap colours within the pair
        winner = play_game(mohex, hexai, model, n, mapping, run_match.opening, hexai_color, budget, iters)
        win_hexai = int(winner == hexai_color)
        hx += win_hexai; tot += 1
        w.writerow([bkey, g, hexai_color, win_hexai]); fh.flush()
        print("  budget=%gs game %d/%d HexAI=%s -> %s  (HexAI %d/%d)" % (
            budget, g + 1, games, "B" if hexai_color == BLACK else "W",
            "WIN" if win_hexai else "loss", hx, tot), flush=True)
    fh.close()
    return hx, tot


def calibrate_iters(hexai, model, n, budget):
    board = [EMPTY] * (n * n)
    for i in range(n):                          # a few stones so search is realistic
        board[i] = BLACK if i % 2 == 0 else WHITE
    ref = 600
    t0 = time.time(); hexai_move(hexai, n, board, BLACK, n - 1, model, ref); dt = time.time() - t0
    iters = max(50, int(ref * budget / max(dt, 1e-3)))
    return iters, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mohex", required=True)
    ap.add_argument("--hexai", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--budgets", default="0.1,0.5,2.0", help="seconds/move, comma-separated")
    ap.add_argument("--coarse-games", type=int, default=20)
    ap.add_argument("--final-games", type=int, default=100)
    ap.add_argument("--openings", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="bench_results.csv")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    for p in (args.mohex, args.hexai, args.model):
        if not os.path.exists(p):
            print("missing: " + p); return 2

    rng = random.Random(args.seed)
    mohex = MoHex(args.mohex)
    print("detecting board convention vs MoHex ...", flush=True)
    mapping = detect_mapping(mohex, args.n)
    if mapping is None:
        print("ABORT: could not reconcile board conventions with MoHex.\n"
              "MoHex did not take a forced win under any transpose/colour mapping.\n"
              "Check coordinate letters (i-skip?) or HTP param names.")
        mohex.close(); return 3
    print("convention OK: transpose=%s swap_colors=%s" % (mapping.transpose, mapping.swap_colors))

    budgets = [float(x) for x in args.budgets.split(",")]
    iters_for = {}
    for b in budgets:
        it, dt = calibrate_iters(args.hexai, args.model, args.n, b)
        iters_for[b] = it
        print("calibrate: budget=%gs -> HexAI iters=%d (ref move %.2fs)" % (b, it, dt))

    if args.verify_only:
        b = budgets[0]
        op = random_opening(args.n, args.openings, rng)
        w = play_game(mohex, args.hexai, args.model, args.n, mapping, op, BLACK, b, iters_for[b])
        print("shakedown game (HexAI=Black, budget=%gs): winner=%s" % (b, "HexAI" if w == BLACK else "MoHex"))
        mohex.close(); return 0

    done = load_done(args.out)
    print("\n== coarse sweep (%d games each) ==" % args.coarse_games)
    results = {}
    for b in budgets:
        hx, tot = run_match(mohex, args.hexai, args.model, args.n, mapping, b,
                            iters_for[b], args.coarse_games, args.openings, rng, args.out, done)
        wr = hx / tot if tot else 0.0
        results[b] = wr
        print("budget=%gs: HexAI winrate %.1f%% (%d/%d)  Elo %+d" % (
            b, 100 * wr, hx, tot, round(winrate_to_elo(wr))))

    best = min(budgets, key=lambda b: abs(results[b] - 0.5))
    print("\n== precise match at the closest-to-even budget: %gs (%d games) ==" % (best, args.final_games))
    hx, tot = run_match(mohex, args.hexai, args.model, args.n, mapping, best,
                        iters_for[best], args.final_games, args.openings, rng, args.out, done)
    wr = hx / tot if tot else 0.0
    print("\n=== RESULT ===")
    print("HexAI (model) vs MoHex @ %gs/move: %.1f%% (%d/%d), Elo %+d" % (
        best, 100 * wr, hx, tot, round(winrate_to_elo(wr))))
    print("=> current model is roughly as strong as MoHex given ~%gs/move on this machine." % best)
    mohex.close(); return 0


if __name__ == "__main__":
    sys.exit(main())
