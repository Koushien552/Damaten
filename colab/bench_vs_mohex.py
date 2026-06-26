#!/usr/bin/env python3
"""Benchmark the Damaten HexAI engine against MoHex, both driven over HTP.

HexAI (built with its `htp` mode) and MoHex (benzene) are launched as persistent
HTP processes and refereed move-by-move from balanced random openings replayed
with colours swapped (so Hex's first-player advantage cancels). HexAI is held at
a fixed strength (--hexai-iters); MoHex's time budget is swept to find the point
where the match is ~50%, then a larger precise match is played there. That point
("our engine is about as strong as MoHex given T seconds/move") is the absolute
yardstick. Games are checkpointed to the CSV so a Colab disconnect resumes.

The two engines are auto-checked for a matching board convention (each must take
a one-move-to-win position); MoHex coordinates are transposed automatically if
its convention differs. Aborts loudly if it cannot reconcile them.

  python bench_vs_mohex.py \
    --mohex ./benzene-vanilla-cmake/build/src/mohex/mohex \
    --hexai ./hexai --model models/hex_model.nn \
    --hexai-iters 1500 --budgets 0.1,0.5,2.0 \
    --coarse-games 20 --final-games 100 --openings 4 --out bench_results.csv
"""
import argparse, csv, math, os, random, subprocess, sys, time

EMPTY, BLACK, WHITE = 0, 1, 2
DR = (-1, -1, 0, 0, 1, 1)
DC = (0, 1, -1, 1, -1, 0)


def has_won(board, player, n):
    seen = [False] * (n * n); st = []
    if player == BLACK:
        for c in range(n):
            if board[c] == BLACK: seen[c] = True; st.append(c)
    else:
        for r in range(n):
            i = r * n
            if board[i] == WHITE: seen[i] = True; st.append(i)
    while st:
        v = st.pop(); rv, cv = divmod(v, n)
        if player == BLACK and rv == n - 1: return True
        if player == WHITE and cv == n - 1: return True
        for d in range(6):
            nr, nc = rv + DR[d], cv + DC[d]
            if 0 <= nr < n and 0 <= nc < n:
                ni = nr * n + nc
                if board[ni] == player and not seen[ni]:
                    seen[ni] = True; st.append(ni)
    return False


def legal_moves(board):
    return [i for i, x in enumerate(board) if x == EMPTY]


def cname(col):
    return "black" if col == BLACK else "white"


def other(col):
    return WHITE if col == BLACK else BLACK


def coord(idx, n, transpose=False):
    r, c = divmod(idx, n)
    if transpose: r, c = c, r
    return chr(97 + c) + str(r + 1)


def uncoord(s, n, transpose=False):
    s = s.strip().lower()
    c = ord(s[0]) - 97
    r = int(s[1:]) - 1
    if transpose: r, c = c, r
    return r * n + c


class Htp:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def _io(self, c):
        self.p.stdin.write(c + "\n"); self.p.stdin.flush()
        status = None; extra = []
        while True:
            line = self.p.stdout.readline()
            if line == "":
                raise RuntimeError("engine exited")
            s = line.rstrip("\n")
            if status is None:
                if s.startswith("=") or s.startswith("?"):
                    status = s
                continue
            if s == "":
                break
            extra.append(s)
        payload = status[1:].strip()
        if extra:
            payload = (payload + "\n" + "\n".join(extra)).strip()
        return status.startswith("="), payload

    def cmd(self, c):
        ok, p = self._io(c)
        if not ok:
            raise RuntimeError("'%s' -> %s" % (c, p))
        return p

    def opt(self, c):
        try:
            return self.cmd(c)
        except Exception:
            return None

    def quit(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass


def forced_win_ok(eng, n, transpose):
    for player in (BLACK, WHITE):
        eng.cmd("boardsize %d" % n); eng.cmd("clear_board")
        if player == BLACK:
            stones = [r * n for r in range(n - 1)]; win = (n - 1) * n
        else:
            stones = [c for c in range(n - 1)]; win = n - 1
        for idx in stones:
            eng.cmd("play %s %s" % (cname(player), coord(idx, n, transpose)))
        got = eng.cmd("genmove %s" % cname(player)).lower()
        if got in ("resign", "pass") or uncoord(got, n, transpose) != win:
            return False
    return True


def detect_transpose(eng, n, label):
    if forced_win_ok(eng, n, False):
        return False
    if forced_win_ok(eng, n, True):
        return True
    raise RuntimeError("could not reconcile %s board convention "
                       "(did not take a forced win under either mapping)" % label)


def play_game(hexai, mohex, hexai_color, opening, n, mtp):
    for e in (hexai, mohex):
        e.cmd("boardsize %d" % n); e.cmd("clear_board")
    board = [EMPTY] * (n * n); cur = BLACK
    for mv in opening:
        hexai.cmd("play %s %s" % (cname(cur), coord(mv, n)))
        mohex.cmd("play %s %s" % (cname(cur), coord(mv, n, mtp)))
        board[mv] = cur; cur = other(cur)
    while legal_moves(board):
        if cur == hexai_color:
            got = hexai.cmd("genmove %s" % cname(cur)).lower()
            tp, opp = False, mohex
        else:
            got = mohex.cmd("genmove %s" % cname(cur)).lower()
            tp, opp = mtp, hexai
        if got == "resign":
            return other(cur)
        if got == "pass":
            cur = other(cur); continue
        mv = uncoord(got, n, tp)
        opp.cmd("play %s %s" % (cname(cur), coord(mv, n, (mtp if opp is mohex else False))))
        board[mv] = cur
        if has_won(board, cur, n):
            return cur
        cur = other(cur)
    return BLACK if has_won(board, BLACK, n) else WHITE


def set_mohex_budget(mohex, budget):
    mohex.opt("param_mohex num_threads 1")
    mohex.opt("param_mohex use_parallel_solver 0")
    mohex.opt("param_mohex max_time %g" % budget)
    mohex.opt("param_mohex max_games 999999999")


def winrate_to_elo(p):
    p = min(0.999, max(0.001, p))
    return -400.0 * math.log10(1.0 / p - 1.0)


def load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                done[(row["budget"], int(row["game"]))] = int(row["winner_is_hexai"])
    return done


def run_match(hexai, mohex, n, mtp, budget, games, openings, rng, out, done):
    set_mohex_budget(mohex, budget)
    hx = tot = 0
    new = not os.path.exists(out)
    fh = open(out, "a", newline=""); w = csv.writer(fh)
    if new:
        w.writerow(["budget", "game", "hexai_color", "winner_is_hexai"]); fh.flush()
    bkey = "%g" % budget; opening = None
    for g in range(games):
        if (bkey, g) in done:
            tot += 1; hx += done[(bkey, g)]; continue
        if g % 2 == 0 or opening is None:
            opening = _rand_opening(n, openings, rng)
        hexai_color = BLACK if g % 2 == 0 else WHITE
        winner = play_game(hexai, mohex, hexai_color, opening, n, mtp)
        win = int(winner == hexai_color)
        hx += win; tot += 1
        w.writerow([bkey, g, hexai_color, win]); fh.flush()
        print("  budget=%gs game %d/%d HexAI=%s -> %s  (HexAI %d/%d)" % (
            budget, g + 1, games, "B" if hexai_color == BLACK else "W",
            "WIN" if win else "loss", hx, tot), flush=True)
    fh.close()
    return hx, tot


def _rand_opening(n, plies, rng):
    board = [EMPTY] * (n * n); cur = BLACK; mv = []
    for _ in range(plies):
        empty = [i for i, x in enumerate(board) if x == EMPTY]
        if not empty: break
        m = rng.choice(empty); mv.append(m); board[m] = cur; cur = other(cur)
    return mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mohex", required=True)
    ap.add_argument("--hexai", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--hexai-iters", type=int, default=1500)
    ap.add_argument("--budgets", default="0.1,0.5,2.0")
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
    hexai = Htp([args.hexai, "htp", "--model", args.model, "--iters", str(args.hexai_iters)])
    mohex = Htp([args.mohex])
    print("hexai:", hexai.opt("name"), "| mohex:", mohex.opt("name"), flush=True)
    detect_transpose(hexai, args.n, "HexAI")            # sanity (expect identity)
    mtp = detect_transpose(mohex, args.n, "MoHex")
    print("convention OK (MoHex transpose=%s)" % mtp, flush=True)

    if args.verify_only:
        op = _rand_opening(args.n, args.openings, rng)
        set_mohex_budget(mohex, float(args.budgets.split(",")[0]))
        w = play_game(hexai, mohex, BLACK, op, args.n, mtp)
        print("shakedown game (HexAI=Black): winner =", "HexAI" if w == BLACK else "MoHex")
        hexai.quit(); mohex.quit(); return 0

    budgets = [float(x) for x in args.budgets.split(",")]
    done = load_done(args.out)
    print("\n== coarse sweep (%d games each, HexAI fixed at %d iters) ==" % (args.coarse_games, args.hexai_iters))
    res = {}
    for b in budgets:
        hx, tot = run_match(hexai, mohex, args.n, mtp, b, args.coarse_games, args.openings, rng, args.out, done)
        res[b] = hx / tot if tot else 0.0
        print("budget=%gs: HexAI %.1f%% (%d/%d)  Elo %+d" % (b, 100 * res[b], hx, tot, round(winrate_to_elo(res[b]))))

    best = min(budgets, key=lambda b: abs(res[b] - 0.5))
    print("\n== precise match at the closest-to-even budget: %gs (%d games) ==" % (best, args.final_games))
    hx, tot = run_match(hexai, mohex, args.n, mtp, best, args.final_games, args.openings, rng, args.out, done)
    wr = hx / tot if tot else 0.0
    print("\n=== RESULT ===")
    print("HexAI(%d iters) vs MoHex @ %gs/move: %.1f%% (%d/%d), Elo %+d" % (
        args.hexai_iters, best, 100 * wr, hx, tot, round(winrate_to_elo(wr))))
    print("=> the model is about as strong as MoHex given ~%gs/move on this machine." % best)
    hexai.quit(); mohex.quit(); return 0


if __name__ == "__main__":
    sys.exit(main())
