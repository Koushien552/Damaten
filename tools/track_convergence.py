#!/usr/bin/env python3
"""Track Damaten training convergence.

Reads the gate history that the pipeline already writes
(``sprt_history.csv`` and ``eval_history.csv`` in the runtime dir) and turns it
into a single at-a-glance view of how model strength is progressing:

  * a console table of every gate (PASS / FAIL / INCONCLUSIVE) with its Elo,
  * the cumulative Elo trajectory over the promotions, and
  * a self-contained HTML report with an SVG chart (no matplotlib needed).

Convergence shows up as the per-promotion Elo gain shrinking toward zero and
the gates turning INCONCLUSIVE/FAIL. Re-run it any time; it always reflects the
current CSVs.

Usage:
  python tools/track_convergence.py [--config path\\to\\damaten.local.json]
                                    [--out path\\to\\report.html]
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

# A promotion this small or smaller is treated as "no real gain" for the
# convergence call.
CONVERGED_GAIN_ELO = 15.0
# How many of the most recent gates to look at when judging convergence.
RECENT_WINDOW = 4


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def find_history_paths(cfg, cfg_path):
    runtime = cfg.get("RuntimeDir")
    if not runtime:
        model = cfg.get("ModelPath")
        runtime = os.path.dirname(model) if model else os.path.dirname(cfg_path)
    sprt = cfg.get("SprtHistoryPath") or os.path.join(runtime, "sprt_history.csv")
    ev = cfg.get("EvalHistoryPath") or os.path.join(runtime, "eval_history.csv")
    return runtime, sprt, ev


def parse_ts(s):
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def read_sprt_gates(path):
    """One record per SPRT run (its final batch row)."""
    if not os.path.exists(path):
        return []
    runs = {}  # run_id -> last row
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rid = row.get("run_id")
            if rid:
                runs[rid] = row  # later rows overwrite -> final state kept
    gates = []
    for rid, row in runs.items():
        status = (row.get("status") or "").strip()
        verdict = {"H1": "PASS", "H0": "FAIL"}.get(status, "INCONCLUSIVE")
        gates.append({
            "ts": parse_ts(row.get("timestamp")),
            "ts_raw": (row.get("timestamp") or "").strip(),
            "source": "sprt",
            "label": (row.get("label") or "").strip(),
            "games": int(_f(row.get("games"))),
            "new_wins": int(_f(row.get("new_wins"))),
            "old_wins": int(_f(row.get("old_wins"))),
            "elo": _f(row.get("elo_est")),
            "verdict": verdict,
        })
    return gates


def read_eval_gates(path):
    """Fixed-N evaluations (older daily-pull path); a PASS = NEW stronger."""
    if not os.path.exists(path):
        return []
    gates = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            verdict_txt = (row.get("verdict") or "")
            if verdict_txt.startswith("NEW stronger"):
                verdict = "PASS"
            elif verdict_txt.startswith("REGRESSION"):
                verdict = "FAIL"
            else:
                verdict = "INCONCLUSIVE"
            gates.append({
                "ts": parse_ts(row.get("timestamp")),
                "ts_raw": (row.get("timestamp") or "").strip(),
                "source": "eval",
                "label": (row.get("label") or "").strip(),
                "games": int(_f(row.get("games"))),
                "new_wins": int(_f(row.get("new_wins"))),
                "old_wins": int(_f(row.get("old_wins"))),
                "elo": _f(row.get("elo")),
                "verdict": verdict,
            })
    return gates


def build_timeline(gates):
    gates = [g for g in gates if g["ts"] is not None]
    gates.sort(key=lambda g: g["ts"])
    cum = 0.0
    for g in gates:
        # A promotion is a PASS with a positive measured Elo. Only promotions
        # move the cumulative strength line.
        g["promoted"] = g["verdict"] == "PASS" and g["elo"] > 0
        if g["promoted"]:
            cum += g["elo"]
        g["cum"] = cum
    return gates


def judge_convergence(gates):
    promos = [g for g in gates if g["promoted"]]
    if not promos:
        return "no-promotions", "まだ昇格が記録されていません（ゲート履歴が貯まると判定できます）。"
    recent = gates[-RECENT_WINDOW:]
    recent_promos = [g for g in recent if g["promoted"]]
    last_gain = promos[-1]["elo"]
    if not recent_promos:
        return "converged", (
            "直近 %d ゲートで昇格なし → ほぼ頭打ち（収束）の可能性が高いです。" % len(recent)
        )
    if all(g["elo"] < CONVERGED_GAIN_ELO for g in recent_promos):
        return "near", (
            "直近の昇格 Elo がすべて +%g 未満 → 収束間近（伸びが鈍化）。" % CONVERGED_GAIN_ELO
        )
    return "improving", (
        "まだ +%g Elo 超の昇格が出ています（直近昇格 +%.0f）→ 改善が続いており収束には至っていません。"
        % (CONVERGED_GAIN_ELO, last_gain)
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def console_report(gates, verdict_key, verdict_msg):
    promos = [g for g in gates if g["promoted"]]
    total = promos[-1]["cum"] if promos else 0.0
    print("=" * 78)
    print("Damaten convergence — %d gates, %d promotions, cumulative +%.0f Elo"
          % (len(gates), len(promos), total))
    print("=" * 78)
    hdr = "%-3s %-16s %-18s %7s %9s %8s %9s" % (
        "#", "date", "label", "score", "verdict", "stepElo", "cumElo")
    print(hdr)
    print("-" * 78)
    for i, g in enumerate(gates, 1):
        score = "%d-%d" % (g["new_wins"], g["old_wins"])
        star = "*" if g["promoted"] else " "
        print("%-3d %-16s %-18s %7s %9s %+8.0f %+8.0f%s" % (
            i, g["ts_raw"][:16], g["label"][:18], score,
            g["verdict"], g["elo"], g["cum"], star))
    print("-" * 78)
    # ASCII cumulative sparkline over promotions
    if promos:
        spark = sparkline([g["cum"] for g in promos])
        print("cumulative Elo: " + spark)
    print()
    print("VERDICT [%s]: %s" % (verdict_key.upper(), verdict_msg))
    print()


def sparkline(values):
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi == lo:
        return blocks[0] * len(values)
    return "".join(blocks[int((v - lo) / (hi - lo) * (len(blocks) - 1))] for v in values)


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def svg_chart(gates):
    """Two stacked panels: cumulative Elo line + per-promotion gain bars."""
    promos = [g for g in gates if g["promoted"]]
    if not promos:
        return "<p>昇格データがまだありません。</p>"

    W, Hc, Hb, pad_l, pad_r, pad_t, pad_b = 860, 230, 190, 56, 20, 24, 40
    n = len(promos)
    cums = [g["cum"] for g in promos]
    gains = [g["elo"] for g in promos]
    cmax = max(cums) * 1.1 + 1
    gmax = max(max(gains), CONVERGED_GAIN_ELO) * 1.2 + 1

    def x(i):
        if n == 1:
            return pad_l + (W - pad_l - pad_r) / 2
        return pad_l + (W - pad_l - pad_r) * i / (n - 1)

    def yc(v):
        return pad_t + (Hc - pad_t - pad_b) * (1 - v / cmax)

    parts = []
    # --- panel A: cumulative Elo line ---
    parts.append('<svg width="%d" height="%d" xmlns="http://www.w3.org/2000/svg" '
                 'style="font-family:system-ui,Segoe UI,sans-serif;font-size:12px">' % (W, Hc + Hb + 20))
    parts.append('<text x="%d" y="16" font-weight="700">累積 Elo（強さの推移）</text>' % pad_l)
    # axes / gridlines
    for k in range(5):
        gv = cmax * k / 4
        gy = yc(gv)
        parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e3e3e3"/>' % (pad_l, gy, W - pad_r, gy))
        parts.append('<text x="%d" y="%.1f" text-anchor="end" fill="#888">%.0f</text>' % (pad_l - 6, gy + 4, gv))
    pts = " ".join("%.1f,%.1f" % (x(i), yc(c)) for i, c in enumerate(cums))
    parts.append('<polyline fill="none" stroke="#1a6fcc" stroke-width="2.5" points="%s"/>' % pts)
    for i, g in enumerate(promos):
        parts.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="#1a6fcc"/>' % (x(i), yc(g["cum"])))
        parts.append('<text x="%.1f" y="%.1f" text-anchor="middle" fill="#1a6fcc">+%.0f</text>'
                     % (x(i), yc(g["cum"]) - 8, g["elo"]))

    # --- panel B: per-promotion gain bars ---
    base = Hc + 10
    def yb(v):
        return base + pad_t + (Hb - pad_t - pad_b) * (1 - v / gmax)
    parts.append('<text x="%d" y="%d" font-weight="700">昇格ごとの伸び Elo（0に近づくほど収束）</text>' % (pad_l, base + 16))
    zero_y = yb(0)
    bw = max(6, (W - pad_l - pad_r) / max(n, 1) * 0.5)
    for i, g in enumerate(promos):
        bh = (zero_y - yb(g["elo"]))
        parts.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#2a9d57" rx="2"/>'
                     % (x(i) - bw / 2, yb(g["elo"]), bw, max(0.5, bh)))
        parts.append('<text x="%.1f" y="%.1f" text-anchor="middle" fill="#666">%s</text>'
                     % (x(i), zero_y + 14, _esc(g["ts_raw"][5:10])))
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#cc2a1a" stroke-dasharray="5,4"/>'
                 % (pad_l, yb(CONVERGED_GAIN_ELO), W - pad_r, yb(CONVERGED_GAIN_ELO)))
    parts.append('<text x="%d" y="%.1f" fill="#cc2a1a">収束しきい値 +%g</text>'
                 % (W - pad_r - 110, yb(CONVERGED_GAIN_ELO) - 4, CONVERGED_GAIN_ELO))
    parts.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#bbb"/>' % (pad_l, zero_y, W - pad_r, zero_y))
    parts.append("</svg>")
    return "".join(parts)


def html_report(gates, verdict_key, verdict_msg, out_path):
    promos = [g for g in gates if g["promoted"]]
    total = promos[-1]["cum"] if promos else 0.0
    badge = {"improving": "#2a9d57", "near": "#e08a00", "converged": "#cc2a1a",
             "no-promotions": "#888"}.get(verdict_key, "#888")
    rows = []
    for i, g in enumerate(gates, 1):
        bg = "background:#eef7f0;" if g["promoted"] else ""
        rows.append(
            "<tr style='%s'><td>%d</td><td>%s</td><td>%s</td><td>%d-%d</td>"
            "<td>%s</td><td style='text-align:right'>%+.0f</td>"
            "<td style='text-align:right'>%+.0f</td></tr>" % (
                bg, i, _esc(g["ts_raw"][:16]), _esc(g["label"]),
                g["new_wins"], g["old_wins"], g["verdict"], g["elo"], g["cum"]))
    html = """<!doctype html><meta charset="utf-8">
<title>Damaten convergence</title>
<body style="font-family:system-ui,Segoe UI,sans-serif;max-width:920px;margin:24px auto;color:#222">
<h2>Damaten 学習収束レポート</h2>
<p style="color:#666">生成: %s ／ ゲート %d 件・昇格 %d 件・累積 <b>+%.0f Elo</b></p>
<p><span style="background:%s;color:#fff;padding:3px 10px;border-radius:4px;font-weight:700">%s</span>
&nbsp;%s</p>
%s
<h3>ゲート履歴</h3>
<table style="border-collapse:collapse;width:100%%" border="0" cellpadding="6">
<tr style="background:#f0f0f0;text-align:left"><th>#</th><th>日時</th><th>label</th><th>NEW-OLD</th><th>判定</th><th>stepElo</th><th>cumElo</th></tr>
%s
</table>
<p style="color:#888;font-size:12px">緑行=昇格(PASS)。stepElo=その回の新旧Elo差、cumElo=昇格の累積。
赤破線(+%g)を下回る昇格が続けば収束目安。</p>
</body>""" % (
        datetime.now().strftime("%Y-%m-%d %H:%M"), len(gates), len(promos), total,
        badge, verdict_key.upper(), _esc(verdict_msg), svg_chart(gates),
        "\n".join(rows), CONVERGED_GAIN_ELO)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def main():
    # The Windows console is cp932 by default and cannot encode the chart/JP
    # text; switch stdout to UTF-8 so printing never crashes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    here = os.path.dirname(os.path.abspath(__file__))
    default_cfg = os.path.join(here, "..", "config", "damaten.local.json")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=default_cfg)
    ap.add_argument("--out", default=None, help="HTML report path (default: <runtime>/convergence_report.html)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runtime, sprt_path, eval_path = find_history_paths(cfg, args.config)
    gates = build_timeline(read_sprt_gates(sprt_path) + read_eval_gates(eval_path))

    if not gates:
        print("No gate history found.")
        print("  sprt: %s" % sprt_path)
        print("  eval: %s" % eval_path)
        return 0

    verdict_key, verdict_msg = judge_convergence(gates)
    console_report(gates, verdict_key, verdict_msg)

    out = args.out or os.path.join(runtime, "convergence_report.html")
    html_report(gates, verdict_key, verdict_msg, out)
    print("HTML report: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
