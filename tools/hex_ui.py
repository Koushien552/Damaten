#!/usr/bin/env python3
"""Local web UI to play Hex against the trained HexAI engine.

Serves a clickable hex board on http://localhost:PORT and bridges your clicks to
the engine's `move` command (the same CLI the Colab notebook uses). No external
dependencies -- stdlib http.server plus the existing HexAI.exe and model.

  python tools/hex_ui.py                       # paths from config/damaten.local.json
  python tools/hex_ui.py --port 8080 --iters 1500
  python tools/hex_ui.py --exe C:\\path\\HexAI.exe --model C:\\path\\hex_model.nn

Blue (you, by default) connects top<->bottom; Red (engine) connects left<->right,
matching HexAI's BLACK=1 / WHITE=2 convention.
"""
import argparse, json, math, os, subprocess, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Damaten Hex</title>
<style>
 body{font-family:system-ui,'Segoe UI',sans-serif;max-width:680px;margin:24px auto;padding:0 16px;color:#222;background:#fafaf7}
 .row{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
 #status{font-size:16px;font-weight:600;min-width:130px}
 .muted{color:#666;font-size:13px}
 button,select{font:inherit;padding:6px 12px;border:1px solid #bbb;border-radius:8px;background:#fff;cursor:pointer}
 input[type=range]{vertical-align:middle}
 svg{width:100%;height:auto;display:block;touch-action:manipulation}
 .pol{font-family:system-ui,'Segoe UI',sans-serif;font-weight:700}
 #val{font-variant-numeric:tabular-nums}
</style></head><body>
<div class="row">
  <span id="status">手番: あなた</span>
  <span id="moves" class="muted">手数 0</span>
  <span id="val" class="muted"></span>
  <span style="flex:1"></span>
  <label class="muted"><input id="polchk" type="checkbox"> 方策ヒートマップ</label>
  <label class="muted">あなた <select id="side"><option value="1">青(先手・上下)</option><option value="2">赤(後手・左右)</option></select></label>
  <label class="muted">AI強さ <input id="iters" type="range" min="200" max="4000" step="100" value="__ITERS__"><span id="itersv">__ITERS__</span></label>
  <button id="undo">一手戻す</button>
  <button id="newgame">新規対局</button>
</div>
<svg id="board" viewBox="0 0 520 330" role="img" aria-label="Hex board"></svg>
<p class="muted"><b style="color:#185FA5">● 青</b> 上下をつなぐ ／ <b style="color:#E24B4A">● 赤</b> 左右をつなぐ。空マスをクリックで着手。<br>
<b>方策ヒートマップ</b>: 手番側から見た方策ネットの確率（合法手のソフトマックス, 探索なし）。濃いほど高確率、数字は%。「評価」は手番側の推定勝率。</p>
<script>
const N=__N__, s=20, NS="http://www.w3.org/2000/svg";
const colSp=Math.sqrt(3)*s,rowSp=1.5*s,shift=colSp/2,X0=30,Y0=34;
const BLUE=1,RED=2,DR=[-1,-1,0,0,1,1],DC=[0,1,-1,1,-1,0];
const svg=document.getElementById("board"),statusEl=document.getElementById("status"),movesEl=document.getElementById("moves");
const sideSel=document.getElementById("side"),itEl=document.getElementById("iters"),itv=document.getElementById("itersv");
const polChk=document.getElementById("polchk"),valEl=document.getElementById("val");
let board,history,winner,winSet,human,engine,busy,pol=null;
function heat(p,pmax){const a=pmax>0?0.10+0.80*(p/pmax):0;return "rgba(31,119,180,"+a.toFixed(3)+")";}
const cx=(r,c)=>X0+c*colSp+r*shift+s, cy=r=>Y0+r*rowSp+s, ix=(r,c)=>r*N+c;
function pts(x,y){let p=[];for(let k=0;k<6;k++){const a=(60*k-90)*Math.PI/180;p.push((x+s*Math.cos(a)).toFixed(1)+","+(y+s*Math.sin(a)).toFixed(1));}return p.join(" ");}
function ln(x1,y1,x2,y2,col){const l=document.createElementNS(NS,"line");l.setAttribute("x1",x1);l.setAttribute("y1",y1);l.setAttribute("x2",x2);l.setAttribute("y2",y2);l.setAttribute("stroke",col);l.setAttribute("stroke-width","5");l.setAttribute("stroke-linecap","round");l.setAttribute("opacity",".85");svg.appendChild(l);}
function win(p){const seen=new Array(N*N).fill(false),st=[];for(let i=0;i<N;i++){if(p===BLUE){if(board[i]===BLUE){seen[i]=1;st.push(i);}}else{const j=i*N;if(board[j]===RED){seen[j]=1;st.push(j);}}}const R=new Set();while(st.length){const v=st.pop();R.add(v);const r=(v/N)|0,c=v%N;for(let d=0;d<6;d++){const nr=r+DR[d],nc=c+DC[d];if(nr>=0&&nr<N&&nc>=0&&nc<N){const ni=nr*N+nc;if(board[ni]===p&&!seen[ni]){seen[ni]=1;st.push(ni);}}}}for(const v of R){const r=(v/N)|0,c=v%N;if((p===BLUE&&r===N-1)||(p===RED&&c===N-1))return R;}return null;}
function render(){while(svg.firstChild)svg.removeChild(svg.firstChild);
 ln(cx(0,0),cy(0)-s,cx(0,N-1),cy(0)-s,"#185FA5");ln(cx(N-1,0),cy(N-1)+s,cx(N-1,N-1),cy(N-1)+s,"#185FA5");
 ln(cx(0,0)-s,cy(0),cx(N-1,0)-s,cy(N-1),"#E24B4A");ln(cx(0,N-1)+s,cy(0),cx(N-1,N-1)+s,cy(N-1),"#E24B4A");
 for(let r=0;r<N;r++)for(let c=0;c<N;c++){const x=cx(r,c),y=cy(r),i=ix(r,c),v=board[i];
  const showPol=pol&&!v&&!winner,p=showPol?pol.probs[i]:0;
  const po=document.createElementNS(NS,"polygon");po.setAttribute("points",pts(x,y));po.setAttribute("fill",showPol?heat(p,pol.pmax):"#fff");po.setAttribute("stroke","#9a988f");po.setAttribute("stroke-width","1");po.style.cursor=(v||winner||busy)?"default":"pointer";po.onclick=()=>click(r,c);svg.appendChild(po);
  if(showPol&&p>=0.02){const t=document.createElementNS(NS,"text");t.setAttribute("x",x);t.setAttribute("y",y+3.5);t.setAttribute("text-anchor","middle");t.setAttribute("class","pol");t.setAttribute("font-size",p===pol.pmax?"11":"9");t.setAttribute("fill",p/pol.pmax>0.55?"#fff":"#0b3d61");t.style.pointerEvents="none";t.textContent=Math.round(p*100);svg.appendChild(t);}
  if(v){const st=document.createElementNS(NS,"circle");st.setAttribute("cx",x);st.setAttribute("cy",y);st.setAttribute("r",s*0.62);st.setAttribute("fill",v===BLUE?"#185FA5":"#E24B4A");if(winSet&&winSet.has(i)){st.setAttribute("stroke","#EF9F27");st.setAttribute("stroke-width","3");}st.style.pointerEvents="none";svg.appendChild(st);}}}
function setStatus(t){statusEl.textContent=t;}
function place(i,col){board[i]=col;history.push(i);movesEl.textContent="手数 "+history.length;pol=null;}
async function fetchPolicy(){
 if(!polChk.checked){pol=null;valEl.textContent="";render();return;}
 if(winner||busy)return;
 const side=(history.length%2===0)?BLUE:RED;
 try{const res=await fetch("/policy",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:board.join(""),player:side})});
  const d=await res.json();
  if(d.error){pol=null;valEl.textContent="方策: "+d.error;render();return;}
  let pmax=0;for(const q of d.probs)if(q>pmax)pmax=q;
  pol={probs:d.probs,pmax:pmax};
  valEl.textContent="評価 "+(side===BLUE?"青":"赤")+"手番 勝率"+Math.round(((d.value+1)/2)*100)+"%";
  render();
 }catch(e){pol=null;}
}
function checkEnd(col){const w=win(col);if(w){winner=col;winSet=w;setStatus((col===human?"あなた":"AI")+"の勝ち！("+(col===BLUE?"青":"赤")+")");render();return true;}return false;}
async function engineMove(last){busy=true;setStatus("AI思考中…");render();
 try{const res=await fetch("/move",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({board:board.join(""),player:engine,last:last,iters:+itEl.value})});
  const d=await res.json();if(d.error){setStatus("エラー: "+d.error);busy=false;return;}
  place(d.move,engine);if(!checkEnd(engine)){setStatus("手番: あなた");}busy=false;render();fetchPolicy();
 }catch(e){setStatus("通信エラー: "+e);busy=false;}}
function click(r,c){if(busy||winner)return;const i=ix(r,c);if(board[i])return;
 const side=(history.length%2===0)?BLUE:RED;if(side!==human)return;
 place(i,human);if(checkEnd(human))return;engineMove(i);}
function newGame(){human=+sideSel.value;engine=human===BLUE?RED:BLUE;board=new Array(N*N).fill(0);history=[];winner=0;winSet=null;busy=false;
 setStatus("手番: あなた");movesEl.textContent="手数 0";render();if(engine===BLUE)engineMove(-1);else fetchPolicy();}
function undo(){if(busy||!history.length)return;winner=0;winSet=null;
 const n=(engine===BLUE&&history.length===1)?1:Math.min(2,history.length);
 for(let k=0;k<n;k++){board[history.pop()]=0;}movesEl.textContent="手数 "+history.length;
 const side=(history.length%2===0)?BLUE:RED;setStatus("手番: あなた");pol=null;render();
 if(side===engine&&history.length===0&&engine===BLUE)engineMove(-1);else fetchPolicy();}
itEl.oninput=()=>itv.textContent=itEl.value;
polChk.onchange=fetchPolicy;
document.getElementById("newgame").onclick=newGame;
document.getElementById("undo").onclick=undo;
newGame();
</script></body></html>"""


def load_paths(args):
    exe, model, n = args.exe, args.model, args.n
    cfg_path = args.config
    if (not exe or not model) and not cfg_path:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.join(here, "..", "config", "damaten.local.json")
        if os.path.exists(cand):
            cfg_path = cand
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8-sig") as fh:
            cfg = json.load(fh)
        exe = exe or cfg.get("ExePath")
        model = model or cfg.get("ModelPath")
        n = n or int(cfg.get("BoardSize", 9))
    return exe, model, (n or 9)


def make_handler(exe, model, n):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = PAGE.replace("__N__", str(n)).replace("__ITERS__", "1200")
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path == "/move":
                self._move(); return
            if self.path == "/policy":
                self._policy(); return
            self._send(404, "not found", "text/plain")

        def _read_req(self):
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length))

        def _move(self):
            try:
                req = self._read_req()
                board = str(req["board"]); player = int(req["player"])
                last = int(req["last"]); iters = int(req.get("iters", 1200))
                cmd = [exe, "move", "--n", str(n), "--board", board,
                       "--player", str(player), "--last", str(last), "--iters", str(iters)]
                if model and os.path.exists(model):
                    cmd += ["--model", model]
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if out.returncode != 0:
                    raise RuntimeError(out.stderr.strip() or ("exit %d" % out.returncode))
                mv = int(out.stdout.split()[0])
                self._send(200, json.dumps({"move": mv}), "application/json")
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}), "application/json")

        # Raw policy-network read for the position: the engine's `eval` command
        # prints the value head plus one logit per cell (no search). We softmax
        # the logits over the *legal* (empty) cells so the numbers are a proper
        # move distribution, and return it for the board heatmap.
        def _policy(self):
            try:
                req = self._read_req()
                board = str(req["board"]); player = int(req["player"])
                if not (model and os.path.exists(model)):
                    raise RuntimeError("no model loaded (pure MCTS has no policy net)")
                cmd = [exe, "eval", "--n", str(n), "--board", board,
                       "--player", str(player), "--model", model]
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if out.returncode != 0:
                    raise RuntimeError(out.stderr.strip() or ("exit %d" % out.returncode))
                value = None
                logits = None
                for line in out.stdout.splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    if parts[0] == "value":
                        value = float(parts[1])
                    elif parts[0] == "logits":
                        logits = [float(x) for x in parts[1:]]
                if value is None or logits is None or len(logits) != n * n:
                    raise RuntimeError("unexpected eval output")
                # Mask occupied cells, then softmax over the rest.
                legal = [i for i in range(n * n) if board[i] == "0"]
                if not legal:
                    self._send(200, json.dumps({"value": value, "probs": [0.0] * (n * n)}),
                               "application/json")
                    return
                mx = max(logits[i] for i in legal)
                exps = [math.exp(logits[i] - mx) if board[i] == "0" else 0.0 for i in range(n * n)]
                z = sum(exps) or 1.0
                probs = [e / z for e in exps]
                self._send(200, json.dumps({"value": value, "probs": probs}),
                           "application/json")
            except Exception as e:
                self._send(200, json.dumps({"error": str(e)}), "application/json")
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    exe, model, n = load_paths(args)
    if not exe or not os.path.exists(exe):
        print("HexAI.exe not found: %r (pass --exe or fix config)" % exe); return 2
    print("engine: %s" % exe)
    print("model : %s" % (model if model and os.path.exists(model) else "(none - pure MCTS)"))
    url = "http://localhost:%d" % args.port
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(exe, model, n))
    print("Damaten Hex UI on %s  (Ctrl+C to stop)" % url)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
