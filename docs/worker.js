// Web Worker that hosts the Damaten Hex engine compiled to WebAssembly.
// Running it off the main thread keeps the page responsive while MCTS thinks.
//
// hexai.js is the Emscripten glue (MODULARIZE=1, EXPORT_NAME="HexAI"); it loads
// hexai.wasm and the preloaded hexai.data (which contains /hex_model.nn).
importScripts("hexai.js");

let bestMove = null, policy = null, ready = false;

HexAI().then((m) => {
  const init = m.cwrap("hex_init", "number", ["number", "string"]);
  bestMove   = m.cwrap("hex_best_move", "number", ["string", "number", "number", "number"]);
  policy     = m.cwrap("hex_policy", "string", ["string", "number"]);
  ready = init(9, "/hex_model.nn") === 1;
  postMessage({ type: "ready", ok: ready });
}).catch((e) => {
  postMessage({ type: "ready", ok: false, error: String(e) });
});

onmessage = (e) => {
  const d = e.data;
  if (!ready) { postMessage({ type: d.type, id: d.id, move: -9, data: "error;not-ready" }); return; }
  if (d.type === "move") {
    const mv = bestMove(d.board, d.player, d.last, d.iters);
    postMessage({ type: "move", id: d.id, move: mv });
  } else if (d.type === "policy") {
    const s = policy(d.board, d.player);
    postMessage({ type: "policy", id: d.id, data: s });
  }
};
