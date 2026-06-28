# Damaten Hex — Web (WebAssembly) build

A fully static, client-side build of the Hex engine. The C++ engine
(`../src/main.cpp`) is compiled to WebAssembly and runs **entirely in the
visitor's browser** (in a Web Worker) — there is no server and no per-move
network traffic. Hosting is just static files on GitHub Pages.

## Files

| File | Role |
|------|------|
| `index.html` | the UI (board, heatmap, controls) |
| `worker.js` | Web Worker that hosts the WASM engine |
| `hexai.js` / `hexai.wasm` / `hexai.data` | **generated** by the build; `.data` bundles the model at `/hex_model.nn` |
| `build_wasm.ps1` | build script (runs `emcc`) |
| `.nojekyll` | tells GitHub Pages to serve `.wasm`/`.data` untouched |

## Build

1. Install the Emscripten SDK once:
   ```powershell
   git clone https://github.com/emscripten-core/emsdk C:\Users\ryama\emsdk
   cd C:\Users\ryama\emsdk; .\emsdk install latest; .\emsdk activate latest
   ```
2. Build (auto-activates emsdk if needed):
   ```powershell
   powershell -ExecutionPolicy Bypass -File docs\build_wasm.ps1          # scalar, portable (default)
   powershell -ExecutionPolicy Bypass -File docs\build_wasm.ps1 -Simd    # AVX2 -> WASM-SIMD, faster (experimental)
   ```
   Output: `docs/hexai.js`, `docs/hexai.wasm`, `docs/hexai.data`.

## Run locally

`file://` cannot load Workers/WASM; serve over HTTP:
```powershell
cd docs; python -m http.server 8000   # then open http://localhost:8000
```

## Deploy (GitHub Pages)

1. Commit `docs/` (including the generated `hexai.*`) and push to GitHub.
2. Repo **Settings → Pages → Build and deployment → Deploy from a branch**,
   Branch = `main`, Folder = `/docs`. Save.
3. The site appears at `https://koushien552.github.io/Damaten/` after ~1 minute.

Notes:
- The repo must be **public** for Pages on the free tier.
- Single-threaded build → no `SharedArrayBuffer`, so no COOP/COEP headers are
  needed, which is why plain GitHub Pages works with zero configuration.
- To update the model, replace `../models/hex_model.nn` and rebuild (it is
  baked into `hexai.data`).
