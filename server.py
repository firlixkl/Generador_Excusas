"""
Servidor del panel (FastAPI). Arranca con:  python server.py   ->  http://127.0.0.1:8000
Expone: inferencia paso a paso (streaming), laboratorio de entrenamiento y estimaciones teóricas.
"""
import os, json, math, time, threading, collections
from types import SimpleNamespace
from typing import Dict, List, Optional
import torch, torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import clasificador as C, generador as G

app = FastAPI(title="Panel Excusas ML")
M = {"clf": None, "gpt": None}
TRAIN = {"running": False, "stop": False, "log": [], "target": None, "estimate": None, "compare": None, "error": None}
CAL = {"fps": 5e12 if torch.cuda.is_available() else 2e10, "done": False}  # FLOPs/s efectivos (se calibra)
_CACHE = {}


def cargar():
    for k, f in (("clf", C.cargar_modelo), ("gpt", G.cargar_generador)):
        try: M[k] = f()
        except Exception: M[k] = None


cargar()
ndj = lambda d: json.dumps(d, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------- páginas / estado
@app.get("/")
def index(): return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def status():
    cfg = None
    if M["gpt"]:
        m = M["gpt"][0]
        cfg = dict(block_size=m.block_size, n_layer=len(m.blocks), n_head=m.blocks[0].attn.n_head, n_embd=m.tok.embedding_dim)
    return {"clf": M["clf"] is not None, "gpt": M["gpt"] is not None, "cfg": cfg, "device": G.dispositivo(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "etiquetas": M["clf"][2] if M["clf"] else []}


@app.post("/api/reload")
def reload():
    cargar(); return status()


# ---------------------------------------------------------------- INFERENCIA
class ClfReq(BaseModel):
    text: str
    umbral: float = 0.5


@app.post("/api/classify")
def classify(r: ClfReq):
    if not M["clf"]: return {"error": "No hay clasificador entrenado. Entrénalo en el Laboratorio."}
    m, vocab, et = M["clf"]
    toks = C.tokenizar(r.text) or ["<vacío>"]
    with torch.no_grad():
        probs = torch.sigmoid(m(C.codificar([r.text], vocab)))[0]
    activas = [e for e, p in zip(et, probs) if p > r.umbral] or [et[int(probs.argmax())]]
    return {"tokens": [{"t": t, "id": vocab.get(t, C.UNK), "unk": t not in vocab} for t in toks],
            "probs": {e: float(p) for e, p in zip(et, probs)}, "activas": activas}


class GenReq(BaseModel):
    text: str
    tags: List[str]
    temperature: float = 0.8
    top_k: int = 20
    max_new: int = 200
    delay_ms: int = 60
    seed: Optional[int] = None


def _pool(v, n=32):
    return F.adaptive_avg_pool1d(v.view(1, 1, -1), n).view(-1).mul(1000).round().div(1000).tolist()


@app.post("/api/generate")
def generate(r: GenReq):
    if not M["gpt"]: raise HTTPException(409, "No hay generador entrenado. Entrénalo en el Laboratorio.")
    model, chars = M["gpt"]; stoi = {c: i for i, c in enumerate(chars)}
    prompt = "".join(f"[CATEGORIA: {t}]" for t in sorted(r.tags)) + f" TÚ: {r.text} -> EXCUSA:"
    ids = [stoi.get(c, stoi[" "]) for c in prompt]
    W = model.tok.weight.detach(); Wc = W - W.mean(0)          # mapa 2D (PCA) de los embeddings
    xy = Wc @ torch.linalg.svd(Wc, full_matrices=False)[2][:2].T

    def stream():
        yield ndj({"type": "prompt", "prompt": prompt, "ids": ids, "block": model.block_size,
                   "emb": [{"c": chars[i], "x": float(xy[i, 0]), "y": float(xy[i, 1])} for i in range(len(chars))]})
        if r.seed is not None: torch.manual_seed(r.seed)
        idx, texto = torch.tensor([ids]), ""
        for i in range(r.max_new):
            ctx = idx[:, -model.block_size:]
            logits, capas, atts = model.forward_trace(ctx)
            p0 = F.softmax(logits, -1)
            lt = logits / max(r.temperature, 1e-5)                  # temperatura
            if r.top_k > 0:
                v, _ = torch.topk(lt, min(r.top_k, lt.numel())); lt[lt < v[-1]] = float("-inf")   # top-k
            p = F.softmax(lt, -1)
            nid = torch.multinomial(p, 1).item(); c = chars[nid]
            nz = p[p > 0]; ent = float(-(nz * nz.log2()).sum())
            tp = torch.topk(p, min(10, p.numel()))
            yield ndj({"type": "step", "i": i, "c": c, "id": nid, "inp": chars[ctx[0, -1].item()], "p": float(p[nid]),
                       "ent": ent, "start": idx.size(1) - ctx.size(1),
                       "top": [{"c": chars[j], "p": float(q), "p0": float(p0[j])} for q, j in zip(tp.values.tolist(), tp.indices.tolist())],
                       "att": [a.mul(1000).round().div(1000).tolist() for a in atts],
                       "capas": [{"n": n, "v": _pool(v)} for n, v in capas]})
            idx = torch.cat([idx, torch.tensor([[nid]])], 1)
            if c == "\n": break
            texto += c; time.sleep(r.delay_ms / 1000)
        yield ndj({"type": "done", "text": texto.strip()})
    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------- LABORATORIO: teoría
def refs():
    """Referencias reales sobre el corpus: entropía condicional de contexto k (lo que lograría un modelo de n-gramas)."""
    if "refs" in _CACHE: return _CACHE["refs"]
    try: txt = open("data/corpus.txt", encoding="utf-8").read()
    except Exception: return {"error": "Falta data/corpus.txt (ejecuta python datos.py)"}
    N = len(txt)

    def H(k):
        d = collections.defaultdict(collections.Counter)
        for i in range(k, N): d[txt[i - k:i]][txt[i]] += 1
        tot = 0.0
        for c in d.values():
            s = sum(c.values())
            for n in c.values(): tot -= n * math.log(n / s)
        return tot / (N - k)
    lines = [len(l) for l in txt.split("\n") if l]
    R = {"N": N, "V": len(set(txt)), "H0": H(0), "H2": H(2), "H6": H(6),
         "avg_line": sum(lines) / len(lines), "max_line": max(lines)}
    _CACHE["refs"] = R; return R


GPT_KEYS = dict(lr=float, batch_size=int, block_size=int, n_layer=int, n_head=int, n_embd=int, dropout=float, max_iters=int, eval_interval=int)
CLF_KEYS = dict(lr=float, batch_size=int, epochs=int)
parse = lambda keys, p: {k: t(p[k]) for k, t in keys.items()}


def n_params(p, V):
    C_, L, T = p["n_embd"], p["n_layer"], p["block_size"]
    return V * C_ + T * C_ + L * (12 * C_ * C_ + 13 * C_) + 2 * C_ + C_ * V   # tok + pos + bloques + ln_f + head


def estimar_gpt(P):
    R = refs()
    if "error" in R: return {"errors": [R["error"]], "tiles": [], "warnings": []}
    p = parse(GPT_KEYS, P); V, N = R["V"], R["N"]
    C_, L, B, T, H = p["n_embd"], p["n_layer"], p["batch_size"], p["block_size"], p["n_head"]
    err, warn = [], []
    if C_ % H: err.append(f"n_embd ({C_}) debe ser múltiplo de n_head ({H}).")
    params = n_params(p, V); tokens = B * T; epochs = p["max_iters"] * tokens / N
    flops = 6 * (L * 12 * C_ * C_ + C_ * V) * tokens + 12 * L * C_ * T * T * B
    spi = flops / CAL["fps"]
    mem = params * 16 + L * (34 * B * T * C_ + 5 * B * H * T * T) * 2 + B * T * V * 12
    lnV, floor = math.log(V), R["H6"]
    cap = min(1.5, (params / 150000) ** 0.35)
    lr = p["lr"]; lr_eff = max(0.05, min(2, (lr / 1e-3) ** 0.36 if lr <= 3e-3 else 3 ** 0.36 * (3e-3 / lr)))
    rate = 1.5 * cap * lr_eff   # constante calibrada con 2 entrenamientos reales (heurística, no ley física)
    pred = lambda it: floor + (lnV - floor) * math.exp(-rate * it * tokens / N)
    xs = list(range(0, p["max_iters"] + 1, max(1, p["eval_interval"])))
    if xs[-1] != p["max_iters"]: xs.append(p["max_iters"])
    curve = [{"x": x, "y": pred(x)} for x in xs]
    if T < R["avg_line"]: warn.append(f"block_size ({T}) < longitud media de línea ({R['avg_line']:.0f}): el modelo no verá la frase entera.")
    elif T < R["max_line"]: warn.append(f"Hay líneas de hasta {R['max_line']} caracteres: las más largas no cabrán enteras en block_size={T}.")
    if lr > 5e-3: warn.append("Learning rate muy alto: riesgo de que la loss oscile o explote.")
    if lr < 1e-4: warn.append("Learning rate muy bajo: aprenderá muy despacio.")
    if B < 16: warn.append("Batch pequeño: el gradiente será ruidoso (curva dentada).")
    if params / N > 0.3: warn.append(f"{params/1e6:.1f}M parámetros para {N/1e6:.1f}M caracteres: riesgo de memorizar (sobreajuste).")
    if epochs < 3: warn.append(f"Solo verá el corpus {epochs:.1f} veces: probablemente quede a medio aprender.")
    if mem > 6e9: warn.append("Memoria estimada > 6 GB: puede no caber.")
    tiles = [("params", "Parámetros", params, "num"), ("mem", "Memoria (est.)", mem, "bytes"), ("tokens", "Caracteres / iteración", tokens, "num"),
             ("epochs", "Pasadas al corpus", epochs, "x"), ("spi", "Tiempo / iteración", spi, "secs"),
             ("total", "Tiempo total", spi * p["max_iters"], "secs"), ("lnV", "Loss inicial (azar) ln V", lnV, "loss"),
             ("final", "Loss final previsto", pred(p["max_iters"]), "loss")]
    return {"errors": err, "warnings": warn, "params": params, "mem": mem, "spi": spi, "flops": flops, "curve": curve,
            "tiles": [{"k": k, "l": l, "v": v, "f": f} for k, l, v, f in tiles], "calibrated": CAL["done"],
            "refs": [{"l": "ln V = azar total", "y": lnV}, {"l": "H₀ solo frecuencias de letras", "y": R["H0"]},
                     {"l": "H₂ mirando 2 letras atrás", "y": R["H2"]}, {"l": "H₆ mirando 6 letras atrás", "y": R["H6"]}]}


def estimar_clf(P):
    p = parse(CLF_KEYS, P)
    if "clf" not in _CACHE:
        try: _CACHE["clf"] = C.cargar_csv("data/clasificador.csv")
        except Exception: return {"errors": ["Falta data/clasificador.csv (ejecuta python datos.py)"], "tiles": [], "warnings": []}
    fr, Y, et = _CACHE["clf"]
    V = len({t for f in fr for t in C.tokenizar(f)}) + 2; ntr = int(0.9 * len(fr))
    params = V * 64 + (64 * 128 + 128) + (128 * len(et) + len(et))
    pr = Y.mean(0).clamp(1e-6, 1 - 1e-6); base = float(-(pr * pr.log() + (1 - pr) * (1 - pr).log()).mean())
    steps = p["epochs"] * math.ceil(ntr / p["batch_size"]); warn = []
    if p["lr"] > 1e-2: warn.append("Learning rate muy alto para un MLP pequeño.")
    if p["epochs"] < 5: warn.append("Pocas épocas: puede quedarse corto.")
    tiles = [("params", "Parámetros", params, "num"), ("steps", "Pasos totales", steps, "num"), ("base", "Loss 'solo frecuencias' (baseline)", base, "loss")]
    return {"errors": [], "warnings": warn, "params": params, "base": base, "curve": [],
            "tiles": [{"k": k, "l": l, "v": v, "f": f} for k, l, v, f in tiles], "refs": [{"l": "baseline: ignorar el texto", "y": base}]}


class EstReq(BaseModel):
    target: str = "gpt"
    params: Dict[str, float]


@app.post("/api/estimate")
def estimate(r: EstReq):
    try: return estimar_gpt(r.params) if r.target == "gpt" else estimar_clf(r.params)
    except Exception as e: return {"errors": [repr(e)], "tiles": [], "warnings": []}


@app.post("/api/benchmark")
def benchmark(r: EstReq):
    if TRAIN["running"]: return {"error": "Hay un entrenamiento en marcha."}
    p = parse(GPT_KEYS, r.params); R = refs(); dev = G.dispositivo()
    m = G.MiniGPT(R["V"], p["block_size"], p["n_layer"], p["n_head"], p["n_embd"], p["dropout"]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=p["lr"]); x = torch.randint(R["V"], (p["batch_size"], p["block_size"]), device=dev)
    for i in range(4):
        if i == 1: t = time.time()
        loss = m(x, x)[1]; opt.zero_grad(); loss.backward(); opt.step()
    spi = (time.time() - t) / 3
    CAL["fps"] = estimar_gpt(r.params)["flops"] / spi; CAL["done"] = True
    return {"spi": spi}


# ---------------------------------------------------------------- LABORATORIO: entrenamiento
class TrainReq(BaseModel):
    target: str = "gpt"
    params: Dict[str, float]


def _fmt(v, f):
    if v is None: return "no medida"
    return {"num": f"{v:,.0f}", "secs": f"{v:.3f} s" if v < 90 else f"{v/60:.1f} min", "bytes": f"{v/1e6:,.0f} MB", "loss": f"{v:.3f}"}[f]


def comparar(target, est, log):
    done = next(e for e in reversed(log) if e["type"] == "done"); evs = [e for e in log if e["type"] == "eval"]
    rows = [("Parámetros", _fmt(est["params"], "num"), _fmt(done["params"], "num"),
             "Idéntico: la fórmula es exacta." if est["params"] == done["params"] else "Diferencia pequeña: el vocabulario real sale solo del 90% de entrenamiento.")]
    if target == "gpt":
        spi = done["seconds"] / max(1, done["iters"])
        rows.append(("Tiempo / iteración", _fmt(est["spi"], "secs"), _fmt(spi, "secs"),
                     "Incluye evaluaciones y muestras. Pulsa «Medir mi máquina» para calibrar la teoría." if not est["calibrated"] else "Calibrado con tu máquina."))
        rows.append(("Memoria", _fmt(est["mem"], "bytes"), _fmt(done["mem"], "bytes"), "Orden de magnitud; la real depende del framework y del sistema."))
        if evs:
            last = evs[-1]; c = min(est["curve"], key=lambda q: abs(q["x"] - last["it"])); d = last["val"] - c["y"]
            rows.append(("Loss val final", _fmt(c["y"], "loss"), _fmt(last["val"], "loss"),
                         "Aprendió más lento de lo previsto: sube iteraciones, lr o capacidad." if d > 0.3 else
                         "Mejor de lo previsto: el corpus es muy repetitivo (plantillas) y el modelo lo memoriza bien." if d < -0.3 else "Previsión acertada."))
            pm = {q["x"]: q["y"] for q in est["curve"]}; er = [abs(e["val"] - pm[e["it"]]) for e in evs if e["it"] in pm]
            if er: rows.append(("Error medio de la curva", "—", f"{sum(er)/len(er):.3f}", "Distancia media entre la curva prevista y la real."))
            gap = last["val"] - last["train"]
            rows.append(("Brecha val − train", "≈ 0", f"{gap:+.3f}", "Sobreajuste: el modelo memoriza." if gap > 0.15 else "Sin sobreajuste apreciable."))
    elif evs:
        last = evs[-1]
        rows.append(("Loss val final", f"< {est['base']:.3f} (baseline)", _fmt(last["val"], "loss"), "Mejor que ignorar el texto." if last["val"] < est["base"] else "No supera al baseline."))
        rows.append(("F1 validación", "—", f"{last['f1']:.3f}", "Ojo: dataset sintético con plantillas repetidas; F1 optimista."))
    return [{"l": a, "t": b, "r": c, "n": d} for a, b, c, d in rows]


def _run(target, a):
    try:
        cb = lambda d: TRAIN["log"].append(d); st = lambda: TRAIN["stop"]
        (G.entrenar if target == "gpt" else C.entrenar)(a, cb, st)
        done = next(e for e in reversed(TRAIN["log"]) if e["type"] == "done")
        if target == "gpt" and done["iters"] > 0:
            CAL["fps"] = TRAIN["estimate"]["flops"] / (done["seconds"] / done["iters"]); CAL["done"] = True
        TRAIN["compare"] = comparar(target, TRAIN["estimate"], TRAIN["log"])
        cargar()
    except Exception as e:
        TRAIN["error"] = repr(e)
    finally:
        TRAIN["running"] = False


@app.post("/api/train/start")
def train_start(r: TrainReq):
    if TRAIN["running"]: return {"error": "Ya hay un entrenamiento en marcha."}
    est = estimate(EstReq(target=r.target, params=r.params))
    if est.get("errors"): return {"error": " · ".join(est["errors"])}
    TRAIN.update(running=True, stop=False, log=[], target=r.target, estimate=est, compare=None, error=None)
    a = SimpleNamespace(**parse(GPT_KEYS if r.target == "gpt" else CLF_KEYS, r.params))
    threading.Thread(target=_run, args=(r.target, a), daemon=True).start()
    return {"ok": True}


@app.post("/api/train/stop")
def train_stop():
    TRAIN["stop"] = True; return {"ok": True}


@app.get("/api/train/status")
def train_status(since: int = 0):
    lg = TRAIN["log"]
    return {"running": TRAIN["running"], "log": lg[since:], "n": len(lg), "error": TRAIN["error"], "compare": TRAIN["compare"]}


if __name__ == "__main__":
    import uvicorn
    print("Abre http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
