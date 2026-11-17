"""
Paso B: Mini-GPT (decoder-only, estilo nanoGPT) a nivel de carácter.
Piezas: embeddings token+posición -> N x [LayerNorm, atención causal
multi-cabeza, residual, LayerNorm, MLP, residual] -> LayerNorm -> lm_head.
Entrenamiento: predecir el siguiente carácter (cross-entropy).
Inferencia: muestreo autorregresivo con temperatura y top-k.
"""
import argparse, math, time, torch, torch.nn as nn, torch.nn.functional as F

TOKEN_FIN = "\n"


class AtencionCausal(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x, return_att=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))       # (B, h, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # no mirar al futuro
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.drop(self.proj(y))
        return (y, att) if return_att else y


class Bloque(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = AtencionCausal(n_embd, n_head, block_size, dropout)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x, return_att=False):
        if return_att:
            a, att = self.attn(self.ln1(x), return_att=True)
            x = x + a
            return x + self.mlp(self.ln2(x)), att
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Bloque(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def forward_trace(self, idx):
        """Como forward, pero devuelve lo que ocurre por dentro (para el panel):
        logits del último carácter, vectores por capa y atención (H, T) de cada bloque."""
        B, T = idx.shape
        te, pe = self.tok(idx), self.pos(torch.arange(T, device=idx.device))
        x = te + pe
        capas, atts = [("embedding carácter", te[0, -1]), ("embedding posición", pe[-1]), ("suma", x[0, -1])], []
        for i, b in enumerate(self.blocks):
            x, att = b(x, return_att=True)
            capas.append((f"bloque {i+1}", x[0, -1])); atts.append(att[0, :, -1, :])
        h = self.ln_f(x); capas.append(("LayerNorm final", h[0, -1]))
        return self.head(h)[0, -1], capas, atts

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, top_k=None, stop_id=None):
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)   # <-- TEMPERATURA
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and nxt.item() == stop_id:
                break
        return idx


def dispositivo():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def memoria_rss():
    try:
        import psutil; return psutil.Process().memory_info().rss
    except Exception:
        return None


def entrenar(a, callback=None, stop=None):
    """callback(dict) recibe eventos (start/iter/eval/done) para el panel; stop() -> bool corta el entrenamiento."""
    log = callback or (lambda d: None)
    torch.manual_seed(1337)
    dev = dispositivo()
    texto = open("data/corpus.txt", encoding="utf-8").read()
    chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in texto], dtype=torch.long, device=dev)  # los datos viven en la GPU: sin copias CPU->GPU
    n = int(0.9 * len(data)); train, val = data[:n], data[n:]
    offs = torch.arange(a.block_size, device=dev)
    amp = dev == "cuda" and torch.cuda.is_bf16_supported()   # precisión mixta bf16 si la GPU lo soporta
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    ac = lambda: torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16, enabled=amp)

    def batch(split):
        d = train if split == "train" else val
        ix = torch.randint(len(d) - a.block_size - 1, (a.batch_size,), device=dev)
        x = d[ix[:, None] + offs]
        y = d[ix[:, None] + offs + 1]  # desplazado 1 posición
        return x, y

    mem0 = memoria_rss()
    if dev == "cuda": torch.cuda.reset_peak_memory_stats()
    model = MiniGPT(len(chars), a.block_size, a.n_layer, a.n_head, a.n_embd, a.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{n_par/1e6:.2f}M parámetros | vocab {len(chars)} | {dev}")
    log({"type": "start", "params": n_par, "device": dev, "amp": amp, "vocab": len(chars), "corpus": len(texto)})
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    @torch.no_grad()
    def evaluar():
        model.eval()
        def perdida(s):
            with ac(): return model(*batch(s))[1].item()
        out = {s: sum(perdida(s) for _ in range(10)) / 10 for s in ("train", "val")}
        out["sample"] = generar_excusa(model, chars, ["VEHICULO"], "Se me ha reventado el coche en la M-30", 0.8, 20, 120)
        model.train(); return out

    t0, it = time.time(), 0
    log_every = max(1, a.max_iters // 500)   # .item() sincroniza CPU<->GPU: solo cuando el panel lo necesita
    for it in range(a.max_iters + 1):
        if stop and stop():
            log({"type": "stopped", "it": it}); break
        # cosine decay del learning rate con warmup corto
        lr = a.lr * min(1, (it + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / a.max_iters)))
        for g in opt.param_groups: g["lr"] = lr
        if it % a.eval_interval == 0 or it == a.max_iters:
            l = evaluar()
            print(f"iter {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | lr {lr:.2e}")
            log({"type": "eval", "it": it, "train": l["train"], "val": l["val"], "sample": l["sample"]})
        with ac(): _, loss = model(*batch("train"))
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if callback and it % log_every == 0:
            log({"type": "iter", "it": it, "loss": loss.item(), "lr": lr, "gnorm": float(gn)})

    if dev == "cuda": mem = torch.cuda.max_memory_allocated()
    else: mem = None if mem0 is None else max(0, memoria_rss() - mem0)
    torch.save({"state": model.state_dict(), "chars": chars,
                "cfg": dict(vocab_size=len(chars), block_size=a.block_size, n_layer=a.n_layer,
                            n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout)}, "generador.pt")
    print("Guardado generador.pt")
    log({"type": "done", "params": n_par, "seconds": time.time() - t0, "iters": it, "mem": mem})


def cargar_generador(ruta="generador.pt"):
    ck = torch.load(ruta, weights_only=False, map_location="cpu")
    m = MiniGPT(**ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    return m, ck["chars"]


def generar_excusa(modelo, chars, etiquetas, frase, temperature=0.8, top_k=20, max_new=300):
    stoi = {c: i for i, c in enumerate(chars)}
    cats = "".join(f"[CATEGORIA: {t}]" for t in sorted(etiquetas))
    prompt = f"{cats} TÚ: {frase} -> EXCUSA:"
    ids = torch.tensor([[stoi.get(c, stoi[" "]) for c in prompt]], device=next(modelo.parameters()).device)
    out = modelo.generate(ids, max_new, temperature, top_k, stop_id=stoi[TOKEN_FIN])[0].tolist()
    return "".join(chars[i] for i in out[ids.size(1):]).strip()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--eval_interval", type=int, default=250)
    entrenar(p.parse_args())
