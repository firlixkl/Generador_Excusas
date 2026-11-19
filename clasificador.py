"""
Paso A: Clasificador multi-label (frase -> lista de etiquetas).
Arquitectura: Embedding -> mean pooling -> MLP -> 1 logit por etiqueta.
Clave multi-label: sigmoide POR etiqueta + BCEWithLogitsLoss (no softmax),
porque las etiquetas no son excluyentes. Predicción: prob > umbral.
"""
import re, csv, random, argparse
import torch, torch.nn as nn

PAD, UNK = 0, 1


def tokenizar(texto):
    return re.findall(r"\w+", texto.lower())


class Clasificador(nn.Module):
    def __init__(self, vocab_size, n_etiquetas, emb=64, hidden=128, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.mlp = nn.Sequential(
            nn.Linear(emb, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_etiquetas),
        )

    def forward(self, ids):                      # ids: (B, T)
        mask = (ids != PAD).unsqueeze(-1).float()  # (B, T, 1)
        x = self.emb(ids) * mask
        x = x.sum(1) / mask.sum(1).clamp(min=1)    # media ignorando padding
        return self.mlp(x)                         # logits (B, n_etiquetas)


def codificar(frases, vocab, word_dropout=0.0):
    seqs = []
    for f in frases:
        ids = [vocab.get(t, UNK) for t in tokenizar(f)] or [UNK]
        if word_dropout:  # aumento de datos: tapar palabras para robustez
            ids = [UNK if random.random() < word_dropout else i for i in ids]
        seqs.append(ids)
    T = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (T - len(s)) for s in seqs])


def cargar_csv(ruta, etiquetas=None):
    frases, tags = [], []
    with open(ruta, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            frases.append(row["frase"]); tags.append(row["etiquetas"].split("|"))
    etiquetas = etiquetas or sorted({t for ts in tags for t in ts})
    Y = torch.zeros(len(frases), len(etiquetas))
    for i, ts in enumerate(tags):
        for t in ts:
            Y[i, etiquetas.index(t)] = 1
    return frases, Y, etiquetas


def cargar_modelo(ruta="clasificador.pt"):
    ck = torch.load(ruta, weights_only=True)
    m = Clasificador(len(ck["vocab"]), len(ck["etiquetas"]))
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck["vocab"], ck["etiquetas"]


@torch.no_grad()
def predecir(texto, modelo, vocab, etiquetas, umbral=0.5):
    probs = torch.sigmoid(modelo(codificar([texto], vocab)))[0]
    activas = [e for e, p in zip(etiquetas, probs) if p > umbral]
    if not activas:  # fallback: la más probable, para no dejar al generador sin tema
        activas = [etiquetas[int(probs.argmax())]]
    return activas, {e: round(float(p), 3) for e, p in zip(etiquetas, probs)}


def entrenar(a, callback=None, stop=None):
    log = callback or (lambda d: None)
    torch.manual_seed(0); random.seed(0)
    frases, Y, etiquetas = cargar_csv("data/clasificador.csv")
    idx = list(range(len(frases))); random.shuffle(idx)
    corte = int(0.9 * len(idx)); tr, va = idx[:corte], idx[corte:]
    contador = {}
    for i in tr:
        for t in tokenizar(frases[i]):
            contador[t] = contador.get(t, 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for t in contador:
        vocab[t] = len(vocab)

    model = Clasificador(len(vocab), len(etiquetas))
    n_par = sum(p.numel() for p in model.parameters())
    pr = Y[tr].mean(0).clamp(1e-6, 1 - 1e-6)  # loss de un modelo que solo conoce la frecuencia de cada etiqueta
    baseline = float(-(pr * pr.log() + (1 - pr) * (1 - pr).log()).mean())
    log({"type": "start", "params": n_par, "baseline": baseline, "vocab": len(vocab)})
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    step = 0
    for ep in range(a.epochs):
        if stop and stop():
            log({"type": "stopped", "it": step}); break
        model.train(); random.shuffle(tr); ls = []
        for b in range(0, len(tr), a.batch_size):
            bi = tr[b:b + a.batch_size]
            x = codificar([frases[i] for i in bi], vocab, word_dropout=0.1)
            loss = loss_fn(model(x), Y[bi])
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1; ls.append(loss.item())
            log({"type": "iter", "it": step, "loss": loss.item()})
        model.eval()
        with torch.no_grad():
            lg = model(codificar([frases[i] for i in va], vocab))
            vl = loss_fn(lg, Y[va]).item(); pred = (torch.sigmoid(lg) > 0.5).float()
        yv = Y[va]
        tp = (pred * yv).sum(); prec = tp / pred.sum().clamp(min=1); rec = tp / yv.sum()
        f1 = float(2 * prec * rec / (prec + rec + 1e-9)); exacto = float((pred == yv).all(1).float().mean())
        print(f"epoch {ep:3d} | loss {ls[-1]:.4f} | F1 val {f1:.3f} | exact-match {exacto:.3f}")
        log({"type": "eval", "it": step, "train": sum(ls) / len(ls), "val": vl, "f1": f1, "exact": exacto})
    torch.save({"state": model.state_dict(), "vocab": vocab, "etiquetas": etiquetas}, "clasificador.pt")
    print("Guardado clasificador.pt")
    log({"type": "done", "params": n_par, "iters": step})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    entrenar(p.parse_args())
