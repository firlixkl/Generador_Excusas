"""
Paso 0: genera los dos datasets del proyecto a partir de semilla.txt.
  data/clasificador.csv  -> frase;etiquetas   (multi-label)
  data/corpus.txt        -> líneas "[CATEGORIA: X][CATEGORIA: Y] TÚ: ... -> EXCUSA: ..."
Edita semilla.txt para añadir frases, excusas o categorías nuevas y vuelve a ejecutar este script.
Las etiquetas salen de las categorías simples; las mixtas (A+B) activan dos etiquetas a la vez.
"""
import os, random, csv, sys

random.seed(42)
MIN_FRASES, MIN_EXCUSAS = 40, 30
PREFIJOS = ["", "", "perdona, ", "oye, ", "jefe, ", "lo siento pero ", "mira, ", "uf, "]
CONECTORES = [" y además ", " y encima ", ", y por si fuera poco, ", " y para colmo "]


def cargar_semilla(ruta="semilla.txt"):
    frases, excusas, destino = {}, {}, None
    for linea in open(ruta, encoding="utf-8"):
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        if linea.startswith("[") and linea.endswith("]"):
            tipo, cat = linea[1:-1].split(":")
            dic = frases if tipo.strip().upper() == "FRASES" else excusas
            destino = dic.setdefault(cat.strip().upper(), [])
        elif destino is not None:
            destino.append(linea)
    return frases, excusas


def validar(frases, excusas):
    ok = True
    for cat in sorted(set(frases) | set(excusas)):
        partes = cat.split("+")
        if partes != sorted(partes) or len(partes) > 2:
            print(f"ERROR: '{cat}' debe tener 1 o 2 etiquetas en orden alfabético"); ok = False
        if cat not in frases or cat not in excusas:
            print(f"ERROR: '{cat}' necesita sección FRASES y EXCUSAS"); ok = False; continue
        if len(frases[cat]) < MIN_FRASES: print(f"AVISO: {cat} tiene {len(frases[cat])} frases (< {MIN_FRASES})")
        if len(excusas[cat]) < MIN_EXCUSAS: print(f"AVISO: {cat} tiene {len(excusas[cat])} excusas (< {MIN_EXCUSAS})")
    for p in {x for c in frases for x in c.split("+")}:
        if p not in frases: print(f"ERROR: la etiqueta '{p}' aparece en una mixta pero no tiene categoría simple"); ok = False
    if not ok: sys.exit(1)


FRASES, EXCUSAS = cargar_semilla()
validar(FRASES, EXCUSAS)
CATEGORIAS = sorted(FRASES)
SIMPLES = [c for c in CATEGORIAS if "+" not in c]
ETIQUETAS = SIMPLES


def un_ejemplo(p_combo=0.15):
    """85%: categoría de la semilla (simple o mixta). 15%: dos categorías simples pegadas al azar."""
    if random.random() < p_combo:
        tags = sorted(random.sample(SIMPLES, 2))
        partes = [random.choice(FRASES[t]) for t in tags]
        frase = random.choice(PREFIJOS) + partes[0] + random.choice(CONECTORES) + partes[1]
        excusa = " ".join(random.choice(EXCUSAS[t]) for t in tags)
    else:
        cat = random.choice(CATEGORIAS)
        tags = cat.split("+")
        frase = random.choice(PREFIJOS) + random.choice(FRASES[cat])
        excusa = random.choice(EXCUSAS[cat])
    return tags, frase[0].upper() + frase[1:], excusa


def main(n_clasif=4000, n_corpus=8000):
    os.makedirs("data", exist_ok=True)
    with open("data/clasificador.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["frase", "etiquetas"])
        for _ in range(n_clasif):
            tags, frase, _ = un_ejemplo()
            w.writerow([frase, "|".join(tags)])
    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        for _ in range(n_corpus):
            tags, frase, excusa = un_ejemplo()
            f.write("".join(f"[CATEGORIA: {t}]" for t in tags) + f" TÚ: {frase} -> EXCUSA: {excusa}\n")
    n_f = sum(len(v) for v in FRASES.values()); n_e = sum(len(v) for v in EXCUSAS.values())
    print(f"Semilla: {len(CATEGORIAS)} categorías ({len(SIMPLES)} simples + {len(CATEGORIAS)-len(SIMPLES)} mixtas), {n_f} frases, {n_e} excusas.")
    print(f"OK: {n_clasif} frases para el clasificador, {n_corpus} líneas de corpus.")


if __name__ == "__main__":
    main()
