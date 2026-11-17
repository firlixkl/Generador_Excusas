"""Pegamento: usuario -> clasificador -> prompt -> generador -> excusa."""
import argparse, sys
from clasificador import cargar_modelo, predecir
from generador import cargar_generador, generar_excusa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temperature", type=float, default=0.8, help="0.3 sensato · 0.8 gracioso · 1.2 locura")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--umbral", type=float, default=0.5)
    a = p.parse_args()
    clf, vocab, etiquetas = cargar_modelo()
    gpt, chars = cargar_generador()
    print("Escribe tu marrón (línea vacía para salir):")
    for linea in sys.stdin:
        frase = linea.strip()
        if not frase: break
        tags, probs = predecir(frase, clf, vocab, etiquetas, a.umbral)
        print(f"  [clasificador] {tags}  {probs}")
        print(f"  EXCUSA: {generar_excusa(gpt, chars, tags, frase, a.temperature, a.top_k)}\n")


if __name__ == "__main__":
    main()
