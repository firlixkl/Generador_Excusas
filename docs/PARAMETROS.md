# Guía de parámetros

Qué se puede modificar en los paneles, qué hace cada valor y cómo afecta al modelo.

## 🔍 Panel de Inferencia
Ninguno de estos valores cambia los pesos del modelo: solo cómo se usa.

| Mando | Qué hace | Efecto |
|---|---|---|
| **Umbral** | Una etiqueta se activa si su probabilidad lo supera. Si ninguna lo hace, se usa la más probable. | Bajo: más etiquetas (y falsos positivos). Alto: menos. El corpus solo tiene 1 o 2 etiquetas por ejemplo; con 3 o más el generador sale peor. |
| **Casillas de etiquetas** | Fuerzan las etiquetas a mano. | Permiten ver cómo cambia la excusa con otras categorías. |
| **Temperatura** | Divide los logits antes del softmax. | `< 1`: distribución afilada, coherente pero repetitiva. `> 1`: aplanada, creativa y, si te pasas, con palabras inventadas (el modelo es de caracteres, así que se nota como erratas). |
| **Top-k** | Solo se sortea entre los k caracteres más probables. | `0` = desactivado; `1` = siempre el favorito. Permite subir la temperatura sin que salgan caracteres absurdos. |
| **Máx. caracteres** | Tope de longitud. | La generación también para si el modelo emite un salto de línea. |
| **Pausa por paso** | Velocidad visual. | No cambia el resultado. |
| **Semilla** | Fija el azar del sorteo. | Misma frase + mismos ajustes + misma semilla = misma excusa. |
| **Capa / cabeza** | Elige qué mapa de atención ver. | Solo visualización. |

## 🧪 Laboratorio · Generador (Mini-GPT)
Estos valores se fijan al entrenar: si cambias uno, hay que reentrenar.

| Parámetro | Qué es | Efecto |
|---|---|---|
| `block_size` | Caracteres que ve el modelo a la vez. | Debe cubrir una línea entera (el corpus llega a ~370). Si prompt + excusa superan la ventana, las etiquetas (al principio) salen del contexto. El coste de la atención crece con su cuadrado. |
| `n_layer` | Bloques Transformer apilados. | Más capacidad, parámetros y tiempo; más riesgo de memorizar. |
| `n_embd` | Longitud del vector de cada carácter y ancho del modelo. | Los parámetros crecen con su cuadrado. |
| `n_head` | Cabezas de atención en paralelo. | No añade parámetros; `n_embd` debe ser múltiplo de `n_head`. |
| `dropout` | Apaga neuronas al azar al entrenar. | Menos sobreajuste, aprendizaje más lento. Desactivado en inferencia. |
| `lr` | Tamaño del paso al corregir los pesos. | Es el máximo: calentamiento de 100 iteraciones y bajada coseno hasta el 10 %. Muy alto: la loss oscila o explota. Muy bajo: lento. Típico: 3e-4 a 3e-3. |
| `batch_size` | Secuencias por actualización. | Más: gradiente más limpio, más memoria y tiempo por paso. |
| `max_iters` | Pasos de entrenamiento. | Más aprendizaje con rendimientos decrecientes; también define la forma de la curva del lr. |
| `eval_interval` | Cada cuánto se mide train/val y se genera una muestra. | Solo resolución de las curvas; evaluar mucho ralentiza. |

## 🧪 Laboratorio · Clasificador
| Parámetro | Efecto |
|---|---|
| `lr` | Igual que arriba; lo normal es 1e-3 a 1e-2. |
| `batch_size` | Frases por actualización. |
| `epochs` | Pasadas completas al dataset. Con datos tan repetitivos, demasiadas épocas memorizan las plantillas. |

## 🔒 Fijo en el código (para experimentar editándolo)
- Clasificador: embedding 64, capa oculta 128, dropout 0.2, «tapar» el 10 % de las palabras al entrenar.
- Reparto 90 % entrenamiento / 10 % validación.
- Generador: 100 iteraciones de calentamiento y recorte de gradiente a 1.0.
- Corpus: 85 % semilla y 15 % combinaciones al azar (en `datos.py`).
