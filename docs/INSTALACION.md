# Guía de instalación y uso · Excusas ML

Contenido del proyecto (todo coherente entre sí, no mezcles versiones antiguas):

    semilla.txt       TUS DATOS: 12 categorías, 480 frases y 360 excusas (editable)
    datos.py          lee semilla.txt y genera data/clasificador.csv y data/corpus.txt
    clasificador.py   Paso A: clasificador multi-label
    generador.py      Paso B: Mini-GPT (nanoGPT)
    excusas.py        flujo completo por terminal (sin paneles)
    server.py         servidor de los paneles
    static/index.html los dos paneles (Inferencia y Laboratorio)
    requirements.txt  dependencias

Necesitas Python 3.10 o superior y ~1-2 GB libres (PyTorch pesa).

---------------------------------------------------------------
## WINDOWS (CMD)
---------------------------------------------------------------
1. Instala Python desde https://www.python.org/downloads/
   IMPORTANTE: marca la casilla "Add python.exe to PATH" en el instalador.
2. Descomprime el zip, abre CMD y entra en la carpeta:

       cd C:\ruta\donde\descomprimiste\generador-excusas-ml

3. Comprueba Python (si "python" no funciona, prueba "py"):

       python --version

4. Crea y activa el entorno virtual:

       python -m venv venv
       venv\Scripts\activate.bat

   (Verás "(venv)" al principio de la línea. En PowerShell se activa con venv\Scripts\Activate.ps1)

5. Instala las dependencias (la primera vez tarda):

       pip install --upgrade pip
       pip install -r requirements.txt

6. Genera los datos:

       python datos.py

7. Arranca el panel:

       python server.py

8. Abre en el navegador:  http://127.0.0.1:8000
   (Si Windows Defender pregunta por el firewall, permite el acceso privado.)
9. Para parar: Ctrl+C en la CMD. Para salir del entorno: deactivate

Siguientes veces solo necesitas: cd a la carpeta, venv\Scripts\activate.bat, python server.py

---------------------------------------------------------------
## KALI LINUX
---------------------------------------------------------------
Kali (como Debian reciente) no deja hacer "pip install" global: usa un entorno virtual.

1. Prepara el sistema:

       sudo apt update
       sudo apt install -y python3 python3-venv python3-pip

2. Entra en la carpeta del proyecto:

       cd ~/generador-excusas-ml

3. Crea y activa el entorno virtual:

       python3 -m venv venv
       source venv/bin/activate

4. Instala las dependencias. Recomendado si NO tienes GPU NVIDIA (mucho más ligero):

       pip install --upgrade pip
       pip install torch --index-url https://download.pytorch.org/whl/cpu
       pip install fastapi uvicorn psutil

   Si tienes GPU NVIDIA con drivers: pip install -r requirements.txt

5. Genera los datos y arranca:

       python datos.py
       python server.py

6. Abre en el navegador:  http://127.0.0.1:8000
   Si Kali corre en una máquina virtual, abre el navegador DENTRO de la VM.
   Para parar: Ctrl+C. Para salir del entorno: deactivate

Siguientes veces: cd ~/generador-excusas-ml && source venv/bin/activate && python server.py

---------------------------------------------------------------
## ORDEN DE USO (igual en ambos sistemas)
---------------------------------------------------------------
1. Pestaña "Laboratorio" -> botón "Clasificador" -> Entrenar (segundos).
2. Pestaña "Laboratorio" -> botón "Generador" -> elige el preset "CPU rápida"
   si no tienes GPU -> pulsa "Medir velocidad de mi máquina" -> Entrenar.
3. Pestaña "Inferencia" -> escribe una frase -> "Ejecutar todo".

Si prefieres terminal, sin paneles:

    python clasificador.py
    python generador.py --n_layer 2 --n_embd 64 --block_size 128 --batch_size 16 --max_iters 1500
    python excusas.py --temperature 0.8

---------------------------------------------------------------
## PROBLEMAS FRECUENTES
---------------------------------------------------------------
- "python no se reconoce" (Windows): reinstala marcando "Add to PATH" o usa "py".
- "No hay clasificador/generador entrenado": entrénalos en el Laboratorio (genera
  clasificador.pt y generador.pt en la carpeta).
- "Falta data/corpus.txt": ejecuta python datos.py.
- Puerto 8000 ocupado: cierra el otro servidor, o cambia port=8000 al final de server.py.
- Las gráficas no aparecen: Chart.js se carga desde internet; comprueba tu conexión.
- Entrenar tarda horas: usa el preset "CPU rápida" o baja max_iters / n_layer / n_embd.
- Cambiaste el tamaño del modelo y falla al cargar: vuelve a entrenar el generador.

---------------------------------------------------------------
## EDITAR TUS DATOS (semilla.txt)
---------------------------------------------------------------
- Una frase o excusa por línea, bajo [FRASES:CATEGORIA] y [EXCUSAS:CATEGORIA].
- Categorías simples: CLIMA, COMPRAS, MASCOTA, PAREJA, SALUD, VEHICULO.
- Categorías mixtas (2 etiquetas): CLIMA+VEHICULO, COMPRAS+PAREJA, MASCOTA+SALUD,
  MASCOTA+VEHICULO, PAREJA+SALUD, CLIMA+COMPRAS (siempre en orden alfabético).
- Cada categoría necesita ambas secciones (mínimo recomendado: 40 frases y 30 excusas).
- Tras editar: python datos.py y vuelve a entrenar clasificador y generador
  (los modelos antiguos no conocen los datos nuevos).
- datos.py avisa si falta alguna sección, el orden es incorrecto o no llegas al mínimo.
- Además de las categorías de la semilla, el corpus incluye un 15% de ejemplos que
  combinan dos categorías simples al azar, para que el modelo aprenda a mezclar.

---------------------------------------------------------------
## ACTUALIZAR A UNA VERSIÓN NUEVA SIN PERDER NADA
---------------------------------------------------------------
1. Para el servidor (Ctrl+C).
2. Descomprime el zip ENCIMA de tu carpeta actual, aceptando sobrescribir.
   (El zip no contiene los .pt ni la carpeta data, así que tus modelos entrenados se conservan.)
3. Arranca de nuevo: python server.py
4. Recarga el navegador con Ctrl+F5.
5. Comprueba la versión: arriba, junto al título, debe poner "v4 · línea temporal".
   Si no la ves, estás sirviendo una carpeta antigua: mira desde qué carpeta ejecutas server.py.

---------------------------------------------------------------
## USAR LA GPU (NVIDIA)
---------------------------------------------------------------
Comprobar si PyTorch ve tu GPU (también aparece arriba a la derecha en el panel):

    python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

- Si imprime False: tienes PyTorch solo-CPU. En Windows, "pip install torch" instala la versión CPU.
  Entra en https://pytorch.org/get-started/locally/, elige Pip + Windows (o Linux) + tu CUDA y
  ejecuta el comando que te dé (tiene esta forma):

      pip uninstall -y torch
      pip install torch --index-url https://download.pytorch.org/whl/cuXXX

  (cuXXX = versión de CUDA que indique la web; necesitas drivers NVIDIA al día.)

- Medir el uso real:
  Windows -> Administrador de tareas > Rendimiento > GPU: cambia el gráfico "3D" por "Cuda"
  (el 3D no mide el cálculo de PyTorch). Kali/Linux -> nvidia-smi -l 1

- Por qué no llega al 100%: el modelo por defecto es pequeño (<1M de parámetros). Una GPU
  moderna lo termina en microsegundos y pasa tiempo esperando a Python. Para exprimirla en el
  Laboratorio sube batch_size (256-512) y/o n_embd (256) y n_layer (6): usará más GPU y mejor
  modelo, pero más riesgo de memorizar y más VRAM (mira la tarjeta "Memoria" de la previsión).
