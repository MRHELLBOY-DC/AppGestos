# AppiGestos

**AppiGestos** es un proyecto innovador de traducción de Lenguaje de Señas en tiempo real. Utiliza Inteligencia Artificial (Redes Neuronales Convolucionales) para detectar letras del abecedario, y heurísticas avanzadas de seguimiento de manos (MediaPipe) para detectar oraciones completas de uso cotidiano. Además, cuenta con un sintetizador de voz (TTS) para hablar las frases o letras detectadas, facilitando la comunicación.

---

## Features (Características)

1. **Reconocimiento del Abecedario (IA)**
   - Utiliza un modelo pre-entrenado (VGG19 modificado) para reconocer estáticamente el abecedario de señas (A-Y) dentro del recuadro verde.
   - Las letras **J y Z** (que normalmente requieren movimiento) se detectan de forma heurística mediante posturas especiales fuera del recuadro.

2. **Acumulación e Interpretación de Texto**
   - El usuario puede ir capturando las letras reconocidas para formar palabras o nombres.
   - Cuenta con controles de teclado para añadir letras, espacios o borrar errores en tiempo real.

3. **Audio de la Interpretación (Google TTS)**
   - Convierte el texto formado o la oración detectada en voz de alta calidad (acento español) utilizando la librería `gTTS`.

4. **Detección Heurística de Oraciones (Smart Gestures)**
   - Fuera del recuadro verde del abecedario, el programa utiliza **MediaPipe Hands** para entender gestos dinámicos en 3D.
   - Reconoce posturas específicas (ej. puño en el pecho) para emitir oraciones completas como *"Tengo hambre"*, *"¿Cuál es tu nombre?"*, etc.

5. **Panel de Referencia Rápida**
   - Al iniciar, muestra un panel de ayuda visual con todas las imágenes de referencia para cada letra del abecedario, ideal para aprender a posicionar las manos.

---

## Cómo Ejecutar el Proyecto

### 1. Requisitos Previos
Asegúrate de tener instalado [Python 3.9+](https://www.python.org/downloads/) en tu computadora.

### 2. Preparación del Entorno Virtual (Recomendado)
Es altamente recomendado usar un entorno virtual para no crear conflictos con otras librerías de tu computadora.
Abre una terminal (PowerShell o CMD) en la carpeta principal del proyecto (`AppiGestos`) y ejecuta:

```powershell
# Crear el entorno virtual llamado 'venv'
python -m venv venv

# Activar el entorno virtual (PowerShell)
.\venv\Scripts\activate
```

### 3. Instalación de Dependencias
Una vez activado el entorno, instala todas las librerías necesarias con el archivo `requirements.txt`:

```powershell
pip install -r requirements.txt
```
*(Esto instalará TensorFlow, OpenCV, MediaPipe, gTTS, PyGame, Numpy, entre otros).*

### 4. Ejecutar la Aplicación
Para ejecutar el sistema principal, lanza el script `main.py`:

```powershell
python main.py
```

### Controles de la Aplicación
Al abrirse la ventana de video, asegúrate de tenerla seleccionada y usa las siguientes teclas:
- **`C`** : Capturar la letra actual (añadirla al texto).
- **`Espacio`** : Añadir un espacio en blanco al texto.
- **`Backspace`** : Borrar la última letra.
- **`L`** : Leer en voz alta el texto acumulado (TTS).
- **`Q`** : Salir del programa.

---

### Cómo Activar las 5 Oraciones y Letras Especiales (Smart Gestures)
Para que el sistema detecte estas acciones automáticamente mediante inteligencia artificial, tu mano debe estar **FUERA del recuadro verde** del abecedario. Mantén la postura indicada durante 1 segundo:

**Oraciones (Ingreso de Bloques):**
Al mantener estas posturas, la oración completa se agregará automáticamente al texto acumulado. Puedes borrar la oración entera de un solo toque usando la tecla `Backspace`.

1. **"Tengo mucha hambre, ¿podemos ir a comer algo?"**: Mantén el **puño cerrado** (sin dedos levantados) en la **parte inferior central** de tu cámara (cerca de tu pecho).
2. **"¡Hola! Yo soy Eduardo Durana, mucho gusto."**: Levanta **solo el dedo índice (apuntándote a ti mismo)** en la **parte central o inferior** de la cámara.
3. **"¡Bienvenidos a la presentación de nuestro proyecto de Algoritmica!"**: Muestra **tus DOS manos juntas y abiertas** en cualquier parte de la cámara (fuera del cuadro verde).
4. **"Estoy muy de acuerdo con lo que acaban de decir."**: Haz el gesto de **"OK" (👌 pulgar e índice tocándose, con los otros 3 dedos levantados)** en cualquier parte de la cámara.
5. **"¡Nos vemos luego, que tengan un excelente día!"**: Mantén la **mano abierta** en la **esquina superior derecha**.

**Letras Dinámicas (J y Z):**
Para escribir estas letras, haz el gesto y presiona la tecla `C` para capturarlas, igual que las demás:
- **Letra "J"**: Haz el gesto de **"Pulgar arriba" (Like)** en la **esquina inferior derecha** de tu pantalla.
- **Letra "Z"**: Haz el gesto de **"Amor y paz" (Índice y Medio levantados)** en la **esquina inferior derecha** de tu pantalla.
