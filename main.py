# -*- coding: utf-8 -*-
"""
AppiGestos - Detección y Traducción de Lenguaje de Señas (Abecedario y Oraciones)
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from gtts import gTTS
import pygame
import threading
import io
import time
import textwrap

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================

# Ruta del modelo pre-entrenado
MODEL_PATH = 'modelo_appigestos.h5'

# Alfabeto soportado por el modelo (excluyendo J y Z porque requieren movimiento)
ALFABETO = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# Directorio que contiene las imágenes de ejemplo para cada seña
DIRECTORIO_SIGNOS = 'Signos'

# Coordenadas de la región de interés (ROI) donde se pondrá la mano para el abecedario
ARRIBA, DERECHA, ABAJO, IZQUIERDA = 100, 100, 400, 400

# (Las inicializaciones se movieron a main() para evitar conflictos de memoria)

# ==========================================
# FUNCIONES AUXILIARES
# ==========================================

def limpiar_texto_para_cv2(texto):
    """Limpia caracteres especiales y tildes para que OpenCV pueda mostrarlos bien."""
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N', '¿': '', '?': '?', '¡': '', '!': '!'
    }
    for orig, nuevo in reemplazos.items():
        texto = texto.replace(orig, nuevo)
    return texto

def dibujar_marco_elegante(img, x1, y1, x2, y2, color=(0, 255, 0), grosor_esquinas=4, longitud=40):
    """Dibuja un rectángulo fino con esquinas gruesas tipo visor tech."""
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # Rectángulo principal muy fino
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 1)
    
    # Esquinas gruesas
    cv2.line(img, (x_min, y_min), (x_min + longitud, y_min), color, grosor_esquinas)
    cv2.line(img, (x_min, y_min), (x_min, y_min + longitud), color, grosor_esquinas)
    
    cv2.line(img, (x_max, y_min), (x_max - longitud, y_min), color, grosor_esquinas)
    cv2.line(img, (x_max, y_min), (x_max, y_min + longitud), color, grosor_esquinas)
    
    cv2.line(img, (x_min, y_max), (x_min + longitud, y_max), color, grosor_esquinas)
    cv2.line(img, (x_min, y_max), (x_min, y_max - longitud), color, grosor_esquinas)
    
    cv2.line(img, (x_max, y_max), (x_max - longitud, y_max), color, grosor_esquinas)
    cv2.line(img, (x_max, y_max), (x_max, y_max - longitud), color, grosor_esquinas)

def procesar_roi(roi, k=2):
    """
    Redimensiona y normaliza la región de interés para que el modelo pueda procesarla.
    """
    roi_redimensionado = cv2.resize(roi, (28 * k, 28 * k), interpolation=cv2.INTER_AREA)
    roi_normalizado = roi_redimensionado / 255.0
    return roi_normalizado

def crear_panel_referencia(directorio):
    """
    Lee todas las imágenes del directorio, las redimensiona y las une en una 
    sola cuadrícula (grid) para mostrarlas juntas como referencia constante.
    """
    if not os.path.exists(directorio):
        return None
    
    archivos = sorted([f for f in os.listdir(directorio) if f.endswith(('.jpg', '.png'))])
    if not archivos:
        return None
    
    columnas = 6
    filas = 4
    tamano_img = 100 
    
    imagenes = []
    for arch in archivos:
        img_path = os.path.join(directorio, arch)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (tamano_img, tamano_img))
            letra = str(arch[0])
            cv2.putText(img, letra, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            imagenes.append(img)
            
    while len(imagenes) < (columnas * filas):
        img_vacia = np.zeros((tamano_img, tamano_img, 3), dtype=np.uint8)
        imagenes.append(img_vacia)
        
    filas_grid = []
    for i in range(filas):
        fila = np.hstack(imagenes[i * columnas : (i + 1) * columnas])
        filas_grid.append(fila)
        
    panel_final = np.vstack(filas_grid)
    
    # Agregar panel de controles abajo
    alto_controles = 120
    panel_controles = np.zeros((alto_controles, panel_final.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(panel_controles, "--- CONTROLES ---", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(panel_controles, "C: Capturar letra | ESPACIO: Separar palabra", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(panel_controles, "BACKSPACE: Borrar ultima | L: Leer en voz alta", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(panel_controles, "Q: Salir del programa", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    panel_final = np.vstack([panel_final, panel_controles])
    
    return panel_final

def hablar_texto_async(texto):
    """
    Reproduce el texto usando Google TTS en un hilo separado para no trabar el video.
    """
    def _hablar():
        try:
            tts = gTTS(text=texto, lang='es')
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            pygame.mixer.music.load(fp, 'mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print("Error reproduciendo audio:", e)
    
    # Solo iniciar si no hay otro audio sonando
    if not pygame.mixer.music.get_busy():
        threading.Thread(target=_hablar, daemon=True).start()

def estado_dedos(hand_landmarks, mp_hands):
    """
    Evalúa si cada dedo está abierto o cerrado.
    Retorna lista [Pulgar, Indice, Medio, Anular, Menique]
    """
    abiertos = []
    # Pulgar (heurística simplificada basada en la posición de la punta vs la articulación)
    abiertos.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
    # Otros dedos (evaluando si la yema está más arriba (menor Y) que la articulación media)
    abiertos.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
    abiertos.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
    abiertos.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y)
    abiertos.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)
    return abiertos

def detectar_oracion_heuristica(resultados_mp, ancho, alto, mp_hands):
    """
    Detecta 5 oraciones basadas en la posición y forma de las manos
    fuera del recuadro verde (ROI).
    """
    if not resultados_mp.multi_hand_landmarks:
        return None
    
    num_manos = len(resultados_mp.multi_hand_landmarks)
    
    for hand_landmarks in resultados_mp.multi_hand_landmarks:
        # Obtener coordenadas de la muñeca (para las heurísticas)
        x_muneca = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * ancho)
        y_muneca = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * alto)
        
        # Obtener coordenadas del centro de la mano (nudillo del dedo medio) para exclusión del ROI
        x_centro = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * ancho)
        y_centro = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * alto)
        
        # Ignorar la mano si su centro está dentro o muy cerca del recuadro verde del abecedario
        if DERECHA - 50 < x_centro < IZQUIERDA + 50 and ARRIBA - 50 < y_centro < ABAJO + 50:
            continue
            
        dedos = estado_dedos(hand_landmarks, mp_hands)
        # Contar dedos levantados excluyendo el pulgar (índices 1 al 4)
        dedos_levantados = sum(dedos[1:]) 
        
        # 1. "Tengo mucha hambre, ¿podemos ir a comer algo?": Puño cerrado (0 dedos) en la mitad inferior de la pantalla (pecho)
        if dedos_levantados == 0 and y_muneca > alto * 0.6:
            return "Tengo mucha hambre, ¿podemos ir a comer algo?"
            
        # 2. "¡Hola! Yo soy Eduardo Durana": Solo dedo índice levantado (apuntándose a sí mismo) en la mitad inferior y no en el borde izquierdo
        if dedos[1] and dedos_levantados == 1 and y_muneca > alto * 0.5 and x_muneca > ancho * 0.4:
            return "¡Hola! Yo soy Eduardo Durana, mucho gusto."
            
        # 3. Presentación: Dos manos detectadas y ambas con dedos abiertos
        if num_manos == 2 and dedos_levantados >= 3:
            return "¡Bienvenidos a la presentación de nuestro proyecto de Algoritmica!"
            
        # 4. Acuerdo: Gesto de "OK" (Pulgar e índice tocándose, otros 3 dedos levantados) en cualquier lugar
        pulgar_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        pulgar_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
        indice_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
        indice_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        distancia_ok = ((pulgar_x - indice_x)**2 + (pulgar_y - indice_y)**2)**0.5
        
        if distancia_ok < 0.06 and dedos[2] and dedos[3] and dedos[4]:
            return "Estoy muy de acuerdo con lo que acaban de decir."
            
        # 5. Despedida: Mano abierta en la esquina superior derecha
        if dedos_levantados >= 3 and x_muneca > ancho * 0.6 and y_muneca < alto * 0.5:
            return "¡Nos vemos luego, que tengan un excelente día!"
            
    return None

def detectar_letra_heuristica(resultados_mp, ancho, alto, mp_hands):
    """
    Detecta las letras J y Z heurísticamente usando gestos especiales
    fuera del recuadro verde (ROI).
    """
    if not resultados_mp.multi_hand_landmarks:
        return None
        
    for hand_landmarks in resultados_mp.multi_hand_landmarks:
        # Obtener coordenadas de la muñeca
        x_muneca = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * ancho)
        y_muneca = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * alto)
        
        # Obtener centro para la exclusión
        x_centro = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * ancho)
        y_centro = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * alto)
        
        # Ignorar si está dentro del ROI verde
        if DERECHA - 50 < x_centro < IZQUIERDA + 50 and ARRIBA - 50 < y_centro < ABAJO + 50:
            continue
            
        dedos = estado_dedos(hand_landmarks, mp_hands)
        dedos_levantados = sum(dedos[1:])
        
        # Medir si el pulgar apunta hacia arriba comparando su punta con su base
        pulgar_arriba = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y

        # "J": Pulgar arriba (como dar "Like") en la esquina inferior derecha
        if pulgar_arriba and dedos_levantados == 0 and x_muneca > ancho * 0.6 and y_muneca > alto * 0.6:
            return "J"
            
        # "Z": Amor y paz (índice y medio levantados) en la esquina inferior derecha
        if dedos[1] and dedos[2] and dedos_levantados == 2 and x_muneca > ancho * 0.6 and y_muneca > alto * 0.6:
            return "Z"
            
    return None

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    print("Iniciando Pygame y MediaPipe...")
    pygame.mixer.init()
    
    mp_hands = mp.solutions.hands
    manos_mp = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_dibujo = mp.solutions.drawing_utils

    print("Cargando el modelo de AppiGestos... Por favor, espera.")
    try:
        modelo = tf.keras.models.load_model(MODEL_PATH)
        print("¡Modelo cargado con éxito!")
    except Exception as e:
        print(f"Error crítico al cargar el modelo: {e}")
        return

    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Mostrar panel de referencia
    panel_senas = crear_panel_referencia(DIRECTORIO_SIGNOS)
    if panel_senas is not None:
        cv2.imshow("Alfabeto Completo (Referencia)", panel_senas)
    else:
        print(f"Advertencia: No se pudo generar el panel de referencias. Verifica la carpeta '{DIRECTORIO_SIGNOS}'.")

    print("\n--- CONTROLES ---")
    print("C       : Capturar letra al texto")
    print("ESPACIO : Separar palabra")
    print("BACKSPC : Borrar ultima letra")
    print("L       : Leer texto en voz alta (Google TTS)")
    print("Q       : Salir del programa.")
    print("-----------------\n")

    # Variables para Interpretación de Texto
    elementos_texto = []
    
    # Variables para suavizar la detección de oraciones y evitar spam de audio
    conteo_oracion = {}
    oracion_actual = None
    ultimo_audio_time = 0
    COOLDOWN_AUDIO = 5 # Segundos a esperar antes de repetir una oración automática

    while True:
        capturado, fotograma = camara.read()
        if not capturado:
            print("Error: No se pudo acceder a la cámara.")
            break

        fotograma = cv2.flip(fotograma, 1)
        fotograma_clon = fotograma.copy()
        alto, ancho = fotograma.shape[:2]

        # ------------------------------------------------
        # 1. DETECCIÓN DEL ABECEDARIO (ROI VERDE)
        # ------------------------------------------------
        roi = fotograma[ARRIBA:ABAJO, DERECHA:IZQUIERDA]
        roi_procesado = procesar_roi(roi, k=2)
        roi_tensor = roi_procesado.reshape(-1, 56, 56, 3)
        
        # verbose=0 evita spam en la consola
        predicciones = modelo.predict(roi_tensor, verbose=0)
        
        indices_ordenados = np.argsort(predicciones[0])
        # Solo necesitamos la letra con mayor probabilidad para capturar
        letra_principal = ALFABETO[indices_ordenados[-1]]

        dibujar_marco_elegante(fotograma_clon, DERECHA, ARRIBA, IZQUIERDA, ABAJO, color=(0, 255, 0), grosor_esquinas=4, longitud=40)
        
        # ------------------------------------------------
        # 2. DETECCIÓN HEURÍSTICA DE ORACIONES Y LETRAS (MEDIAPIPE)
        # ------------------------------------------------
        rgb_frame = cv2.cvtColor(fotograma, cv2.COLOR_BGR2RGB)
        resultados_mp = manos_mp.process(rgb_frame)
        
        # Dibujar puntos de las manos para feedback visual
        if resultados_mp.multi_hand_landmarks:
            for hand_landmarks in resultados_mp.multi_hand_landmarks:
                mp_dibujo.draw_landmarks(fotograma_clon, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # Reemplazar la letra del abecedario estático si se detecta J o Z
        letra_especial = detectar_letra_heuristica(resultados_mp, ancho, alto, mp_hands) if resultados_mp else None
        if letra_especial:
            letra_principal = letra_especial
            
        # Centrar la letra principal arriba del recuadro
        (text_width, text_height), _ = cv2.getTextSize(letra_principal, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        centro_x = (IZQUIERDA + DERECHA) // 2
        x_letra = centro_x - (text_width // 2)
        y_letra = ARRIBA - 15
        
        cv2.putText(fotograma_clon, letra_principal, (x_letra, y_letra), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
        # Clasificar la postura
        oracion_detectada = detectar_oracion_heuristica(resultados_mp, ancho, alto, mp_hands)
        
        if oracion_detectada:
            conteo_oracion[oracion_detectada] = conteo_oracion.get(oracion_detectada, 0) + 1
            # Si se mantiene la misma postura por ~15 fotogramas (aprox. 0.5 a 1 segundo)
            if conteo_oracion[oracion_detectada] > 15:
                # Verificar cooldown para no repetir audio
                if time.time() - ultimo_audio_time > COOLDOWN_AUDIO:
                    print(f"\n[!] Oración Detectada: {oracion_detectada}")
                    oracion_actual = oracion_detectada
                    # Agregar la oración como un solo bloque
                    elementos_texto.append(f" {oracion_detectada} ")
                    ultimo_audio_time = time.time()
                # Limpiar el conteo después de activarlo
                conteo_oracion[oracion_detectada] = 0
        else:
            # Reducir el conteo gradualmente si se pierde la postura para evitar falsos positivos rápidos
            for key in conteo_oracion:
                if conteo_oracion[key] > 0:
                    conteo_oracion[key] -= 1
                    
        # Borrar el texto de la oración emergente de la pantalla después de 3 segundos
        if oracion_actual and time.time() - ultimo_audio_time > 3:
            oracion_actual = None

        # ------------------------------------------------
        # 3. DIBUJAR TEXTO Y MENSAJES EN PANTALLA
        # ------------------------------------------------
        # Envolvemos el texto acumulado
        texto_acumulado = "".join(elementos_texto)
        texto_limpio = limpiar_texto_para_cv2(texto_acumulado)
        texto_completo = f"Texto: {texto_limpio}"
        
        # Envolvemos el texto en líneas de máximo 80 caracteres (para 1280px)
        lineas_texto = textwrap.wrap(texto_completo, width=80)
        
        # Preparamos las líneas de la oración dinámica si existe
        lineas_oracion = []
        if oracion_actual:
            oracion_limpia = limpiar_texto_para_cv2(oracion_actual)
            lineas_oracion = textwrap.wrap(f"Oracion: {oracion_limpia}", width=80)
            
        # Calcular el alto del panel de texto (mínimo 150 píxeles, se expande si hay mucho texto)
        total_lineas = len(lineas_texto) + len(lineas_oracion)
        alto_panel = max(150, (total_lineas + 1) * 35)
        
        # Crear el panel negro para el texto
        panel_texto = np.zeros((alto_panel, ancho, 3), dtype=np.uint8)
        
        # Dibujar la oración dinámica en el panel
        y_offset = 40
        for linea in lineas_oracion:
            cv2.putText(panel_texto, linea, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            y_offset += 35
            
        # Dibujar el texto acumulado en el panel
        if lineas_oracion:
            y_offset += 15 # Pequeño espacio de separación
            
        for linea in lineas_texto:
            cv2.putText(panel_texto, linea, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 35
            
        # Unir el fotograma de la cámara con el panel de texto
        pantalla_final = np.vstack((fotograma_clon, panel_texto))

        cv2.imshow("Video Feed - AppiGestos", pantalla_final)

        # ------------------------------------------------
        # 4. CONTROLES DE TECLADO
        # ------------------------------------------------
        tecla = cv2.waitKey(1) & 0xFF
        
        if tecla == ord("q") or tecla == ord("Q"):
            break
        elif tecla == ord("c") or tecla == ord("C"):
            elementos_texto.append(letra_principal)
            print(f"Texto actual: {''.join(elementos_texto)}")
        elif tecla == 32: # ESPACIO
            elementos_texto.append(" ")
            print(f"Texto actual: {''.join(elementos_texto)}")
        elif tecla == 8:  # BACKSPACE
            if elementos_texto:
                elementos_texto.pop()
            print(f"Texto actual: {''.join(elementos_texto)}")
        elif tecla == ord("l") or tecla == ord("L"):
            texto_hablar = "".join(elementos_texto).strip()
            if texto_hablar != "":
                print(f"Hablando texto: {texto_hablar}")
                hablar_texto_async(texto_hablar)

    camara.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()