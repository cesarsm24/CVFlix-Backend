"""
video_utils.py

Utilidades para procesamiento, análisis y manipulación de archivos de video.
Incluye extracción de metadatos, detección de cambios de escena, cálculo de
configuraciones óptimas y estimación de tiempos de procesamiento.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
from typing import Tuple, Optional, List
import numpy as np


def get_video_info(video_path: str) -> Optional[dict]:
    """
    Extrae metadatos técnicos completos de archivo de video.

    Args:
        video_path: Ruta al archivo de video.

    Returns:
        Diccionario con información del video:
            width (int): Ancho en píxeles.
            height (int): Alto en píxeles.
            fps (float): Frame rate (frames por segundo).
            total_frames (int): Número total de frames.
            duration (float): Duración en segundos.
            codec (int): Código FourCC del codec.
        None si el video no puede abrirse.

    Notes:
        Utiliza propiedades CAP_PROP de OpenCV para metadatos. La duración
        se calcula como total_frames/fps. El codec se retorna como entero
        FourCC que puede convertirse a string con fourcc_to_string().
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": 0,
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    }

    if info["fps"] > 0:
        info["duration"] = info["total_frames"] / info["fps"]

    cap.release()
    return info


def get_optimal_skip_rate(video_path: str, target_analysis_time: float = 120) -> int:
    """
    Calcula skip rate óptimo basado en duración del video.

    Ajusta frecuencia de muestreo para completar análisis en tiempo objetivo
    manteniendo calidad mínima de análisis.

    Args:
        video_path: Ruta al archivo de video.
        target_analysis_time: Tiempo objetivo de análisis en segundos.
            Por defecto 120s (2 minutos).

    Returns:
        Skip rate recomendado (procesar 1 de cada N frames).

    Notes:
        Estrategia de muestreo por duración:
            - <2 min: skip=10 (análisis denso, alta calidad)
            - 2-10 min: skip=15 (análisis estándar)
            - >10 min: skip calculado para target_analysis_time

        El skip rate se limita a [15, 60] para evitar:
            - Skip <15: procesamiento excesivamente lento
            - Skip >60: pérdida significativa de eventos temporales

        Con target 120s y skip calculado, se procesa ~5 fps efectivos
        (600 frames en 120 segundos de procesamiento).
    """
    info = get_video_info(video_path)
    if not info:
        return 30

    duration = info['duration']

    if duration < 120:
        return 10
    elif duration < 600:
        return 15
    else:
        total_frames = info['total_frames']
        frames_to_analyze = int(target_analysis_time * 5)
        skip = max(15, total_frames // frames_to_analyze)
        return min(skip, 60)


def extract_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """
    Extrae frame específico en timestamp dado.

    Args:
        video_path: Ruta al archivo de video.
        time_seconds: Timestamp en segundos.

    Returns:
        Frame BGR como numpy array, o None si hay error o timestamp inválido.

    Notes:
        Utiliza CAP_PROP_POS_MSEC para posicionamiento preciso. Más confiable
        que CAP_PROP_POS_FRAMES en videos con frame rate variable. Si el
        timestamp excede duración del video, retorna None.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    time_ms = time_seconds * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def extract_key_frames(video_path: str, max_frames: int = 50) -> List[Tuple[int, np.ndarray]]:
    """
    Extrae frames clave distribuidos uniformemente en el video.

    Útil para análisis previo rápido, generación de thumbnails o creación
    de storyboards sin procesar video completo.

    Args:
        video_path: Ruta al archivo de video.
        max_frames: Número máximo de frames a extraer.

    Returns:
        Lista de tuplas (frame_number, frame_array) ordenadas por posición
        temporal. Lista vacía si hay error.

    Notes:
        Los frames se distribuyen uniformemente calculando intervalo como
        total_frames/max_frames. Esta estrategia simple es más rápida que
        detección de cambios de escena pero puede perder eventos importantes
        entre intervalos.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)

    key_frames = []
    frame_nums = range(0, total_frames, interval)

    for frame_num in frame_nums[:max_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            key_frames.append((frame_num, frame))

    cap.release()
    return key_frames


def extract_frames_batch(video_path: str, frame_numbers: list) -> list:
    """
    Extrae múltiples frames específicos en un solo pase.

    Args:
        video_path: Ruta al archivo de video.
        frame_numbers: Lista de índices de frames a extraer.

    Returns:
        Lista de frames en mismo orden que frame_numbers. Posiciones con
        errores contienen None.

    Notes:
        Los frame_numbers se ordenan antes de extracción para minimizar
        seeks del video file. Más eficiente que múltiples llamadas a
        extract_frame_at_time() cuando se necesitan varios frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return frames

    for frame_num in sorted(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            frames.append(None)

    cap.release()
    return frames


def create_video_writer(output_path: str, fps: float,
                        frame_size: Tuple[int, int],
                        codec: str = 'mp4v') -> cv2.VideoWriter:
    """
    Crea VideoWriter configurado para guardar video.

    Args:
        output_path: Ruta donde guardar archivo de salida.
        fps: Frame rate del video de salida.
        frame_size: Dimensiones (width, height) de frames.
        codec: Código de 4 caracteres del codec. Por defecto 'mp4v' (MPEG-4).

    Returns:
        VideoWriter configurado y listo para escritura de frames.

    Notes:
        Codecs comunes:
            - 'mp4v': MPEG-4, buena compatibilidad
            - 'avc1': H.264, mejor compresión pero requiere libs
            - 'XVID': Xvid codec, open source
            - 'MJPG': Motion JPEG, sin compresión inter-frame
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return writer


def calculate_video_bitrate(video_path: str) -> Optional[float]:
    """
    Calcula bitrate promedio del video.

    Args:
        video_path: Ruta al archivo de video.

    Returns:
        Bitrate en kilobits por segundo (kbps), o None si hay error.

    Notes:
        El cálculo es aproximado: (file_size_bytes * 8) / duration_seconds.
        No distingue entre bitrate de video y audio. Bitrates típicos:
            - 720p: 1500-4000 kbps
            - 1080p: 4000-8000 kbps
            - 4K: 15000-40000 kbps
    """
    import os

    if not os.path.exists(video_path):
        return None

    file_size = os.path.getsize(video_path)

    info = get_video_info(video_path)
    if not info or info["duration"] == 0:
        return None

    bitrate_bps = (file_size * 8) / info["duration"]
    bitrate_kbps = bitrate_bps / 1000

    return round(bitrate_kbps, 2)


def split_video_into_scenes(video_path: str, threshold: float = 30.0,
                            min_scene_length: int = 30) -> list:
    """
    Detecta cambios de escena mediante análisis de diferencia entre frames.

    Args:
        video_path: Ruta al archivo de video.
        threshold: Umbral de diferencia promedio para detectar corte [0-255].
            Valores típicos: 25-35.
        min_scene_length: Longitud mínima de escena en frames para prevenir
            detecciones falsas por flashs o movimientos bruscos.

    Returns:
        Lista de números de frame donde ocurren cambios de escena.
        Siempre incluye frame 0 como inicio de primera escena.

    Notes:
        Algoritmo:
            1. Convierte frames a escala de grises 320x180 (velocidad)
            2. Calcula diferencia absoluta con frame anterior
            3. Si diferencia > threshold y min_scene_length cumplido → corte

        La reducción a 320x180 acelera cálculo ~10x con precisión suficiente.
        min_scene_length previene falsos positivos por:
            - Flashs de cámara
            - Movimientos muy rápidos de cámara
            - Cambios bruscos de iluminación dentro de escena
    """
    cap = cv2.VideoCapture(video_path)
    scene_changes = [0]

    if not cap.isOpened():
        return scene_changes

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return scene_changes

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (320, 180))

    frame_num = 1
    last_scene_change = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = diff.mean()

        if mean_diff > threshold and (frame_num - last_scene_change) >= min_scene_length:
            scene_changes.append(frame_num)
            last_scene_change = frame_num

        prev_gray = gray
        frame_num += 1

    cap.release()
    return scene_changes


def detect_scene_changes_fast(video_path: str, sample_rate: int = 30) -> list:
    """
    Detección rápida de cambios de escena mediante muestreo y histogramas.

    Versión optimizada que procesa solo 1 de cada sample_rate frames usando
    comparación de histogramas en lugar de diferencia pixel-a-pixel.

    Args:
        video_path: Ruta al archivo de video.
        sample_rate: Analizar cada N frames. Por defecto 30.

    Returns:
        Lista de números de frame con cambios de escena detectados.

    Notes:
        Optimizaciones:
            - Resolución 160x90 (16:9 aspect ratio mínimo)
            - Histograma de 16 bins (vs 256) para velocidad
            - Correlación de histogramas más robusta que diferencia de píxeles

        La correlación < 0.7 indica cambio significativo de distribución de
        intensidades, más robusto ante movimiento de cámara que diferencia
        directa. Más rápido que split_video_into_scenes() pero menos preciso
        en frame exacto de corte.
    """
    cap = cv2.VideoCapture(video_path)
    scene_changes = [0]

    if not cap.isOpened():
        return scene_changes

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return scene_changes

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (160, 90))

    frame_num = sample_rate

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 90))

        hist_prev = cv2.calcHist([prev_gray], [0], None, [16], [0, 256])
        hist_curr = cv2.calcHist([gray], [0], None, [16], [0, 256])

        correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

        if correlation < 0.7:
            scene_changes.append(frame_num)

        prev_gray = gray
        frame_num += sample_rate

    cap.release()
    return scene_changes


def estimate_video_quality(video_path: str) -> dict:
    """
    Clasifica calidad del video según resolución y frame rate.

    Args:
        video_path: Ruta al archivo de video.

    Returns:
        Diccionario con clasificaciones:
            resolution (str): Resolución en formato "WxH".
            resolution_quality (str): Clasificación de resolución.
            fps (float): Frame rate del video.
            fps_quality (str): Clasificación de frame rate.
            total_frames (int): Número total de frames.
            duration (float): Duración en segundos.

    Notes:
        Clasificación de resolución:
            - 4K/Ultra HD: ≥3840x2160 píxeles
            - Full HD: ≥1920x1080 píxeles
            - HD: ≥1280x720 píxeles
            - SD: ≥854x480 píxeles
            - Low: <480p

        Clasificación de FPS:
            - Alta: ≥60 fps (video deportes, gaming)
            - Estándar: 30 fps (video web, TV)
            - Cinematográfica: 24 fps (películas)
            - Baja: <24 fps (video antiguo, cámaras económicas)
    """
    info = get_video_info(video_path)
    if not info:
        return {}

    width, height = info["width"], info["height"]
    pixels = width * height

    if pixels >= 3840 * 2160:
        resolution_quality = "4K/Ultra HD"
    elif pixels >= 1920 * 1080:
        resolution_quality = "Full HD (1080p)"
    elif pixels >= 1280 * 720:
        resolution_quality = "HD (720p)"
    elif pixels >= 854 * 480:
        resolution_quality = "SD (480p)"
    else:
        resolution_quality = "Low Resolution"

    fps = info["fps"]
    if fps >= 60:
        fps_quality = "Alta (60+ fps)"
    elif fps >= 30:
        fps_quality = "Estándar (30 fps)"
    elif fps >= 24:
        fps_quality = "Cinematográfica (24 fps)"
    else:
        fps_quality = "Baja (<24 fps)"

    return {
        "resolution": f"{width}x{height}",
        "resolution_quality": resolution_quality,
        "fps": fps,
        "fps_quality": fps_quality,
        "total_frames": info["total_frames"],
        "duration": round(info["duration"], 2)
    }


def calculate_optimal_settings(video_path: str) -> dict:
    """
    Calcula configuración óptima de procesamiento según características del video.

    Args:
        video_path: Ruta al archivo de video.

    Returns:
        Diccionario con configuración recomendada:
            face_detection_skip (int): Frames entre detecciones faciales.
            full_analysis_skip (int): Frames entre análisis completos.
            max_frame_width (int): Ancho máximo para procesamiento.
            jpeg_quality (int): Calidad JPEG para transmisión [0-100].
            priority (str): Prioridad del perfil (quality/balanced/speed).

    Notes:
        Perfiles por duración:
            - <2 min (corto): prioriza calidad con skip bajo
            - 2-10 min (mediano): balance entre calidad y velocidad
            - >10 min (largo): prioriza velocidad con skip alto

        El max_frame_width se reduce en videos largos para acelerar
        procesamiento. La calidad JPEG se ajusta para reducir latencia
        de transmisión WebSocket en videos largos.
    """
    info = get_video_info(video_path)
    if not info:
        return {}

    duration = info['duration']

    if duration < 120:
        return {
            "face_detection_skip": 10,
            "full_analysis_skip": 30,
            "max_frame_width": 960,
            "jpeg_quality": 60,
            "priority": "quality"
        }
    elif duration < 600:
        return {
            "face_detection_skip": 15,
            "full_analysis_skip": 45,
            "max_frame_width": 720,
            "jpeg_quality": 50,
            "priority": "balanced"
        }
    else:
        return {
            "face_detection_skip": 20,
            "full_analysis_skip": 60,
            "max_frame_width": 640,
            "jpeg_quality": 45,
            "priority": "speed"
        }


def estimate_processing_time(video_path: str, skip_rate: int = 15) -> dict:
    """
    Estima tiempo de procesamiento basado en características del video.

    Args:
        video_path: Ruta al archivo de video.
        skip_rate: Skip rate a utilizar en procesamiento.

    Returns:
        Diccionario con estimaciones:
            total_frames (int): Frames totales del video.
            frames_to_analyze (int): Frames que se procesarán.
            estimated_seconds (int): Tiempo estimado en segundos.
            estimated_minutes (float): Tiempo estimado en minutos.
            skip_rate (int): Skip rate utilizado para estimación.

    Notes:
        La estimación asume ~0.5 segundos por frame analizado, que es
        aproximación para hardware típico (CPU Intel i5/i7, sin GPU).

        El tiempo real varía según:
            - Hardware (GPU acelera 5-10x)
            - Configuración de análisis (emociones, composición)
            - Número de rostros por frame
            - Resolución del video

        Considerar esta estimación como límite superior conservador.
        Hardware moderno con GPU puede ser 5-10x más rápido.
    """
    info = get_video_info(video_path)
    if not info:
        return {}

    total_frames = info['total_frames']
    frames_to_analyze = total_frames // skip_rate

    estimated_seconds = frames_to_analyze * 0.5

    return {
        "total_frames": total_frames,
        "frames_to_analyze": frames_to_analyze,
        "estimated_seconds": round(estimated_seconds),
        "estimated_minutes": round(estimated_seconds / 60, 1),
        "skip_rate": skip_rate
    }