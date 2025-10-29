"""
video_utils.py

Utilidades para procesamiento, análisis y manipulación de archivos de vídeo.
Proporciona funciones para extracción de metadatos, manejo de frames y análisis
de características técnicas de vídeo.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Procesamiento de vídeo y frames
    - numpy: Operaciones numéricas

Usage:
    from app.utils.video_utils import get_video_info, extract_frame_at_time

    info = get_video_info("video.mp4")
    frame = extract_frame_at_time("video.mp4", 10.5)
"""

import cv2
import os
from typing import Tuple, Optional, List, Dict, Any
import numpy as np


def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Extrae metadatos técnicos completos de archivo de vídeo.

    Args:
        video_path: Ruta absoluta o relativa al archivo de vídeo

    Returns:
        Diccionario con información del vídeo:
            - width: Ancho del frame en píxeles
            - height: Alto del frame en píxeles
            - fps: Frame rate (frames por segundo)
            - total_frames: Número total de frames en el vídeo
            - duration: Duración total en segundos
            - codec: Código FourCC del codec utilizado

        None si el vídeo no puede abrirse o no existe

    Notes:
        Utiliza propiedades CAP_PROP_* de OpenCV para extraer metadatos.
        La duración se calcula como total_frames / fps. El codec se devuelve
        como entero FourCC convertible a string.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": 0.0,
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC))
    }

    if info["fps"] > 0:
        info["duration"] = info["total_frames"] / info["fps"]

    cap.release()
    return info


def calculate_video_bitrate(video_path: str) -> Optional[float]:
    """
    Calcula bitrate promedio aproximado del archivo de vídeo.

    Args:
        video_path: Ruta al archivo de vídeo

    Returns:
        Bitrate en kilobits por segundo (kbps), o None si hay error

    Notes:
        El cálculo es aproximado mediante la fórmula (file_size_bytes * 8) / duration_seconds.
        No distingue entre bitrate de vídeo y audio. Valores de referencia típicos:
        480p (SD): 500-2000 kbps, 720p (HD): 1500-4000 kbps, 1080p (Full HD): 4000-8000 kbps,
        4K (Ultra HD): 15000-40000 kbps.
    """
    if not os.path.exists(video_path):
        return None

    file_size = os.path.getsize(video_path)

    info = get_video_info(video_path)
    if not info or info["duration"] == 0:
        return None

    bitrate_bps = (file_size * 8) / info["duration"]
    bitrate_kbps = bitrate_bps / 1000

    return round(bitrate_kbps, 2)


def extract_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """
    Extrae frame específico en tiempo dado del vídeo.

    Args:
        video_path: Ruta al archivo de vídeo
        time_seconds: Tiempo en segundos donde extraer frame (puede ser decimal)

    Returns:
        Frame como numpy array BGR, o None si hay error

    Notes:
        Usa CAP_PROP_POS_MSEC para posicionamiento temporal. La precisión depende
        del keyframe interval del codec. Para H.264/H.265 puede no ser exacto
        debido a B-frames. Se recomienda extract_frame_at_number() para precisión exacta.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    time_ms = time_seconds * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def extract_frame_at_number(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extrae frame específico por número de índice.

    Args:
        video_path: Ruta al archivo de vídeo
        frame_number: Índice del frame (0-based)

    Returns:
        Frame como numpy array BGR, o None si hay error

    Notes:
        Más preciso que extract_frame_at_time para frames específicos.
        frame_number debe estar en rango [0, total_frames-1].
        Usa CAP_PROP_POS_FRAMES para posicionamiento exacto.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def extract_frames_batch(video_path: str, frame_numbers: List[int]) -> List[Optional[np.ndarray]]:
    """
    Extrae múltiples frames específicos eficientemente.

    Args:
        video_path: Ruta al archivo de vídeo
        frame_numbers: Lista de índices de frames a extraer

    Returns:
        Lista de frames en mismo orden que frame_numbers.
        Posiciones con errores contienen None

    Notes:
        Los frame_numbers se ordenan antes de extracción para minimizar seeks
        del video file. Más eficiente que múltiples llamadas a extract_frame_at_number()
        cuando se necesitan varios frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        return [None] * len(frame_numbers)

    for frame_num in sorted(frame_numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        frames.append(frame if ret else None)

    cap.release()
    return frames


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Crea VideoWriter configurado para guardar vídeo.

    Args:
        output_path: Ruta donde guardar archivo de salida
        fps: Frame rate del vídeo de salida
        frame_size: Dimensiones (width, height) de frames
        codec: Código de 4 caracteres del codec. Por defecto 'mp4v' (MPEG-4)

    Returns:
        VideoWriter configurado y listo para escritura de frames

    Notes:
        Codecs comunes: 'mp4v' (MPEG-4, buena compatibilidad), 'avc1' (H.264,
        excelente compresión), 'XVID' (Xvid, open source), 'MJPG' (Motion JPEG,
        archivos grandes), 'VP80' (VP8/WebM, open source). El codec debe estar
        instalado en el sistema.
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    return writer


def split_video_into_scenes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_length: int = 30
) -> List[int]:
    """
    Detecta cambios de escena mediante análisis de diferencia entre frames.

    Args:
        video_path: Ruta al archivo de vídeo
        threshold: Umbral de diferencia media para detectar corte (0-255).
            Valores típicos: 20-40. Mayor valor indica menor sensibilidad
        min_scene_length: Mínimo de frames entre cortes para evitar falsos positivos

    Returns:
        Lista de números de frame donde hay cambios de escena.
        Siempre incluye frame 0 como inicio de primera escena

    Notes:
        Algoritmo: calcula diferencia absoluta entre frames consecutivos en escala
        de grises. Si mean(diff) > threshold y han pasado min_scene_length frames,
        se registra nuevo corte. No detecta fundidos (fades) o transiciones graduales.
        Para vídeos con movimientos rápidos, incrementar threshold.
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
    frame_num = 1
    last_scene_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = diff.mean()

        if mean_diff > threshold and (frame_num - last_scene_frame) >= min_scene_length:
            scene_changes.append(frame_num)
            last_scene_frame = frame_num

        prev_gray = gray
        frame_num += 1

    cap.release()
    return scene_changes


def estimate_video_quality(video_path: str) -> Dict[str, Any]:
    """
    Estima calidad técnica del vídeo basándose en metadatos.

    Args:
        video_path: Ruta al archivo de vídeo

    Returns:
        Diccionario con estimaciones de calidad:
            - resolution_quality: Clasificación de resolución
            - fps_quality: Clasificación de frame rate
            - bitrate: Bitrate en kbps
            - bitrate_quality: Clasificación de bitrate
            - overall_score: Puntuación general 0-10

    Notes:
        Clasificaciones de resolución: 4K/Ultra HD (≥3840x2160), Full HD/1080p
        (≥1920x1080), HD/720p (≥1280x720), SD/480p (≥854x480), Low (<854x480).
        Clasificaciones de FPS: Cinematic (23-25), Standard (29-31), Smooth (50-60),
        High Speed (>60). El overall_score combina resolución (40%), FPS (30%)
        y bitrate (30%).
    """
    info = get_video_info(video_path)
    if not info:
        return {}

    width, height = info["width"], info["height"]
    pixels = width * height

    if pixels >= 3840 * 2160:
        resolution_quality = "4K/Ultra HD"
        resolution_score = 10
    elif pixels >= 1920 * 1080:
        resolution_quality = "Full HD (1080p)"
        resolution_score = 8
    elif pixels >= 1280 * 720:
        resolution_quality = "HD (720p)"
        resolution_score = 6
    elif pixels >= 854 * 480:
        resolution_quality = "SD (480p)"
        resolution_score = 4
    else:
        resolution_quality = "Low Quality"
        resolution_score = 2

    fps = info["fps"]
    if 23 <= fps <= 25:
        fps_quality = "Cinematic (24fps)"
        fps_score = 7
    elif 29 <= fps <= 31:
        fps_quality = "Standard (30fps)"
        fps_score = 8
    elif 50 <= fps <= 60:
        fps_quality = "Smooth (50-60fps)"
        fps_score = 9
    elif fps > 60:
        fps_quality = "High Speed (>60fps)"
        fps_score = 10
    else:
        fps_quality = "Low FPS"
        fps_score = 5

    bitrate = calculate_video_bitrate(video_path)
    if bitrate:
        if bitrate >= 8000:
            bitrate_quality = "Excellent"
            bitrate_score = 10
        elif bitrate >= 4000:
            bitrate_quality = "Good"
            bitrate_score = 8
        elif bitrate >= 2000:
            bitrate_quality = "Fair"
            bitrate_score = 6
        else:
            bitrate_quality = "Poor"
            bitrate_score = 4
    else:
        bitrate_quality = "Unknown"
        bitrate_score = 5

    overall_score = (
        resolution_score * 0.4 +
        fps_score * 0.3 +
        bitrate_score * 0.3
    )

    return {
        "resolution_quality": resolution_quality,
        "fps_quality": fps_quality,
        "bitrate": bitrate,
        "bitrate_quality": bitrate_quality,
        "overall_score": round(overall_score, 1)
    }