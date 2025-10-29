"""
image_utils.py

Utilidades para procesamiento de imágenes y frames de vídeo con funciones
optimizadas para conversión de formato, validación de coordenadas y extracción
segura de regiones de interés.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Procesamiento de imágenes
    - numpy: Operaciones con arrays

Usage:
    from app.utils.image_utils import frame_to_base64, to_grayscale_safe

    base64_str = frame_to_base64(frame, quality=50, max_width=720)
    gray = to_grayscale_safe(frame)
    valid_box = validate_bbox((x1, y1, x2, y2), width, height)
    region = crop_region_safe(frame, box, padding=0.2)
"""

import cv2
import numpy as np
import base64
from typing import Tuple, Optional


def frame_to_base64(frame: np.ndarray, quality: int = 50,
                    max_width: int = 720) -> str:
    """
    Convierte frame a base64 con compresión JPEG para transmisión.

    Aplica redimensionamiento proporcional si el ancho excede el límite
    especificado y comprime la imagen usando el codec JPEG con calidad
    configurable.

    Args:
        frame: Frame en formato BGR (OpenCV) a convertir
        quality: Calidad de compresión JPEG [0-100]. Por defecto 50
        max_width: Ancho máximo en píxeles. Por defecto 720

    Returns:
        String base64 con prefijo data:image/jpeg para uso en HTML

    Notes:
        La interpolación INTER_AREA realiza promediado de píxeles al reducir,
        produciendo imágenes más suaves sin aliasing. El prefijo data:image/jpeg
        permite uso directo en atributos src de elementos img en HTML.
    """
    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)

    base64_str = base64.b64encode(buffer).decode('utf-8')

    return f"data:image/jpeg;base64,{base64_str}"


def to_grayscale_safe(frame: np.ndarray) -> np.ndarray:
    """
    Convierte frame a escala de grises manejando múltiples formatos de entrada.

    Detecta automáticamente el formato del frame de entrada y aplica la
    conversión apropiada. Soporta frames en grayscale 2D, grayscale 3D con
    canal único, BGR estándar y BGRA con canal alpha.

    Args:
        frame: Frame de entrada en cualquiera de los siguientes formatos:
            - Grayscale 2D: (H, W)
            - Grayscale 3D: (H, W, 1)
            - BGR: (H, W, 3)
            - BGRA: (H, W, 4)

    Returns:
        Frame en escala de grises 2D con dimensiones (H, W) y dtype uint8

    Raises:
        ValueError: Si el frame tiene forma inesperada o está vacío

    Notes:
        La verificación de dimensiones previene errores al procesar frames de
        diferentes fuentes. Los frames BGRA se convierten primero a BGR antes
        de la conversión a grayscale para mantener consistencia.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame debe ser numpy.ndarray, recibido: {type(frame)}")

    if frame.size == 0:
        raise ValueError("frame está vacío (size=0)")

    if len(frame.shape) == 2:
        return frame

    elif len(frame.shape) == 3 and frame.shape[2] == 1:
        return frame.squeeze()

    elif len(frame.shape) == 3 and frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    else:
        raise ValueError(
            f"Frame con forma inesperada: {frame.shape}. "
            f"Esperado: (H,W), (H,W,1), (H,W,3) o (H,W,4)"
        )


def validate_bbox(box: Tuple[int, int, int, int],
                  frame_width: int,
                  frame_height: int,
                  min_size: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """
    Valida y ajusta bounding box para que esté dentro de los límites del frame.

    Realiza clipping automático de las coordenadas a los límites del frame y
    verifica que el box resultante tenga dimensiones mínimas válidas.

    Args:
        box: Tupla con coordenadas (x1, y1, x2, y2) donde (x1, y1) representa
            la esquina superior izquierda y (x2, y2) la esquina inferior derecha
        frame_width: Ancho del frame en píxeles
        frame_height: Alto del frame en píxeles
        min_size: Tamaño mínimo en píxeles para ancho y alto. Por defecto 10

    Returns:
        Tupla (x1, y1, x2, y2) con coordenadas ajustadas, o None si el box
        resultante es demasiado pequeño o completamente fuera del frame

    Notes:
        El parámetro min_size previene el procesamiento de regiones que no
        contienen información útil, causan errores en redimensionamiento o
        producen detecciones espurias.
    """
    x1, y1, x2, y2 = box

    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    width = x2 - x1
    height = y2 - y1

    if width < min_size or height < min_size:
        return None

    return (x1, y1, x2, y2)


def crop_region_safe(frame: np.ndarray,
                     box: Tuple[int, int, int, int],
                     padding: float = 0.0) -> Optional[np.ndarray]:
    """
    Recorta región del frame con validación automática y padding opcional.

    Combina validación de coordenadas con extracción segura de subregión.
    Permite expansión proporcional del bounding box mediante el parámetro
    padding para incluir contexto adicional alrededor de la región de interés.

    Args:
        frame: Frame BGR o grayscale a recortar
        box: Coordenadas (x1, y1, x2, y2) de la región a extraer
        padding: Expansión proporcional del box en todas las direcciones [0.0-1.0]
            donde 0.0 indica sin padding, 0.2 expande 20% del tamaño y 0.5
            expande 50% del tamaño en cada dirección. Por defecto 0.0

    Returns:
        Array numpy con la región recortada, o None si las coordenadas son
        inválidas, la región resultante es demasiado pequeña o el frame está vacío

    Notes:
        El padding se calcula como porcentaje del tamaño del box mediante las
        fórmulas padding_x = width * padding y padding_y = height * padding.
        Después de aplicar el padding, las coordenadas se validan nuevamente
        para asegurar que no excedan los límites del frame.
    """
    if frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    if padding > 0:
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * padding)
        pad_y = int(box_height * padding)

        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

    validated = validate_bbox((x1, y1, x2, y2), w, h)

    if validated is None:
        return None

    x1, y1, x2, y2 = validated

    return frame[y1:y2, x1:x2]