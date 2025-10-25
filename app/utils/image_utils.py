"""
image_utils.py

Utilidades optimizadas para procesamiento de imágenes y frames de video.
Incluye conversión base64, redimensionamiento eficiente, recorte de regiones,
ajustes de imagen y cálculo de diferencias para tracking.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
import base64
from typing import Tuple


def frame_to_base64(frame: np.ndarray, quality: int = 50,
                    max_width: int = 720) -> str:
    """
    Convierte frame a base64 con compresión JPEG para transmisión WebSocket.

    Optimizado para balance entre calidad visual y tamaño de payload. Reduce
    resolución si excede ancho máximo y aplica compresión JPEG configurable.

    Args:
        frame: Frame en formato BGR (OpenCV) a convertir.
        quality: Calidad de compresión JPEG [0-100]. Por defecto 50 para
            balance entre tamaño y calidad visual.
        max_width: Ancho máximo en píxeles. Frames más anchos se redimensionan
            proporcionalmente. Por defecto 720px para reducir latencia.

    Returns:
        String base64 con prefijo data:image/jpeg para uso directo en HTML
        <img src="...">.

    Notes:
        Optimizaciones implementadas:
            - Calidad JPEG reducida a 50 (vs 60 default) para ~30% menos bytes
            - Max width 720px (vs 1280px) para ~50% menos píxeles
            - INTER_AREA para mejor calidad al reducir tamaño vs INTER_LINEAR

        La interpolación INTER_AREA realiza promediado de píxeles al reducir,
        produciendo imágenes más suaves y sin aliasing comparado con métodos
        de muestreo simple como INTER_LINEAR o INTER_NEAREST.
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


def frame_to_base64_fast(frame: np.ndarray, quality: int = 45,
                         max_width: int = 640) -> str:
    """
    Versión ultra-rápida con compresión máxima para alta frecuencia de frames.

    Sacrifica calidad visual por velocidad de procesamiento y mínimo tamaño
    de payload. Útil para preview en tiempo real o baja prioridad visual.

    Args:
        frame: Frame BGR a convertir.
        quality: Calidad JPEG. Por defecto 45 para máxima compresión.
        max_width: Ancho máximo. Por defecto 640px para mayor reducción.

    Returns:
        String base64 con prefijo data:image/jpeg.

    Notes:
        Optimizaciones agresivas:
            - INTER_NEAREST: interpolación más rápida sin anti-aliasing
            - JPEG_OPTIMIZE: activa optimización Huffman de encoder JPEG
            - Calidad 45: reduce ~40% tamaño vs calidad 60
            - Max width 640px: ~60% menos píxeles vs 1280px

        INTER_NEAREST produce edges visiblemente escalonados pero es 3-4x
        más rápido que INTER_AREA. Usar solo cuando velocidad es crítica.
    """
    h, w = frame.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY), quality,
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1
    ]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)

    base64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_str}"


def resize_for_detection(frame: np.ndarray, target_width: int = 480) -> Tuple[np.ndarray, float]:
    """
    Redimensiona frame para detección facial optimizada.

    Reduce resolución para acelerar algoritmos de detección DNN manteniendo
    suficiente detalle para localización precisa de rostros.

    Args:
        frame: Frame original en cualquier resolución.
        target_width: Ancho objetivo para detección. Por defecto 480px como
            balance entre velocidad y precisión de detección.

    Returns:
        Tupla (frame_redimensionado, factor_escala) donde factor_escala
        permite reconstruir coordenadas en resolución original.

    Notes:
        Un width de 480px acelera detección facial ~4x comparado con 1080p
        manteniendo 90%+ de precisión de detección. Las coordenadas detectadas
        deben multiplicarse por factor_escala para mapearlas al frame original.
    """
    h, w = frame.shape[:2]

    if w <= target_width:
        return frame, 1.0

    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def base64_to_frame(base64_str: str) -> np.ndarray:
    """
    Convierte string base64 a frame numpy array.

    Args:
        base64_str: String en formato base64, opcionalmente con prefijo
            data:image/jpeg;base64,.

    Returns:
        Frame BGR como numpy array listo para procesamiento OpenCV.

    Notes:
        Maneja automáticamente prefijo data URI si está presente. El decoder
        de OpenCV soporta JPEG, PNG, BMP y otros formatos de imagen comunes.
    """
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return frame


def resize_frame(frame: np.ndarray, width: int = None,
                 height: int = None, keep_aspect: bool = True,
                 fast: bool = False) -> np.ndarray:
    """
    Redimensiona frame con control fino de dimensiones e interpolación.

    Args:
        frame: Frame a redimensionar.
        width: Ancho deseado en píxeles. None mantiene ancho proporcional.
        height: Alto deseado en píxeles. None mantiene alto proporcional.
        keep_aspect: Si True, mantiene relación de aspecto original.
        fast: Si True, usa INTER_NEAREST (rápido) vs INTER_AREA (calidad).

    Returns:
        Frame redimensionado según parámetros especificados.

    Notes:
        Con keep_aspect=True y solo width especificado, calcula height
        automáticamente manteniendo proporción. Útil para thumbnails y
        previews donde preservar aspect ratio es importante.
    """
    h, w = frame.shape[:2]

    if width is None and height is None:
        return frame

    interpolation = cv2.INTER_NEAREST if fast else cv2.INTER_AREA

    if keep_aspect:
        if width is not None:
            aspect = width / w
            new_h = int(h * aspect)
            return cv2.resize(frame, (width, new_h), interpolation=interpolation)
        elif height is not None:
            aspect = height / h
            new_w = int(w * aspect)
            return cv2.resize(frame, (new_w, height), interpolation=interpolation)
    else:
        if width is None:
            width = w
        if height is None:
            height = h
        return cv2.resize(frame, (width, height), interpolation=interpolation)

    return frame


def crop_frame(frame: np.ndarray, x: int, y: int,
               width: int, height: int) -> np.ndarray:
    """
    Recorta región rectangular del frame con clipping automático.

    Args:
        frame: Frame a recortar.
        x, y: Coordenadas de esquina superior izquierda.
        width, height: Dimensiones del recorte.

    Returns:
        Región recortada del frame. Si coordenadas exceden límites,
        se ajustan automáticamente a bordes válidos.

    Notes:
        Las coordenadas se clipean a los límites del frame para prevenir
        errores de índice fuera de rango. Útil cuando coordenadas vienen
        de detecciones que pueden estar parcialmente fuera del frame.
    """
    h, w = frame.shape[:2]

    x = max(0, min(x, w))
    y = max(0, min(y, h))
    x2 = max(0, min(x + width, w))
    y2 = max(0, min(y + height, h))

    return frame[y:y2, x:x2]


def crop_face_region(frame: np.ndarray, face_box: Tuple[int, int, int, int],
                     padding: float = 0.2) -> np.ndarray:
    """
    Recorta región facial con padding para análisis emocional.

    Expande bounding box de rostro para incluir contexto adicional útil
    para detección de emociones (cabello, cuello, orejas).

    Args:
        frame: Frame completo BGR.
        face_box: Coordenadas (x1, y1, x2, y2) del rostro.
        padding: Porcentaje de expansión. 0.2 = 20% del tamaño de rostro
            añadido en cada dirección.

    Returns:
        Región recortada centrada en rostro con padding aplicado.

    Notes:
        El padding de 20% por defecto es óptimo para modelos de emoción
        entrenados con FER2013 y similares datasets que incluyen contexto
        facial extendido. Permite capturar expresiones en mejillas y frente.
    """
    x1, y1, x2, y2 = face_box
    h, w = frame.shape[:2]

    face_w = x2 - x1
    face_h = y2 - y1
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return frame[y1:y2, x1:x2]


def blend_frames(frame1: np.ndarray, frame2: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """
    Mezcla dos frames con transparencia para efectos de transición.

    Args:
        frame1: Primer frame BGR.
        frame2: Segundo frame BGR con mismas dimensiones que frame1.
        alpha: Factor de mezcla [0.0, 1.0]. 0.5 = mezcla equitativa.

    Returns:
        Frame resultante de blend: alpha*frame1 + (1-alpha)*frame2.

    Notes:
        Útil para crossfade, overlays semi-transparentes y análisis de
        diferencias visuales entre frames. Ambos frames deben tener
        dimensiones idénticas.
    """
    return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)


def apply_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Aplica desenfoque gaussiano para suavizado de imagen.

    Args:
        frame: Frame a desenfocar.
        kernel_size: Tamaño del kernel gaussiano (debe ser impar). Mayor
            tamaño produce mayor desenfoque.

    Returns:
        Frame con desenfoque aplicado.

    Notes:
        Desenfoque gaussiano reduce ruido y aliasing. Útil como preprocesamiento
        antes de detección de bordes o para efectos visuales.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def apply_sharpen(frame: np.ndarray) -> np.ndarray:
    """
    Aplica filtro de nitidez mediante kernel Laplaciano.

    Args:
        frame: Frame a afilar.

    Returns:
        Frame con bordes acentuados y detalles más definidos.

    Notes:
        El kernel enfatiza diferencias con vecinos, resaltando bordes y
        texturas. Puede amplificar ruido en imágenes de baja calidad.
    """
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(frame, -1, kernel)


def adjust_brightness(frame: np.ndarray, value: int = 30) -> np.ndarray:
    """
    Ajusta brillo del frame en espacio de color HSV.

    Args:
        frame: Frame BGR a ajustar.
        value: Cantidad a añadir al canal V (valor). Positivo aumenta brillo,
            negativo lo reduce. Rango típico [-100, 100].

    Returns:
        Frame con brillo ajustado.

    Notes:
        Opera en espacio HSV para preservar matiz y saturación mientras
        modifica luminosidad. El canal V se clipea a [0, 255] para prevenir
        overflow/underflow.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(frame: np.ndarray, alpha: float = 1.5) -> np.ndarray:
    """
    Ajusta contraste mediante escalado lineal de intensidades.

    Args:
        frame: Frame a ajustar.
        alpha: Factor multiplicativo de contraste. 1.0 = sin cambio,
            >1.0 aumenta contraste, <1.0 lo reduce.

    Returns:
        Frame con contraste ajustado mediante transformación lineal
        output = alpha * input + beta.

    Notes:
        Valores alpha típicos: 1.5-2.0 para aumentar, 0.5-0.8 para reducir.
        convertScaleAbs previene overflow saturando a [0, 255].
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calcula diferencia promedio entre dos frames para detección de cambios.

    Args:
        frame1: Primer frame BGR.
        frame2: Segundo frame BGR con mismas dimensiones.

    Returns:
        Diferencia promedio de intensidad [0.0, 255.0]. Valores altos
        indican cambios significativos entre frames.

    Notes:
        Útil para:
            - Detección de cambios de escena
            - Optimización de skip frames en análisis
            - Tracking de movimiento de cámara

        Diferencias típicas: <10 = movimiento mínimo, 10-30 = movimiento
        moderado, >30 = cambio significativo o corte.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    return float(diff.mean())


def downsample_for_analysis(frame: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Reduce resolución por factor entero para análisis que no requieren detalle.

    Args:
        frame: Frame original.
        factor: Factor de reducción. 2 = mitad del tamaño en cada dimensión
            (cuarto de píxeles totales).

    Returns:
        Frame reducido con dimensiones width//factor x height//factor.

    Notes:
        Factor 2 reduce píxeles en 75%, acelerando análisis ~4x. Útil para:
            - Análisis de color global
            - Detección de iluminación
            - Histogramas RGB

        No recomendado para detección facial o análisis que requieren
        resolución espacial precisa.
    """
    h, w = frame.shape[:2]
    new_h = h // factor
    new_w = w // factor

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_frame_hash(frame: np.ndarray) -> str:
    """
    Genera hash MD5 del frame para sistema de caché y deduplicación.

    Args:
        frame: Frame a hashear.

    Returns:
        String hexadecimal de 32 caracteres (hash MD5).

    Notes:
        Utiliza thumbnail 32x32 para velocidad, sacrificando precisión.
        Frames visualmente idénticos pero con ruido mínimo pueden producir
        hashes diferentes. Útil para cache lookup rápido y detección de
        frames duplicados exactos.

        MD5 es suficientemente único para espacio de frames típico. Para
        seguridad criptográfica usar SHA-256 (más lento).
    """
    import hashlib

    small = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)
    frame_bytes = small.tobytes()

    return hashlib.md5(frame_bytes).hexdigest()