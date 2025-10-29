"""
visualization_utils.py

Utilidades centralizadas para visualización de análisis cinematográfico.
Proporciona funciones para superponer información visual sobre frames procesados.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Renderizado de gráficos y texto
    - numpy: Operaciones con arrays

Usage:
    from app.utils.visualization_utils import apply_all_overlays

    annotated_frame = apply_all_overlays(
        frame=frame,
        results=analysis_results,
        show_faces=True,
        show_rule_of_thirds=True
    )
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)

try:
    from app.config import (
        EMOTION_COLORS,
        EMOTIONS,
        COMPOSITION_CONFIG,
        SHOT_TYPES
    )
except ImportError:
    EMOTION_COLORS = {
        "angry": (0, 0, 255),
        "happy": (0, 255, 255),
        "sad": (255, 0, 0),
        "neutral": (128, 128, 128),
        "surprise": (0, 165, 255),
        "fear": (128, 0, 128),
        "disgust": (0, 128, 0)
    }
    EMOTIONS = {
        "angry": "Enfadado/a",
        "happy": "Feliz",
        "sad": "Triste",
        "neutral": "Neutral",
        "surprise": "Sorpresa",
        "fear": "Miedo",
        "disgust": "Disgusto"
    }
    COMPOSITION_CONFIG = {
        "rule_of_thirds": {
            "tolerance": 0.1,
            "grid_color": (0, 255, 255),
            "line_thickness": 2,
        }
    }
    SHOT_TYPES = {
        "ECU": "Plano Detalle",
        "CU": "Primer Plano",
        "MCU": "Plano Medio Corto",
        "MS": "Plano Medio",
        "MWS": "Plano Americano",
        "WS": "Plano General",
        "EWS": "Gran Plano General"
    }


EMOTION_COLOR_MAP = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "fear": (128, 0, 128),
    "surprise": (0, 165, 255),
    "disgust": (0, 128, 0),
    "neutral": (128, 128, 128)
}


def apply_all_overlays(
        frame: np.ndarray,
        results: Dict[str, Any],
        show_faces: bool = True,
        show_rule_of_thirds: bool = False,
        show_lighting: bool = False,
        show_camera_movement: bool = False,
        show_shot_type: bool = False,
        show_color_palette: bool = False,
        opacity: float = 1.0
) -> np.ndarray:
    """
    Aplica todos los overlays de análisis sobre el frame.

    Superpone información visual del análisis cinematográfico incluyendo
    detecciones faciales, regla de tercios, indicadores de iluminación,
    movimiento de cámara, tipo de plano y paleta de colores.

    Args:
        frame: Frame BGR sobre el que aplicar overlays
        results: Diccionario con resultados de análisis del frame
        show_faces: Mostrar detecciones y reconocimiento facial
        show_rule_of_thirds: Mostrar grid de regla de tercios
        show_lighting: Mostrar indicador de tipo de iluminación
        show_camera_movement: Mostrar indicador de movimiento de cámara
        show_shot_type: Mostrar badge de tipo de plano
        show_color_palette: Mostrar paleta de colores dominantes
        opacity: Opacidad de los overlays [0.0-1.0]

    Returns:
        Frame con overlays aplicados

    Notes:
        Los overlays se aplican en orden específico para correcta superposición:
        regla de tercios como fondo, luego indicadores de análisis y finalmente
        rostros con emociones. La opacidad controla la transparencia de todos
        los elementos visuales.
    """
    frame_annotated = frame.copy()

    if show_rule_of_thirds:
        frame_annotated = draw_rule_of_thirds(frame_annotated, opacity=opacity * 0.6)

    if show_lighting and results.get("lighting"):
        frame_annotated = draw_lighting_indicator(
            frame_annotated,
            results["lighting"],
            opacity=opacity
        )

    if show_camera_movement and results.get("camera_movement"):
        frame_annotated = draw_camera_movement_indicator(
            frame_annotated,
            results["camera_movement"],
            opacity=opacity
        )

    if show_shot_type and results.get("shot_type"):
        frame_annotated = draw_shot_type_badge(
            frame_annotated,
            results["shot_type"],
            opacity=opacity
        )

    if show_color_palette and results.get("colors"):
        frame_annotated = draw_color_palette(
            frame_annotated,
            results["colors"],
            opacity=opacity
        )

    if show_faces and results.get("faces"):
        frame_annotated = draw_faces_with_emotions(
            frame_annotated,
            results["faces"],
            opacity=opacity
        )

    return frame_annotated


def draw_faces_with_emotions(
        frame: np.ndarray,
        faces: List[Dict[str, Any]],
        opacity: float = 1.0
) -> np.ndarray:
    """
    Dibuja rostros detectados con sistema de colores por emoción.

    Renderiza bounding boxes amarillos para rostros detectados con etiquetas
    de nombre de actor y emoción. El color de fondo de la etiqueta de emoción
    corresponde al tipo de emoción detectada.

    Args:
        frame: Frame BGR sobre el que dibujar
        faces: Lista de rostros detectados con información de emoción
        opacity: Opacidad de las anotaciones [0.0-1.0]

    Returns:
        Frame con rostros anotados

    Notes:
        Sistema de colores: rectángulo amarillo para cara, fondo blanco para
        nombre de actor, fondo con color de emoción para etiqueta de emoción.
        Rostros no reconocidos se etiquetan como "Desconocido".
    """
    for face in faces:
        try:
            box = face.get("box")
            if not box or not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            h, w = frame.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
                continue

            face_box_color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), face_box_color, 2)

            emotion_str = "neutral"
            emotion_confidence = 0.0
            emotion_color = EMOTION_COLOR_MAP["neutral"]

            if "emotion" in face:
                emotion = face["emotion"]
                if isinstance(emotion, dict):
                    emotion_str = emotion.get("emotion", "neutral").lower().strip()
                    emotion_confidence = emotion.get("confidence", 0.0)
                else:
                    emotion_str = str(emotion).lower().strip()

                emotion_color = EMOTION_COLOR_MAP.get(emotion_str, EMOTION_COLOR_MAP["neutral"])

            emotion_label = EMOTIONS.get(emotion_str, emotion_str.capitalize())

            is_recognized = face.get("recognized", False)
            name = face.get("name", face.get("actor", ""))

            if not name or not is_recognized:
                name = "Desconocido"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            current_y = y1 - 10

            (text_width, text_height), baseline = cv2.getTextSize(
                name, font, font_scale, thickness
            )

            name_y = max(current_y, text_height + 10)

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x1, name_y - text_height - 5),
                (x1 + text_width + 10, name_y + 5),
                (255, 255, 255),
                -1
            )
            cv2.addWeighted(overlay, opacity * 0.7, frame, 1 - opacity * 0.7, 0, frame)

            cv2.putText(
                frame,
                name,
                (x1 + 5, name_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

            current_y = name_y + 5

            if emotion_confidence > 0:
                emotion_text = f"{emotion_label} ({emotion_confidence:.0%})"
            else:
                emotion_text = emotion_label

            (text_width, text_height), baseline = cv2.getTextSize(
                emotion_text, font, font_scale, thickness
            )

            emotion_y = current_y + text_height + 5

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x1, emotion_y - text_height - 5),
                (x1 + text_width + 10, emotion_y + 5),
                emotion_color,
                -1
            )
            cv2.addWeighted(overlay, opacity * 0.7, frame, 1 - opacity * 0.7, 0, frame)

            cv2.putText(
                frame,
                emotion_text,
                (x1 + 5, emotion_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        except Exception as e:
            logger.warning(f"Error dibujando rostro: {e}")
            continue

    return frame


def draw_rule_of_thirds(
        frame: np.ndarray,
        opacity: float = 0.6
) -> np.ndarray:
    """
    Dibuja grid de regla de tercios sobre el frame.

    Superpone líneas verticales y horizontales que dividen el frame en tercios
    con puntos en las intersecciones para guía de composición visual.

    Args:
        frame: Frame BGR sobre el que dibujar
        opacity: Opacidad del grid [0.0-1.0]

    Returns:
        Frame con grid de regla de tercios

    Notes:
        El grid incluye dos líneas verticales y dos horizontales que dividen
        el frame en nueve secciones iguales. Los puntos de intersección
        marcan las posiciones de interés compositivo según la regla de tercios.
    """
    h, w = frame.shape[:2]

    config = COMPOSITION_CONFIG.get("rule_of_thirds", {})
    color = config.get("grid_color", (0, 255, 255))
    thickness = config.get("line_thickness", 2)

    overlay = frame.copy()

    x1 = w // 3
    x2 = 2 * w // 3
    cv2.line(overlay, (x1, 0), (x1, h), color, thickness)
    cv2.line(overlay, (x2, 0), (x2, h), color, thickness)

    y1 = h // 3
    y2 = 2 * h // 3
    cv2.line(overlay, (0, y1), (w, y1), color, thickness)
    cv2.line(overlay, (0, y2), (w, y2), color, thickness)

    point_radius = 5
    points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for point in points:
        cv2.circle(overlay, point, point_radius, color, -1)

    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    return frame


def draw_lighting_indicator(
        frame: np.ndarray,
        lighting: Dict[str, Any],
        opacity: float = 1.0,
        position: str = "top-left"
) -> np.ndarray:
    """
    Dibuja indicador de tipo de iluminación.

    Renderiza badge con icono y etiqueta del tipo de iluminación detectada
    en posición configurable del frame.

    Args:
        frame: Frame BGR sobre el que dibujar
        lighting: Diccionario con información de iluminación
        opacity: Opacidad del indicador [0.0-1.0]
        position: Posición del badge ("top-left", "top-right", "bottom-left", "bottom-right")

    Returns:
        Frame con indicador de iluminación

    Notes:
        Tipos de iluminación: High Key (clave alta), Low Key (clave baja),
        Normal. Cada tipo tiene icono y color de fondo distintivo.
    """
    lighting_type = lighting.get("type", "Normal")

    lighting_names = {
        "High Key": "Clave Alta",
        "Low Key": "Clave Baja",
        "Normal": "Normal"
    }

    lighting_display = lighting_names.get(lighting_type, lighting_type)

    icons = {
        "High Key": ("[+]", (0, 255, 255)),
        "Low Key": ("[-]", (139, 0, 139)),
        "Normal": ("[=]", (128, 128, 128))
    }

    icon, bg_color = icons.get(lighting_type, ("[?]", (128, 128, 128)))
    text = f"{icon} {lighting_display}"

    padding = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    h, w = frame.shape[:2]

    if position == "top-left":
        x, y = padding, padding + text_height
    elif position == "top-right":
        x, y = w - text_width - padding - 20, padding + text_height
    elif position == "bottom-left":
        x, y = padding, h - padding
    else:
        x, y = w - text_width - padding - 20, h - padding

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - text_height - 5),
        (x + text_width + 10, y + baseline),
        bg_color,
        -1
    )
    cv2.addWeighted(overlay, opacity * 0.7, frame, 1 - opacity * 0.7, 0, frame)

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return frame


def draw_camera_movement_indicator(
        frame: np.ndarray,
        movement: Dict[str, Any],
        opacity: float = 1.0
) -> np.ndarray:
    """
    Dibuja indicador de movimiento de cámara.

    Renderiza flecha direccional o indicador de zoom en el centro del frame
    con etiqueta descriptiva del tipo y dirección de movimiento.

    Args:
        frame: Frame BGR sobre el que dibujar
        movement: Diccionario con información de movimiento de cámara
        opacity: Opacidad del indicador [0.0-1.0]

    Returns:
        Frame con indicador de movimiento de cámara

    Notes:
        Tipos de movimiento: Pan (panorámica), Tilt (inclinación), Zoom, Static.
        Para movimientos estáticos no se renderiza indicador. La longitud de
        la flecha es proporcional a la magnitud del movimiento.
    """
    movement_type = movement.get("type", "Static")
    direction = movement.get("direction", "")
    magnitude = movement.get("magnitude", 0.0)

    if movement_type == "Static":
        return frame

    movement_names = {
        "Pan": "Panoramica",
        "Tilt": "Inclinacion",
        "Zoom": "Zoom",
        "Static": "Estatico"
    }

    direction_names = {
        "Left": "Izquierda",
        "Right": "Derecha",
        "Up": "Arriba",
        "Down": "Abajo",
        "In": "Acercamiento",
        "Out": "Alejamiento"
    }

    movement_display = movement_names.get(movement_type, movement_type)
    direction_display = direction_names.get(direction, direction)

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    arrow_length = int(50 + magnitude * 10)

    vectors = {
        "Left": (-arrow_length, 0),
        "Right": (arrow_length, 0),
        "Up": (0, -arrow_length),
        "Down": (0, arrow_length),
        "In": (0, 0),
        "Out": (0, 0)
    }

    dx, dy = vectors.get(direction, (0, 0))
    color = (0, 255, 0)

    if "Zoom" in movement_type:
        if direction == "In":
            cv2.circle(frame, (center_x, center_y), 40, color, 3)
            cv2.arrowedLine(
                frame,
                (center_x + 30, center_y),
                (center_x + 15, center_y),
                color, 2, tipLength=0.5
            )
        else:
            cv2.circle(frame, (center_x, center_y), 30, color, 3)
            cv2.arrowedLine(
                frame,
                (center_x + 15, center_y),
                (center_x + 30, center_y),
                color, 2, tipLength=0.5
            )
    else:
        end_x = center_x + dx
        end_y = center_y + dy
        cv2.arrowedLine(
            frame,
            (center_x, center_y),
            (end_x, end_y),
            color, 3, tipLength=0.3
        )

    label = f"[CAM] {movement_display}"
    if direction_display:
        label += f" {direction_display}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    text_x = center_x - text_width // 2
    text_y = center_y + 80

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (text_x - 5, text_y - text_height - 5),
        (text_x + text_width + 5, text_y + baseline),
        (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, opacity * 0.7, frame, 1 - opacity * 0.7, 0, frame)

    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return frame


def draw_shot_type_badge(
        frame: np.ndarray,
        shot_type: Any,
        opacity: float = 1.0,
        position: str = "bottom-right"
) -> np.ndarray:
    """
    Dibuja badge con tipo de plano.

    Renderiza etiqueta con clasificación del tipo de plano cinematográfico
    en posición configurable del frame.

    Args:
        frame: Frame BGR sobre el que dibujar
        shot_type: Tipo de plano (string o diccionario con campo "shot_type")
        opacity: Opacidad del badge [0.0-1.0]
        position: Posición del badge ("top-left", "top-right", "bottom-left", "bottom-right")

    Returns:
        Frame con badge de tipo de plano

    Notes:
        Tipos de plano soportados: ECU (Plano Detalle), CU (Primer Plano),
        MCU (Plano Medio Corto), MS (Plano Medio), MWS (Plano Americano),
        WS (Plano General), EWS (Gran Plano General).
    """
    if shot_type is None:
        return frame

    if isinstance(shot_type, dict):
        shot_type_str = shot_type.get("shot_type") or shot_type.get("type") or "Unknown"
    else:
        shot_type_str = str(shot_type).strip()

    if not shot_type_str or shot_type_str == "None":
        return frame

    full_name = SHOT_TYPES.get(shot_type_str, shot_type_str)
    label = f"[PLANO] {full_name}"

    padding = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    h, w = frame.shape[:2]

    if position == "bottom-right":
        x = w - text_width - padding - 20
        y = h - padding - 10
    elif position == "bottom-left":
        x = padding
        y = h - padding - 10
    elif position == "top-right":
        x = w - text_width - padding - 20
        y = padding + text_height + 10
    else:
        x = padding
        y = padding + text_height + 10

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - 5, y - text_height - 5),
        (x + text_width + 10, y + baseline),
        (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, opacity * 0.8, frame, 1 - opacity * 0.8, 0, frame)

    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return frame


def draw_color_palette(
        frame: np.ndarray,
        colors: Dict[str, Any],
        opacity: float = 1.0,
        position: str = "top-right"
) -> np.ndarray:
    """
    Dibuja paleta de colores dominantes.

    Renderiza panel con muestras de los colores dominantes del frame junto
    con sus porcentajes de presencia.

    Args:
        frame: Frame BGR sobre el que dibujar
        colors: Diccionario con análisis de colores
        opacity: Opacidad del panel [0.0-1.0]
        position: Posición del panel ("top-left", "top-right", "bottom-left", "bottom-right")

    Returns:
        Frame con paleta de colores

    Notes:
        Muestra hasta 5 colores dominantes con su porcentaje de presencia.
        Cada color se representa con un cuadrado de muestra y su porcentaje
        en texto.
    """
    if not colors or not isinstance(colors, dict):
        return frame

    dominant_colors = colors.get("dominant_colors", [])
    temperature_data = colors.get("temperature", {})

    if isinstance(temperature_data, dict):
        temperature = temperature_data.get("label", "Neutral")
    else:
        temperature = str(temperature_data) if temperature_data else "Neutral"

    if not dominant_colors:
        return frame

    temp_names = {
        "Warm": "Calido",
        "Cool": "Frio",
        "Neutral": "Neutral"
    }

    temp_display = temp_names.get(temperature, temperature)

    temp_icons = {
        "Warm": "[+]",
        "Cool": "[-]",
        "Neutral": "[=]"
    }

    h, w = frame.shape[:2]
    padding = 10
    swatch_size = 30
    swatch_spacing = 5

    palette_width = 140

    if position == "top-right":
        start_x = w - padding - palette_width
        start_y = padding
    elif position == "top-left":
        start_x = padding
        start_y = padding
    elif position == "bottom-right":
        start_x = w - padding - palette_width
        start_y = h - padding - (len(dominant_colors[:5]) * (swatch_size + swatch_spacing)) - 40
    else:
        start_x = padding
        start_y = h - padding - (len(dominant_colors[:5]) * (swatch_size + swatch_spacing)) - 40

    overlay = frame.copy()
    palette_height = len(dominant_colors[:5]) * (swatch_size + swatch_spacing) + 40

    cv2.rectangle(
        overlay,
        (start_x - 5, start_y - 5),
        (start_x + palette_width, start_y + palette_height),
        (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, opacity * 0.8, frame, 1 - opacity * 0.8, 0, frame)

    title = "[COLOR]"
    cv2.putText(
        frame,
        title,
        (start_x, start_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    y_offset = start_y + 35
    for color_info in dominant_colors[:5]:
        rgb = color_info.get("rgb", [128, 128, 128])
        percentage = color_info.get("percentage", 0)

        if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        else:
            bgr = (128, 128, 128)

        cv2.rectangle(
            frame,
            (start_x, y_offset),
            (start_x + swatch_size, y_offset + swatch_size),
            bgr, -1
        )

        cv2.rectangle(
            frame,
            (start_x, y_offset),
            (start_x + swatch_size, y_offset + swatch_size),
            (255, 255, 255), 1
        )

        label = f"{percentage:.0f}%"
        cv2.putText(
            frame,
            label,
            (start_x + swatch_size + 10, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y_offset += swatch_size + swatch_spacing

    return frame


__all__ = [
    "apply_all_overlays",
    "draw_faces_with_emotions",
    "draw_rule_of_thirds",
    "draw_lighting_indicator",
    "draw_camera_movement_indicator",
    "draw_shot_type_badge",
    "draw_color_palette",
]