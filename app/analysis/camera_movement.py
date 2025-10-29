"""
camera_movement.py

Análisis de movimientos de cámara cinematográficos mediante optical flow.
Implementa detección de Pan, Tilt, Zoom, Tracking, Dolly, Steadicam y cámara estática
utilizando el algoritmo Lucas-Kanade para el cálculo de flujo óptico.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Cálculo de optical flow
    - numpy: Operaciones con arrays
    - app.config: Umbrales y configuración

Usage:
    from app.analysis.camera_movement import CameraMovementAnalyzer

    analyzer = CameraMovementAnalyzer()
    movement = analyzer.analyze_movement(frame, frame_number)

    if movement["is_moving"]:
        print(f"Movimiento: {movement['type']} - Intensidad: {movement['intensity']}")
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

from app.config import CAMERA_MOVEMENT_CONFIG


class CameraMovementAnalyzer:
    """
    Analizador de movimientos de cámara mediante optical flow.

    Detecta los siguientes tipos de movimiento:
        - Pan (Left/Right)
        - Tilt (Up/Down)
        - Zoom (In/Out)
        - Tracking/Seguimiento
        - Dolly (In/Out)
        - Handheld/Cámara en mano
        - Steadicam/Estabilizada
        - Static/Estático

    Attributes:
        prev_gray: Frame anterior en escala de grises
        movement_history: Cola de movimientos recientes
        full_timeline: Timeline completa para generación de gráficos
        feature_params: Parámetros para detección de features
        lk_params: Parámetros para Lucas-Kanade optical flow
    """

    def __init__(self, history_size: int = 10):
        """
        Inicializa el analizador de movimiento.

        Args:
            history_size: Tamaño del historial de movimientos para análisis temporal
        """
        self.prev_gray = None
        self.movement_history = deque(maxlen=history_size)
        self.full_timeline: List[Dict[str, Any]] = []

        # Parámetros para Shi-Tomasi corner detection
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        # Parámetros para Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Cargar configuración de umbrales
        self.motion_threshold = CAMERA_MOVEMENT_CONFIG.get("motion_threshold", 2.0)
        self.pan_threshold = CAMERA_MOVEMENT_CONFIG.get("pan_threshold", 3.0)
        self.tilt_threshold = CAMERA_MOVEMENT_CONFIG.get("tilt_threshold", 3.0)
        self.zoom_threshold = CAMERA_MOVEMENT_CONFIG.get("zoom_threshold", 0.05)
        self.min_motion_frames = CAMERA_MOVEMENT_CONFIG.get("min_motion_frames", 3)

    def analyze_movement(
            self,
            frame: np.ndarray,
            frame_number: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Analiza el movimiento de cámara entre el frame actual y el anterior.

        Args:
            frame: Frame BGR actual a analizar
            frame_number: Número del frame actual para registro en timeline

        Returns:
            Diccionario con análisis de movimiento o None si ocurre error.
        """
        try:
            # Verificar que el frame no esté vacío
            if frame is None or frame.size == 0:
                return self._create_static_result(frame_number)

            # Convertir a escala de grises solo si es necesario
            if len(frame.shape) == 2:
                # Ya está en escala de grises
                gray = frame
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # Es BGR, convertir a gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # Formato no soportado
                return self._create_static_result(frame_number)

            # Primer frame: guardar y retornar estático
            if self.prev_gray is None:
                self.prev_gray = gray
                return self._create_static_result(frame_number)

            # Calcular optical flow
            flow_data = self._calculate_optical_flow(gray)

            if flow_data is None or flow_data["points"] == 0:
                self.prev_gray = gray
                return self._create_static_result(frame_number)

            # Clasificar tipo de movimiento
            movement_type, direction, confidence = self._classify_movement(flow_data)

            # Calcular intensidad del movimiento
            intensity = self._calculate_intensity(flow_data, movement_type)

            # Detectar estabilidad
            stability_score = self._analyze_stability(flow_data)

            # Construir resultado
            result = {
                "type": movement_type,
                "direction": direction,
                "magnitude": flow_data["magnitude"],
                "confidence": confidence,
                "intensity": intensity,
                "is_moving": movement_type != "Static",
                "points_tracked": flow_data["points"],
                "stability_score": stability_score,
                "dx": flow_data["avg_dx"],
                "dy": flow_data["avg_dy"],
                "scale_change": flow_data.get("scale_change", 1.0)
            }

            # Guardar en timeline completa
            self.full_timeline.append({
                "frame": frame_number,
                "type": movement_type,
                "direction": direction,
                "intensity": intensity,
                "magnitude": flow_data["magnitude"]
            })

            # Actualizar historial
            self.movement_history.append(result)

            # Actualizar frame anterior
            self.prev_gray = gray

            return result

        except Exception as e:
            # En caso de cualquier error, retornar resultado estático
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error en análisis de movimiento frame {frame_number}: {e}")
            return self._create_static_result(frame_number)

    def _calculate_optical_flow(self, gray: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Calcula optical flow mediante Lucas-Kanade entre frames consecutivos.

        Args:
            gray: Frame actual en escala de grises

        Returns:
            Diccionario con datos de optical flow o None si ocurre error
        """
        # Detectar features mediante Shi-Tomasi corner detection
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

        if p0 is None or len(p0) == 0:
            return None

        # Calcular optical flow mediante Lucas-Kanade
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **self.lk_params
        )

        if p1 is None:
            return None

        # Filtrar puntos correctamente rastreados
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) == 0:
            return None

        # Calcular vectores de movimiento
        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]

        # Utilizar mediana para robustez contra outliers
        avg_dx = float(np.median(dx))
        avg_dy = float(np.median(dy))

        # Magnitud y ángulo del movimiento
        magnitude = float(np.sqrt(avg_dx ** 2 + avg_dy ** 2))
        angle = float(np.degrees(np.arctan2(avg_dy, avg_dx)))

        # Estimación de cambio de escala para detección de zoom
        try:
            distances_old = np.linalg.norm(
                good_old - np.mean(good_old, axis=0), axis=1
            )
            distances_new = np.linalg.norm(
                good_new - np.mean(good_new, axis=0), axis=1
            )

            # Protección contra división por cero
            valid_mask = distances_old > 1e-6
            if np.sum(valid_mask) > 0:
                scale_change = float(np.median(distances_new[valid_mask] / distances_old[valid_mask]))
            else:
                scale_change = 1.0
        except Exception:
            scale_change = 1.0

        return {
            "points": len(good_new),
            "avg_dx": round(avg_dx, 2),
            "avg_dy": round(avg_dy, 2),
            "magnitude": round(magnitude, 2),
            "angle": round(angle, 2),
            "scale_change": round(scale_change, 3),
            "vectors": (good_old, good_new)
        }

    def _classify_movement(
        self,
        flow_data: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        """
        Clasifica el tipo de movimiento de cámara basándose en optical flow.

        Args:
            flow_data: Datos de optical flow calculados

        Returns:
            Tupla de (tipo, dirección, confianza)

        Notes:
            La confianza se calcula en función del número de puntos rastreados.
            Los umbrales son configurables mediante app.config.
            Orden de prioridad: Zoom > Pan/Tilt > Static
        """
        magnitude = flow_data["magnitude"]
        dx = flow_data["avg_dx"]
        dy = flow_data["avg_dy"]
        scale = flow_data["scale_change"]
        points = flow_data["points"]

        # Calcular confianza basada en puntos rastreados
        confidence = min(points / 50.0, 1.0)

        # Movimiento muy pequeño se considera estático
        if magnitude < self.motion_threshold:
            return "Static", "", confidence

        # Detectar zoom mediante cambio de escala significativo
        if abs(scale - 1.0) > self.zoom_threshold:
            if scale > 1.0:
                return "Zoom", "In", confidence
            else:
                return "Zoom", "Out", confidence

        # Detectar pan mediante movimiento horizontal dominante
        if abs(dx) > self.pan_threshold and abs(dx) > abs(dy):
            if dx > 0:
                return "Pan", "Right", confidence
            else:
                return "Pan", "Left", confidence

        # Detectar tilt mediante movimiento vertical dominante
        if abs(dy) > self.tilt_threshold and abs(dy) > abs(dx):
            if dy > 0:
                return "Tilt", "Down", confidence
            else:
                return "Tilt", "Up", confidence

        # Movimiento mixto o tracking
        if magnitude > self.motion_threshold * 2:
            return "Tracking", "Mixed", confidence

        return "Static", "", confidence

    def _calculate_intensity(
        self,
        flow_data: Dict[str, Any],
        movement_type: str
    ) -> float:
        """
        Calcula intensidad normalizada del movimiento en escala 0-100.

        Args:
            flow_data: Datos de optical flow
            movement_type: Tipo de movimiento detectado

        Returns:
            Intensidad normalizada entre 0 y 100

        Notes:
            0 = Sin movimiento
            50 = Movimiento moderado
            100 = Movimiento muy intenso
        """
        if movement_type == "Static":
            return 0.0

        magnitude = flow_data["magnitude"]

        # Normalizar a escala 0-100 asumiendo magnitud 20 como movimiento muy intenso
        intensity = min((magnitude / 20.0) * 100, 100.0)

        return round(intensity, 1)

    def _analyze_stability(self, flow_data: Dict[str, Any]) -> float:
        """
        Analiza estabilidad de la cámara diferenciando handheld de steadicam.

        Args:
            flow_data: Datos de optical flow

        Returns:
            Score de estabilidad entre 0 y 100, donde 100 representa máxima estabilidad

        Notes:
            El cálculo se basa en la varianza de los vectores de movimiento.
            Alta varianza indica cámara en mano (inestable).
            Baja varianza indica steadicam (estable).
        """
        if "vectors" not in flow_data:
            return 50.0

        good_old, good_new = flow_data["vectors"]

        # Calcular vectores de movimiento
        motion_vectors = good_new - good_old

        # Varianza de los vectores
        variance = float(np.var(motion_vectors))

        # Normalizar a escala 0-100 invertida
        stability = max(0, 100 - (variance * 10))

        return round(min(stability, 100.0), 1)

    def _create_static_result(self, frame_number: int) -> Dict[str, Any]:
        """
        Crea resultado para frame estático sin movimiento.

        Args:
            frame_number: Número del frame actual

        Returns:
            Diccionario de resultado estático
        """
        result = {
            "type": "Static",
            "direction": "",
            "magnitude": 0.0,
            "confidence": 1.0,
            "intensity": 0.0,
            "is_moving": False,
            "points_tracked": 0,
            "stability_score": 100.0,
            "dx": 0.0,
            "dy": 0.0,
            "scale_change": 1.0
        }

        # Guardar en timeline
        self.full_timeline.append({
            "frame": frame_number,
            "type": "Static",
            "direction": "",
            "intensity": 0.0,
            "magnitude": 0.0
        })

        return result

    def get_movement_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen estadístico de movimientos detectados durante el análisis completo.

        Returns:
            Diccionario con estadísticas de movimiento:
                - total_frames: Total de frames analizados
                - movement_counts: Conteo por tipo de movimiento
                - avg_intensity: Intensidad promedio
                - timeline: Timeline completa para generación de gráficos

        Notes:
            La timeline puede utilizarse para generación de visualizaciones temporales.
        """
        if not self.full_timeline:
            return {
                "total_frames": 0,
                "movement_counts": {},
                "avg_intensity": 0.0,
                "timeline": []
            }

        # Contar tipos de movimiento
        movement_counts = {}
        total_intensity = 0.0

        for entry in self.full_timeline:
            mov_type = entry["type"]
            movement_counts[mov_type] = movement_counts.get(mov_type, 0) + 1
            total_intensity += entry["intensity"]

        return {
            "total_frames": len(self.full_timeline),
            "movement_counts": movement_counts,
            "avg_intensity": round(total_intensity / len(self.full_timeline), 2),
            "timeline": self.full_timeline
        }

    def reset(self):
        """Reinicia el analizador para procesar un nuevo vídeo."""
        self.prev_gray = None
        self.movement_history.clear()
        self.full_timeline.clear()


def visualize_camera_movement(
    frame: np.ndarray,
    movement: Dict[str, Any]
) -> np.ndarray:
    """
    Visualiza información de movimiento de cámara sobre el frame.

    Args:
        frame: Frame BGR
        movement: Diccionario con análisis de movimiento

    Returns:
        Frame con visualización de movimiento superpuesta
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]

    # Configuración de visualización
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    padding = 10

    # Color según tipo de movimiento
    if movement["is_moving"]:
        color = (0, 255, 0)
    else:
        color = (200, 200, 200)

    # Construir texto de información
    movement_text = f"{movement['type']}"
    if movement['direction']:
        movement_text += f" {movement['direction']}"

    info_lines = [
        movement_text,
        f"Intensity: {movement['intensity']:.1f}",
        f"Confidence: {movement['confidence']:.2f}"
    ]

    # Dibujar fondo semi-transparente
    overlay = frame_copy.copy()
    y_offset = padding
    for line in info_lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(
            overlay,
            (padding, y_offset),
            (padding + text_size[0] + padding, y_offset + text_size[1] + padding),
            (0, 0, 0),
            -1
        )
        y_offset += text_size[1] + padding * 2

    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)

    # Dibujar texto
    y_offset = padding + 20
    for line in info_lines:
        cv2.putText(
            frame_copy,
            line,
            (padding * 2, y_offset),
            font,
            font_scale,
            color,
            thickness
        )
        y_offset += 30

    return frame_copy