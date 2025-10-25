"""
camera_movement.py

Módulo para detección y análisis de movimientos de cámara en secuencias de video
mediante técnicas de flujo óptico. Clasifica movimientos cinematográficos y evalúa
estabilidad de cámara con métricas cuantitativas.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum
from collections import deque


class CameraMovement(Enum):
    """
    Enumeración de tipos de movimiento de cámara detectables.

    Define los diferentes movimientos cinematográficos que el sistema
    puede identificar mediante análisis de flujo óptico.
    """
    STATIC = "Estático"
    PAN_LEFT = "Pan Izquierda"
    PAN_RIGHT = "Pan Derecha"
    TILT_UP = "Tilt Arriba"
    TILT_DOWN = "Tilt Abajo"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    TRACKING = "Tracking/Seguimiento"
    DOLLY_IN = "Dolly In"
    DOLLY_OUT = "Dolly Out"
    HANDHELD = "Cámara en Mano"
    STEADICAM = "Steadicam/Estabilizada"


class CameraMovementAnalyzer:
    """
    Analizador de movimientos de cámara mediante flujo óptico.

    Utiliza el algoritmo Lucas-Kanade para detectar puntos característicos
    y analizar su desplazamiento entre frames consecutivos. Clasifica
    automáticamente el tipo de movimiento, calcula la intensidad y evalúa
    la estabilidad de la cámara.

    Attributes:
        prev_frame (Optional[np.ndarray]): Frame anterior almacenado para comparación.
        prev_gray (Optional[np.ndarray]): Versión en escala de grises del frame anterior.
        movement_history (deque): Cola de historial de movimientos recientes con
            tamaño máximo definido en la inicialización.
        full_timeline (List[Dict]): Lista completa de datos por frame para generación
            de gráficos temporales.
        feature_params (Dict): Parámetros para detección de esquinas mediante algoritmo
            Shi-Tomasi (goodFeaturesToTrack).
        lk_params (Dict): Parámetros para cálculo de flujo óptico mediante algoritmo
            Lucas-Kanade piramidal.
    """

    def __init__(self, history_size: int = 10):
        """
        Inicializa el analizador de movimientos de cámara.

        Configura los parámetros para detección de características y cálculo de
        flujo óptico. Inicializa las estructuras de datos para almacenamiento
        de historial temporal.

        Args:
            history_size: Tamaño de la ventana de historial para análisis temporal
                de movimientos. Define cuántos frames recientes se mantienen en
                memoria para cálculo de estadísticas. Por defecto 10 frames.
        """
        self.prev_frame = None
        self.prev_gray = None
        self.movement_history = deque(maxlen=history_size)
        self.full_timeline = []

        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def analyze_movement(self, frame: np.ndarray, frame_number: int = 0) -> Dict:
        """
        Analiza el movimiento de cámara entre el frame actual y el anterior.

        Args:
            frame: Frame actual en escala de grises o BGR.
            frame_number: Número de frame en la secuencia de video.

        Returns:
            Diccionario con análisis de movimiento.
        """
        # Verificar si ya está en escala de grises
        if len(frame.shape) == 2:
            # Ya es grayscale (2D)
            gray = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            # Ya es grayscale con dimensión extra (H, W, 1)
            gray = frame.squeeze()
        elif len(frame.shape) == 3:
            # Convertir BGR a grayscale (H, W, 3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Caso inesperado
            raise ValueError(f"Frame con forma inesperada: {frame.shape}")

        if self.prev_gray is None:
            self.prev_gray = gray
            return {
                "movement_type": CameraMovement.STATIC.value,
                "confidence": 0.0,
                "motion_vectors": None,
                "intensity": 0.0
            }

        flow_data = self._calculate_optical_flow(gray)
        movement_type, confidence = self._classify_movement(flow_data)
        stability = self._analyze_stability(flow_data)
        intensity = self._calculate_intensity(flow_data, movement_type)

        self.full_timeline.append({
            "frame": frame_number,
            "type": movement_type.value,
            "intensity": intensity
        })

        self.movement_history.append({
            "type": movement_type.value,
            "vectors": flow_data
        })

        self.prev_gray = gray

        return {
            "movement_type": movement_type.value,
            "confidence": confidence,
            "motion_vectors": flow_data,
            "stability": stability,
            "is_moving": movement_type != CameraMovement.STATIC,
            "intensity": intensity
        }

    def _calculate_optical_flow(self, gray: np.ndarray) -> Dict:
        """
        Calcula el flujo óptico entre el frame anterior y el actual.

        Utiliza el algoritmo Lucas-Kanade piramidal para detectar y seguir puntos
        característicos entre frames consecutivos. Detecta esquinas mediante
        Shi-Tomasi y calcula estadísticas del movimiento incluyendo dirección,
        magnitud y cambio de escala entre puntos rastreados.

        Args:
            gray: Frame actual en escala de grises como numpy array 2D.

        Returns:
            Diccionario con métricas del flujo óptico:
                points (int): Número de puntos rastreados exitosamente.
                avg_dx (float): Desplazamiento promedio horizontal en píxeles.
                avg_dy (float): Desplazamiento promedio vertical en píxeles.
                magnitude (float): Magnitud euclidiana del vector de movimiento promedio.
                angle (float): Ángulo de dirección del movimiento en grados [-180, 180].
                scale_change (float): Factor de cambio de escala relativo. Valores > 1
                    indican expansión (zoom in), < 1 contracción (zoom out).
                vectors (Optional[Tuple]): Tupla con arrays de puntos antiguos y nuevos
                    para visualización. None si no hay puntos detectados.

        Notes:
            Utiliza la mediana en lugar de la media para mayor robustez ante outliers.
            Retorna valores cero si no se detectan suficientes puntos característicos.
        """
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)

        if p0 is None:
            return {
                "points": 0,
                "avg_dx": 0,
                "avg_dy": 0,
                "magnitude": 0,
                "angle": 0,
                "scale_change": 0
            }

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, p0, None, **self.lk_params
        )

        if p1 is None:
            return {
                "points": 0,
                "avg_dx": 0,
                "avg_dy": 0,
                "magnitude": 0,
                "angle": 0,
                "scale_change": 0
            }

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        if len(good_new) == 0:
            return {
                "points": 0,
                "avg_dx": 0,
                "avg_dy": 0,
                "magnitude": 0,
                "angle": 0,
                "scale_change": 0
            }

        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]

        avg_dx = float(np.median(dx))
        avg_dy = float(np.median(dy))

        magnitude = float(np.sqrt(avg_dx**2 + avg_dy**2))
        angle = float(np.degrees(np.arctan2(avg_dy, avg_dx)))

        distances_old = np.linalg.norm(good_old - np.mean(good_old, axis=0), axis=1)
        distances_new = np.linalg.norm(good_new - np.mean(good_new, axis=0), axis=1)

        scale_change = float(np.median(distances_new / (distances_old + 1e-6)))

        return {
            "points": len(good_new),
            "avg_dx": round(avg_dx, 2),
            "avg_dy": round(avg_dy, 2),
            "magnitude": round(magnitude, 2),
            "angle": round(angle, 2),
            "scale_change": round(scale_change, 3),
            "vectors": (good_old, good_new)
        }

    def _classify_movement(self, flow_data: Dict) -> Tuple[CameraMovement, float]:
        """
        Clasifica el tipo de movimiento de cámara según datos de flujo óptico.

        Analiza la magnitud, dirección y cambio de escala del flujo para determinar
        el tipo de movimiento cinematográfico. Calcula nivel de confianza basado en
        coherencia de vectores y número de puntos rastreados.

        Args:
            flow_data: Diccionario con datos de flujo óptico calculados previamente
                por _calculate_optical_flow.

        Returns:
            Tupla con dos elementos:
                - CameraMovement: Enum del tipo de movimiento clasificado.
                - float: Nivel de confianza de la clasificación en rango [0.0, 1.0].

        Notes:
            Umbrales de clasificación:
                - Magnitud < 2: Movimiento estático
                - Magnitud >= 2 con análisis direccional:
                    * Pan: Predominancia horizontal (abs(dx) > 2 * abs(dy))
                    * Tilt: Predominancia vertical (abs(dy) > 2 * abs(dx))
                    * Zoom: Cambio de escala significativo (>1.05 o <0.95)
                    * Tracking: Movimiento complejo sin predominancia clara
            La confianza depende del número de puntos rastreados y magnitud del movimiento.
        """
        magnitude = flow_data["magnitude"]
        dx = flow_data["avg_dx"]
        dy = flow_data["avg_dy"]
        scale = flow_data["scale_change"]

        static_threshold = 2
        zoom_threshold = 0.05

        if magnitude < static_threshold:
            return CameraMovement.STATIC, 1.0

        if abs(dx) > abs(dy) * 2:
            movement = CameraMovement.PAN_RIGHT if dx > 0 else CameraMovement.PAN_LEFT
        elif abs(dy) > abs(dx) * 2:
            movement = CameraMovement.TILT_DOWN if dy > 0 else CameraMovement.TILT_UP
        elif scale > (1.0 + zoom_threshold):
            movement = CameraMovement.ZOOM_IN
        elif scale < (1.0 - zoom_threshold):
            movement = CameraMovement.ZOOM_OUT
        else:
            movement = CameraMovement.TRACKING

        confidence = min(1.0, (flow_data["points"] / 50) * (magnitude / 10))
        confidence = max(0.3, min(confidence, 1.0))

        return movement, round(confidence, 2)

    def _calculate_intensity(self, flow_data: Dict, movement_type: CameraMovement) -> float:
        """
        Calcula la intensidad normalizada del movimiento de cámara.

        Escala la magnitud del movimiento a un rango [0, 100] donde valores altos
        indican movimientos rápidos o pronunciados. Para zooms, incorpora también
        el factor de cambio de escala en el cálculo.

        Args:
            flow_data: Diccionario con datos del flujo óptico.
            movement_type: Tipo de movimiento clasificado previamente.

        Returns:
            Intensidad del movimiento en escala [0.0, 100.0] redondeada a un decimal.

        Notes:
            Fórmula base: intensity = (magnitude / 20.0) * 100
            Para zooms se añade: scale_contribution = abs(scale_change - 1.0) * 200
            La intensidad se limita al máximo de 100.0.
        """
        if movement_type == CameraMovement.STATIC:
            return 0.0

        magnitude = flow_data["magnitude"]
        scale_change = abs(flow_data["scale_change"] - 1.0)

        intensity = min(100.0, (magnitude / 20.0) * 100)

        if movement_type in [CameraMovement.ZOOM_IN, CameraMovement.ZOOM_OUT]:
            intensity = min(100.0, intensity + (scale_change * 200))

        return round(intensity, 1)

    def _analyze_stability(self, flow_data: Dict) -> Dict:
        """
        Analiza la estabilidad de la cámara mediante varianza del movimiento.

        Evalúa la consistencia del flujo óptico calculando la varianza de los vectores
        de desplazamiento. Una varianza baja indica movimientos uniformes y estables
        (steadicam, trípode), mientras que varianza alta sugiere movimiento irregular
        (cámara en mano, handheld).

        Args:
            flow_data: Diccionario con datos del flujo óptico incluyendo vectores
                de puntos rastreados.

        Returns:
            Diccionario con análisis de estabilidad:
                is_stable (bool): True si la cámara está estabilizada según umbral.
                shake_level (float): Nivel de temblor normalizado en rango [0.0, 1.0]
                    donde 0.0 es perfectamente estable y 1.0 es muy inestable.
                variance (float): Varianza total del movimiento (suma de varianzas
                    en ejes X e Y).
                camera_style (str): Estilo de cámara detectado, valores posibles:
                    'Steadicam/Estabilizada' o 'Cámara en Mano'.

        Notes:
            Umbrales de varianza para clasificación:
                - Varianza < 2: Steadicam (shake_level=0.0, stable=True)
                - Varianza < 5: Estable (shake_level=0.3, stable=True)
                - Varianza < 15: Handheld moderado (shake_level=0.6, stable=False)
                - Varianza >= 15: Handheld pronunciado (shake_level=1.0, stable=False)
        """
        if "vectors" not in flow_data or flow_data["vectors"] is None:
            return {
                "is_stable": True,
                "shake_level": 0.0,
                "camera_style": CameraMovement.STATIC.value
            }

        good_old, good_new = flow_data["vectors"]

        dx = good_new[:, 0] - good_old[:, 0]
        dy = good_new[:, 1] - good_old[:, 1]

        variance_x = float(np.var(dx))
        variance_y = float(np.var(dy))
        total_variance = variance_x + variance_y

        if total_variance < 2:
            shake_level = 0.0
            is_stable = True
            camera_style = CameraMovement.STEADICAM
        elif total_variance < 5:
            shake_level = 0.3
            is_stable = True
            camera_style = CameraMovement.STEADICAM
        elif total_variance < 15:
            shake_level = 0.6
            is_stable = False
            camera_style = CameraMovement.HANDHELD
        else:
            shake_level = 1.0
            is_stable = False
            camera_style = CameraMovement.HANDHELD

        return {
            "is_stable": is_stable,
            "shake_level": round(shake_level, 2),
            "variance": round(total_variance, 2),
            "camera_style": camera_style.value
        }

    def get_movement_summary(self) -> Dict:
        """
        Genera un resumen estadístico de los movimientos detectados.

        Analiza el historial de movimientos para determinar patrones predominantes
        y calcular la distribución porcentual de cada tipo de movimiento en la ventana
        temporal definida por history_size.

        Returns:
            Diccionario con resumen estadístico:
                total_frames (int): Total de frames analizados en el historial.
                predominant_movement (str): Tipo de movimiento más frecuente.
                most_common (str): Alias de predominant_movement para compatibilidad.
                distribution (Dict[str, float]): Diccionario con distribución porcentual
                    de cada tipo de movimiento detectado, donde las claves son los
                    nombres de movimientos y los valores son porcentajes redondeados
                    a un decimal.

        Notes:
            Si el historial está vacío, retorna valores por defecto con movimiento
            STATIC y distribución vacía.
        """
        if not self.movement_history:
            return {
                "total_frames": 0,
                "predominant_movement": CameraMovement.STATIC.value,
                "most_common": CameraMovement.STATIC.value,
                "distribution": {}
            }

        movement_counts = {}
        for entry in self.movement_history:
            mov_type = entry["type"]
            movement_counts[mov_type] = movement_counts.get(mov_type, 0) + 1

        total = len(self.movement_history)
        movement_distribution = {
            k: round(v / total * 100, 1)
            for k, v in movement_counts.items()
        }

        predominant = max(movement_counts.items(), key=lambda x: x[1])[0]

        return {
            "total_frames": total,
            "predominant_movement": predominant,
            "most_common": predominant,
            "distribution": movement_distribution
        }

    def get_timeline_data(self) -> List[Dict]:
        """
        Obtiene la timeline completa de movimientos para visualización.

        Retorna todos los datos de análisis de cada frame procesado desde la
        inicialización del analizador. Útil para generar gráficos temporales
        de movimiento y análisis post-procesamiento.

        Returns:
            Lista de diccionarios con datos por frame, cada elemento contiene:
                frame (int): Número de frame en la secuencia.
                type (str): Tipo de movimiento detectado.
                intensity (float): Intensidad del movimiento en escala [0, 100].

        Notes:
            A diferencia de movement_history que tiene tamaño limitado, full_timeline
            mantiene datos de todos los frames procesados hasta que se llame a reset().
        """
        return self.full_timeline

    def reset(self):
        """
        Reinicia el estado interno del analizador.

        Limpia todas las estructuras de datos temporales incluyendo historial de
        movimientos, timeline completa y frames almacenados. Útil para comenzar
        el análisis de un nuevo video sin necesidad de crear una nueva instancia
        del analizador, manteniendo la configuración de parámetros.

        Notes:
            Después de llamar a reset(), el próximo frame procesado será tratado
            como el primer frame de una nueva secuencia (sin frame anterior para
            comparación).
        """
        self.prev_gray = None
        self.movement_history.clear()
        self.full_timeline.clear()


def visualize_camera_movement(frame: np.ndarray, movement_info: Dict,
                              flow_data: Dict = None) -> np.ndarray:
    """
    Visualiza el movimiento de cámara detectado mediante overlay gráfico en el frame.

    Renderiza información textual del tipo de movimiento y estilo de cámara en la
    esquina inferior izquierda del frame, con fondo negro semi-transparente para
    mejorar legibilidad. Opcionalmente dibuja vectores de flujo óptico si los datos
    están disponibles.

    Args:
        frame: Frame de video en formato BGR (numpy array con dimensiones H x W x 3)
            sobre el cual se aplicará el overlay de visualización.
        movement_info: Diccionario con información del movimiento detectado, típicamente
            el resultado retornado por el método analyze_movement(). Debe contener al
            menos la clave 'movement_type'. Opcionalmente puede incluir 'stability' con
            información de estilo de cámara.
        flow_data: Diccionario opcional con datos de flujo óptico para visualizar
            vectores de movimiento. Si se proporciona y contiene la clave 'vectors',
            se dibujarán líneas verdes representando el flujo y círculos rojos en
            las posiciones finales de los puntos rastreados.

    Returns:
        Frame modificado con overlay de visualización aplicado. El array original
        es modificado in-place y también retornado.

    Notes:
        Los vectores de flujo se muestrean cada 20 puntos para evitar saturación
        visual en el frame. La visualización consiste en:
            - Líneas verdes: Trayectoria del movimiento de cada punto
            - Círculos rojos: Posición final de los puntos rastreados
            - Texto con fondo negro: Tipo de movimiento y estilo de cámara
    """
    h, w = frame.shape[:2]

    text = f"Movimiento: {movement_info['movement_type']}"
    if movement_info.get('stability'):
        text += f" | Estabilidad: {movement_info['stability']['camera_style']}"

    position = (20, h - 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (position[0] - 5, position[1] - text_h - 5),
        (position[0] + text_w + 5, position[1] + baseline + 5),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame, text, position, font, font_scale,
        (0, 255, 0), thickness
    )

    if flow_data and "vectors" in flow_data and flow_data["vectors"] is not None:
        good_old, good_new = flow_data["vectors"]

        step = max(1, len(good_old) // 20)
        for i in range(0, len(good_old), step):
            old_pt = tuple(good_old[i].astype(int))
            new_pt = tuple(good_new[i].astype(int))

            cv2.line(frame, old_pt, new_pt, (0, 255, 0), 1)
            cv2.circle(frame, new_pt, 3, (0, 0, 255), -1)

    return frame