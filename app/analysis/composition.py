"""
composition.py

Análisis de composición cinematográfica y principios visuales.
Evalúa regla de tercios, simetría, líneas guía y balance visual mediante
técnicas de detección de features, análisis de bordes y comparación estructural.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Detección de edges, líneas y features
    - numpy: Operaciones con arrays y cálculos
    - app.config: Configuración de umbrales

Usage:
    from app.analysis.composition import CompositionAnalyzer

    analyzer = CompositionAnalyzer()
    analysis = analyzer.analyze(frame)

    print(f"Regla de tercios: {analysis['rule_of_thirds']['complies']}")
    print(f"Simetría: {analysis['symmetry']['is_symmetric']}")
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from app.config import COMPOSITION_CONFIG


class CompositionAnalyzer:
    """
    Analizador de composición cinematográfica.

    Analiza principios fundamentales de composición visual:
        - Regla de tercios (Rule of Thirds)
        - Simetría (vertical y horizontal)
        - Líneas guía (Leading Lines)
        - Balance visual
        - Puntos de interés

    Attributes:
        config: Configuración de composición desde config.py
        tolerance: Tolerancia para puntos en intersecciones (0-1)
    """

    def __init__(self):
        """Inicializa el analizador de composición."""
        self.config = COMPOSITION_CONFIG
        self.tolerance = self.config.get("rule_of_thirds", {}).get("tolerance", 0.1)

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Realiza análisis completo de composición.

        Args:
            frame: Frame BGR a analizar

        Returns:
            Diccionario con todos los análisis:
                rule_of_thirds: Análisis de regla de tercios
                symmetry: Análisis de simetría
                leading_lines: Detección de líneas guía
                balance: Análisis de balance visual
                overall_score: Puntuación general 0-100

        Notes:
            La puntuación overall_score combina todos los análisis mediante
            ponderación. Retorna valores por defecto en caso de error.
        """
        rule_of_thirds = self.analyze_rule_of_thirds(frame)
        symmetry = self.analyze_symmetry(frame)
        leading_lines = self.detect_leading_lines(frame)
        balance = self.analyze_balance(frame)

        overall_score = self._calculate_overall_score(
            rule_of_thirds, symmetry, leading_lines, balance
        )

        return {
            "rule_of_thirds": rule_of_thirds,
            "symmetry": symmetry,
            "leading_lines": leading_lines,
            "balance": balance,
            "overall_score": overall_score
        }

    def analyze_rule_of_thirds(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analiza si la composición sigue la regla de tercios.

        La regla de tercios divide el frame en 9 partes iguales mediante una
        cuadrícula 3x3. Los puntos de interés deberían estar cerca de las
        cuatro intersecciones de esta cuadrícula para lograr una composición
        visualmente equilibrada.

        Args:
            frame: Frame BGR a analizar

        Returns:
            Diccionario con análisis:
                complies: Boolean indicando si cumple la regla
                score: Puntuación 0-100
                points_near_intersections: Número de puntos de interés cerca
                intersection_points: Coordenadas de las 4 intersecciones
                interest_points: Puntos de interés detectados

        Notes:
            Utiliza detección de esquinas de Shi-Tomasi para identificar puntos
            de interés visual. La tolerancia es configurable mediante config.py.
            El score se calcula en función de la proximidad de los puntos detectados
            a las intersecciones de la cuadrícula.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = frame.shape[:2]

        # Calcular puntos de intersección de la regla de tercios
        x1, x2 = w // 3, 2 * w // 3
        y1, y2 = h // 3, 2 * h // 3

        intersection_points = [
            (x1, y1), (x2, y1),
            (x1, y2), (x2, y2)
        ]

        # Detectar puntos de interés mediante Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=20,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )

        if corners is None:
            return {
                "complies": False,
                "score": 0.0,
                "points_near_intersections": 0,
                "intersection_points": intersection_points,
                "interest_points": []
            }

        interest_points = [(int(x[0][0]), int(x[0][1])) for x in corners]

        # Contar puntos cerca de intersecciones
        tolerance_pixels = min(w, h) * self.tolerance
        points_near = 0

        for point in interest_points:
            for intersection in intersection_points:
                distance = np.sqrt(
                    (point[0] - intersection[0])**2 +
                    (point[1] - intersection[1])**2
                )
                if distance < tolerance_pixels:
                    points_near += 1
                    break

        score = min((points_near / 4.0) * 100, 100)
        complies = points_near >= 2

        return {
            "complies": complies,
            "score": round(score, 1),
            "points_near_intersections": points_near,
            "intersection_points": intersection_points,
            "interest_points": interest_points
        }

    def analyze_symmetry(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analiza simetría vertical y horizontal del frame.

        Compara las mitades opuestas del frame para determinar el grado de
        simetría mediante cálculo de diferencias estructurales. Evalúa tanto
        simetría vertical (izquierda-derecha) como horizontal (arriba-abajo).

        Args:
            frame: Frame BGR a analizar

        Returns:
            Diccionario con análisis:
                is_symmetric: Boolean indicando si es simétrico
                score: Puntuación de simetría 0-100
                axis: Eje de simetría ("vertical", "horizontal", "both", None)
                vertical_score: Score de simetría vertical
                horizontal_score: Score de simetría horizontal

        Notes:
            Compara mitades opuestas mediante diferencia absoluta de píxeles.
            El umbral de simetría es configurable mediante config.py.
            Una imagen es considerada simétrica si el score supera el umbral
            configurado (por defecto 85%).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # Analizar simetría vertical
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)

        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]

        vertical_diff = cv2.absdiff(left_half, right_half)
        vertical_score = 100 * (1 - np.mean(vertical_diff) / 255.0)

        # Analizar simetría horizontal
        top_half = gray[:h//2, :]
        bottom_half = cv2.flip(gray[h//2:, :], 0)

        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]

        horizontal_diff = cv2.absdiff(top_half, bottom_half)
        horizontal_score = 100 * (1 - np.mean(horizontal_diff) / 255.0)

        # Determinar eje de simetría
        threshold = self.config.get("symmetry", {}).get("threshold", 0.85) * 100

        vertical_symmetric = vertical_score >= threshold
        horizontal_symmetric = horizontal_score >= threshold

        if vertical_symmetric and horizontal_symmetric:
            axis = "both"
            is_symmetric = True
            score = (vertical_score + horizontal_score) / 2
        elif vertical_symmetric:
            axis = "vertical"
            is_symmetric = True
            score = vertical_score
        elif horizontal_symmetric:
            axis = "horizontal"
            is_symmetric = True
            score = horizontal_score
        else:
            axis = None
            is_symmetric = False
            score = max(vertical_score, horizontal_score)

        return {
            "is_symmetric": is_symmetric,
            "score": round(score, 1),
            "axis": axis,
            "vertical_score": round(vertical_score, 1),
            "horizontal_score": round(horizontal_score, 1)
        }

    def detect_leading_lines(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detecta líneas guía en la composición.

        Las líneas guía (leading lines) son elementos visuales que dirigen
        la atención del espectador hacia puntos de interés específicos en
        la imagen. Utiliza transformada de Hough para detección de líneas
        en bordes detectados mediante Canny.

        Args:
            frame: Frame BGR a analizar

        Returns:
            Diccionario con análisis:
                has_leading_lines: Boolean indicando si hay líneas guía
                num_lines: Número de líneas detectadas
                lines: Lista de líneas [(x1, y1, x2, y2), ...]
                dominant_angle: Ángulo dominante de las líneas en grados
                line_types: Clasificación por tipo (diagonal, horizontal, vertical)

        Notes:
            Parámetros de detección:
                - Canny: umbrales 50-150
                - HoughLinesP: longitud mínima 100px, gap máximo 10px
            Las líneas se clasifican según su ángulo:
                - Horizontal: ángulo < 15° o > 165°
                - Vertical: 75° < ángulo < 105°
                - Diagonal: otros ángulos
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección de bordes con Canny
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detección de líneas con transformada de Hough
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=100,
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return {
                "has_leading_lines": False,
                "num_lines": 0,
                "lines": [],
                "dominant_angle": None,
                "line_types": []
            }

        detected_lines = []
        angles = []
        line_types = {"diagonal": 0, "horizontal": 0, "vertical": 0}

        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)

            abs_angle = abs(angle)
            if abs_angle < 15 or abs_angle > 165:
                line_types["horizontal"] += 1
            elif 75 < abs_angle < 105:
                line_types["vertical"] += 1
            else:
                line_types["diagonal"] += 1

            detected_lines.append((int(x1), int(y1), int(x2), int(y2)))

        dominant_angle = np.median(angles) if angles else None

        return {
            "has_leading_lines": True,
            "num_lines": len(detected_lines),
            "lines": detected_lines,
            "dominant_angle": round(float(dominant_angle), 1) if dominant_angle is not None else None,
            "line_types": line_types
        }

    def analyze_balance(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analiza el balance visual del frame.

        Evalúa la distribución de peso visual entre las mitades izquierda y
        derecha del frame. El peso visual se calcula mediante la suma de
        intensidades de píxeles en cada mitad.

        Args:
            frame: Frame BGR a analizar

        Returns:
            Diccionario con análisis:
                is_balanced: Boolean indicando si está balanceado
                score: Puntuación de balance 0-100
                weight_left: Peso visual lado izquierdo
                weight_right: Peso visual lado derecho
                weight_difference: Diferencia porcentual entre mitades

        Notes:
            El balance se considera adecuado cuando la diferencia entre mitades
            es inferior al umbral configurado (por defecto 30%). Un score de 100
            indica balance perfecto. El cálculo se basa en la distribución de
            píxeles brillantes en cada mitad del frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]

        weight_left = int(np.sum(left_half))
        weight_right = int(np.sum(right_half))

        total_weight = weight_left + weight_right
        if total_weight == 0:
            weight_difference = 0.0
        else:
            weight_difference = abs(weight_left - weight_right) / total_weight * 100

        threshold = self.config.get("balance", {}).get("weight_threshold", 0.3) * 100

        is_balanced = weight_difference <= threshold
        score = 100 - min(weight_difference, 100)

        return {
            "is_balanced": is_balanced,
            "score": round(score, 1),
            "weight_left": weight_left,
            "weight_right": weight_right,
            "weight_difference": round(weight_difference, 1)
        }

    def _calculate_overall_score(
        self,
        rule_of_thirds: Dict,
        symmetry: Dict,
        leading_lines: Dict,
        balance: Dict
    ) -> float:
        """
        Calcula puntuación general de composición mediante ponderación.

        Args:
            rule_of_thirds: Análisis de regla de tercios
            symmetry: Análisis de simetría
            leading_lines: Análisis de líneas guía
            balance: Análisis de balance

        Returns:
            Puntuación general entre 0 y 100

        Notes:
            Ponderación aplicada:
                - Regla de tercios: 40%
                - Balance: 30%
                - Simetría: 20%
                - Líneas guía: 10%
        """
        score = (
            rule_of_thirds["score"] * 0.4 +
            balance["score"] * 0.3 +
            symmetry["score"] * 0.2 +
            (100 if leading_lines["has_leading_lines"] else 0) * 0.1
        )

        return round(score, 1)


def visualize_composition(
    frame: np.ndarray,
    composition: Dict[str, Any]
) -> np.ndarray:
    """
    Visualiza grid de regla de tercios sobre el frame.

    Args:
        frame: Frame BGR
        composition: Diccionario con análisis de composición

    Returns:
        Frame con cuadrícula de regla de tercios superpuesta

    Notes:
        Dibuja las líneas de la cuadrícula 3x3 y marca las intersecciones
        con círculos. La visualización ayuda a evaluar la aplicación de la
        regla de tercios en la composición.
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]

    x1, x2 = w // 3, 2 * w // 3
    y1, y2 = h // 3, 2 * h // 3

    color = (0, 255, 255)
    thickness = 2

    # Dibujar líneas verticales
    cv2.line(frame_copy, (x1, 0), (x1, h), color, thickness)
    cv2.line(frame_copy, (x2, 0), (x2, h), color, thickness)

    # Dibujar líneas horizontales
    cv2.line(frame_copy, (0, y1), (w, y1), color, thickness)
    cv2.line(frame_copy, (0, y2), (w, y2), color, thickness)

    # Dibujar círculos en intersecciones
    intersection_points = composition.get("rule_of_thirds", {}).get("intersection_points", [])
    for point in intersection_points:
        cv2.circle(frame_copy, point, 8, color, -1)

    return frame_copy