"""
composition_analyzer.py

Módulo para análisis de composición cinematográfica mediante detección de bordes,
análisis geométrico y evaluación de principios compositivos clásicos como regla
de tercios, simetría y balance visual.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class CompositionAnalyzer:
    """
    Analizador de composición visual cinematográfica.

    Evalúa la calidad compositiva de frames mediante análisis cuantitativo de
    principios cinematográficos: regla de tercios, simetría, líneas dominantes
    y balance visual. Acumula métricas a lo largo de la secuencia para generar
    estadísticas globales y datos para visualización temporal.

    Attributes:
        all_rule_of_thirds_scores (List[float]): Acumulador de puntuaciones de
            regla de tercios para cada frame analizado, valores en rango [0.0, 1.0].
        all_symmetry_scores (List[float]): Acumulador de puntuaciones de simetría
            vertical para cada frame, valores en rango [0.0, 1.0].
        all_balance_scores (List[float]): Acumulador de puntuaciones de balance
            horizontal para cada frame, valores en rango [0.0, 1.0].
        all_lines_counts (List[int]): Acumulador de conteos de líneas detectadas
            mediante Hough Transform por frame.
    """

    def __init__(self):
        """
        Inicializa el analizador de composición.

        Configura los acumuladores vacíos para recopilar métricas de composición
        a lo largo de múltiples frames, permitiendo análisis temporal y generación
        de estadísticas agregadas de las características compositivas del video.
        """
        self.all_rule_of_thirds_scores = []
        self.all_symmetry_scores = []
        self.all_balance_scores = []
        self.all_lines_counts = []

    def analyze_composition(self, frame: np.ndarray) -> Dict:
        """
        Analiza la composición visual del frame mediante múltiples métricas.

        Ejecuta análisis compositivo completo evaluando el frame según principios
        fundamentales de cinematografía: regla de tercios (división en cuadrícula
        3x3), simetría (reflexión vertical y horizontal), líneas dominantes
        (Hough Transform) y balance visual (distribución de intensidades). Los
        resultados se acumulan automáticamente para análisis temporal posterior.

        Args:
            frame: Frame a analizar en formato BGR (OpenCV estándar) como numpy
                array con dimensiones (H, W, 3).

        Returns:
            Diccionario con análisis compositivo completo conteniendo:
                rule_of_thirds (Dict): Evaluación de regla de tercios con puntuación
                    normalizada, densidades por sección de cuadrícula 3x3, densidades
                    en puntos de intersección y flag de cumplimiento.
                symmetry (Dict): Métricas de simetría con valores separados para
                    simetría vertical (espejo izquierda-derecha) y horizontal
                    (espejo arriba-abajo), más evaluaciones booleanas.
                lines (Dict): Análisis de líneas detectadas mediante Hough Transform
                    con clasificación por orientación (horizontal, vertical, diagonal),
                    conteo total y dirección dominante.
                balance (Dict): Evaluación de balance visual con distribución de pesos
                    en diferentes regiones del frame (mitades y tercios), métricas
                    de balance horizontal y vertical normalizadas.

        Notes:
            Todas las puntuaciones se normalizan en el rango [0.0, 1.0] donde valores
            más altos indican mejor adherencia al principio compositivo correspondiente.
            El frame se convierte a escala de grises internamente para todos los análisis.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rule_of_thirds = self._analyze_rule_of_thirds(gray, h, w)
        symmetry = self._analyze_symmetry(gray)
        lines_analysis = self._analyze_lines(gray)
        balance = self._analyze_balance(gray, h, w)

        self.all_rule_of_thirds_scores.append(rule_of_thirds["score"])
        self.all_symmetry_scores.append(symmetry["vertical_symmetry"])
        self.all_balance_scores.append(balance["horizontal_balance"])
        self.all_lines_counts.append(lines_analysis["num_lines"])

        return {
            "rule_of_thirds": rule_of_thirds,
            "symmetry": symmetry,
            "lines": lines_analysis,
            "balance": balance
        }

    def get_composition_summary(self) -> Dict:
        """
        Genera resumen estadístico de las métricas de composición acumuladas.

        Calcula promedios de todas las métricas compositivas recopiladas durante
        el análisis de la secuencia completa, proporcionando una evaluación global
        de la calidad compositiva del video que resume tendencias y características
        predominantes.

        Returns:
            Diccionario con estadísticas resumidas:
                total_analyzed (int): Número total de frames analizados desde la
                    inicialización o último reset.
                avg_rule_of_thirds (float): Puntuación promedio de adherencia a
                    regla de tercios en escala [0.0-1.0].
                avg_symmetry (float): Simetría promedio en escala [0.0-1.0] donde
                    1.0 indica simetría perfecta.
                avg_balance (float): Balance visual promedio en escala [0.0-1.0]
                    donde 1.0 indica distribución perfectamente equilibrada.
                avg_lines (float): Promedio de líneas detectadas por frame, útil
                    para evaluar complejidad geométrica del video.

        Notes:
            Si no se han analizado frames, retorna valores por defecto en cero.
            Los promedios se calculan como media aritmética simple de los valores
            acumulados.
        """
        if not self.all_rule_of_thirds_scores:
            return {
                "total_analyzed": 0,
                "avg_rule_of_thirds": 0.0,
                "avg_symmetry": 0.0,
                "avg_balance": 0.0,
                "avg_lines": 0.0
            }

        return {
            "total_analyzed": len(self.all_rule_of_thirds_scores),
            "avg_rule_of_thirds": float(np.mean(self.all_rule_of_thirds_scores)),
            "avg_symmetry": float(np.mean(self.all_symmetry_scores)),
            "avg_balance": float(np.mean(self.all_balance_scores)),
            "avg_lines": float(np.mean(self.all_lines_counts))
        }

    def get_composition_data(self) -> Dict:
        """
        Obtiene los datos completos de composición para visualización temporal.

        Retorna todas las métricas acumuladas frame por frame como listas ordenadas
        temporalmente, permitiendo generar gráficos de evolución de características
        compositivas a lo largo del video completo.

        Returns:
            Diccionario con arrays de métricas temporales:
                rule_of_thirds_scores (List[float]): Puntuaciones de regla de tercios
                    por frame en orden temporal.
                symmetry_scores (List[float]): Puntuaciones de simetría vertical por
                    frame en orden temporal.
                balance_scores (List[float]): Puntuaciones de balance horizontal por
                    frame en orden temporal.
                lines_count (List[int]): Conteo de líneas detectadas por frame en
                    orden temporal.

        Notes:
            Los datos se convierten explícitamente a tipos nativos de Python (float, int)
            para garantizar serialización JSON correcta. Útil para generar visualizaciones
            interactivas en frontend o análisis estadísticos post-procesamiento.
        """
        return {
            "rule_of_thirds_scores": [float(s) for s in self.all_rule_of_thirds_scores],
            "symmetry_scores": [float(s) for s in self.all_symmetry_scores],
            "balance_scores": [float(s) for s in self.all_balance_scores],
            "lines_count": [int(s) for s in self.all_lines_counts]
        }

    def reset(self):
        """
        Reinicia los acumuladores de métricas de composición.

        Limpia todas las listas de métricas acumuladas estableciéndolas como listas
        vacías. Útil para comenzar el análisis de un nuevo video sin necesidad de
        crear una nueva instancia del analizador, manteniendo la configuración.

        Notes:
            Después de llamar a reset(), el próximo frame procesado iniciará una
            nueva secuencia de acumulación desde cero. No afecta configuración
            interna de algoritmos.
        """
        self.all_rule_of_thirds_scores = []
        self.all_symmetry_scores = []
        self.all_balance_scores = []
        self.all_lines_counts = []

    def _analyze_rule_of_thirds(self, gray: np.ndarray, h: int, w: int) -> Dict:
        """
        Evalúa el cumplimiento de la regla de tercios compositiva.

        Divide el frame en una cuadrícula de 3x3 secciones y evalúa la densidad
        de bordes mediante detector Canny en cada región, con especial énfasis en
        los cuatro puntos de intersección (puntos fuertes) donde idealmente se
        deben ubicar elementos de interés visual según principios cinematográficos.

        Args:
            gray: Frame en escala de grises como numpy array 2D.
            h: Altura del frame en píxeles.
            w: Ancho del frame en píxeles.

        Returns:
            Diccionario con análisis detallado de regla de tercios:
                score (float): Puntuación global normalizada [0.0-1.0] basada en
                    densidad promedio de bordes en los cuatro puntos de intersección.
                    Valores altos indican mayor concentración de interés visual en
                    puntos óptimos.
                sections (List[Dict]): Lista de 9 diccionarios (uno por sección) con
                    densidad de bordes y posición (i,j) en cuadrícula donde (0,0) es
                    esquina superior izquierda.
                intersection_densities (List[float]): Densidades específicas en los
                    cuatro puntos fuertes de intersección (centro de regiones de
                    30px de radio alrededor de cada intersección).
                follows_rule (bool): Bandera que indica si cumple la regla según
                    umbral establecido (score > 0.15).

        Notes:
            Detector de bordes: Canny con umbrales 50-150.
            Radio de evaluación en intersecciones: 30 píxeles.
            Umbral de cumplimiento: score > 0.15 (15% de densidad promedio).
            Las secciones se indexan de [0,0] a [2,2] en orden fila-columna.
        """
        edges = cv2.Canny(gray, 50, 150)

        h_third = h // 3
        w_third = w // 3

        sections = []
        for i in range(3):
            for j in range(3):
                section = edges[i * h_third:(i + 1) * h_third, j * w_third:(j + 1) * w_third]
                density = np.sum(section > 0) / section.size
                sections.append({
                    "row": i,
                    "col": j,
                    "density": float(density)
                })

        intersection_points = [
            (w_third, h_third), (2 * w_third, h_third),
            (w_third, 2 * h_third), (2 * w_third, 2 * h_third)
        ]

        intersection_densities = []
        radius = 30
        for px, py in intersection_points:
            x1 = max(0, px - radius)
            x2 = min(w, px + radius)
            y1 = max(0, py - radius)
            y2 = min(h, py + radius)

            region = edges[y1:y2, x1:x2]
            if region.size > 0:
                density = np.sum(region > 0) / region.size
                intersection_densities.append(float(density))
            else:
                intersection_densities.append(0.0)

        score = np.mean(intersection_densities) if intersection_densities else 0.0

        return {
            "score": float(score),
            "sections": sections,
            "intersection_densities": intersection_densities,
            "follows_rule": score > 0.15
        }

    def _analyze_symmetry(self, gray: np.ndarray) -> Dict:
        """
        Analiza la simetría del frame mediante comparación de regiones reflejadas.

        Calcula métricas de simetría tanto vertical (espejo izquierda-derecha) como
        horizontal (espejo arriba-abajo) mediante error absoluto medio normalizado
        entre cada mitad y su reflexión especular.

        Args:
            gray: Frame en escala de grises como numpy array 2D.

        Returns:
            Diccionario con métricas de simetría:
                vertical_symmetry (float): Puntuación de simetría vertical [0.0-1.0]
                    donde 1.0 indica simetría perfecta entre mitades izquierda y
                    derecha. Calculada como 1 - (MAE / 255).
                horizontal_symmetry (float): Puntuación de simetría horizontal [0.0-1.0]
                    donde 1.0 indica simetría perfecta entre mitades superior e inferior.
                is_symmetric_vertical (bool): Bandera que indica simetría vertical
                    significativa (puntuación > 0.8).
                is_symmetric_horizontal (bool): Bandera que indica simetría horizontal
                    significativa (puntuación > 0.8).

        Notes:
            Fórmula de simetría: 1 - (MAE / 255)
            donde MAE = Mean Absolute Error entre mitad y su reflexión
            Umbral de simetría significativa: 0.8 (80% de similitud)

            Para frames con dimensiones impares, se descarta la columna/fila central
            en el análisis de simetría respectivo.
        """
        h, w = gray.shape

        left_half = gray[:, :w // 2]
        right_half = gray[:, w // 2:]
        right_half_flipped = np.fliplr(right_half)

        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        vertical_diff = np.mean(np.abs(
            left_half[:, :min_width].astype(float) -
            right_half_flipped[:, :min_width].astype(float)
        ))

        top_half = gray[:h // 2, :]
        bottom_half = gray[h // 2:, :]
        bottom_half_flipped = np.flipud(bottom_half)

        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        horizontal_diff = np.mean(np.abs(
            top_half[:min_height, :].astype(float) -
            bottom_half_flipped[:min_height, :].astype(float)
        ))

        vertical_symmetry = 1 - (vertical_diff / 255)
        horizontal_symmetry = 1 - (horizontal_diff / 255)

        return {
            "vertical_symmetry": float(vertical_symmetry),
            "horizontal_symmetry": float(horizontal_symmetry),
            "is_symmetric_vertical": vertical_symmetry > 0.8,
            "is_symmetric_horizontal": horizontal_symmetry > 0.8
        }

    def _analyze_lines(self, gray: np.ndarray) -> Dict:
        """
        Detecta y clasifica líneas dominantes mediante Hough Transform.

        Aplica detector de bordes Canny seguido de Transformada de Hough probabilística
        para detectar segmentos de línea recta en el frame. Clasifica las líneas según
        su orientación angular en tres categorías: horizontales, verticales y diagonales.

        Args:
            gray: Frame en escala de grises como numpy array 2D.

        Returns:
            Diccionario con análisis de líneas detectadas:
                num_lines (int): Número total de líneas detectadas.
                has_diagonals (bool): Indica presencia significativa de líneas diagonales
                    (más del 30% del total).
                has_horizontals (bool): Indica presencia significativa de líneas
                    horizontales (más del 30% del total).
                has_verticals (bool): Indica presencia significativa de líneas verticales
                    (más del 30% del total).
                dominant_direction (str): Dirección predominante entre "horizontal",
                    "vertical", "diagonal" o "none" si no hay líneas.
                line_counts (Dict): Conteo específico por cada tipo de línea con claves
                    "horizontal", "vertical" y "diagonal".

        Notes:
            Parámetros Canny: umbrales 50-150, aperture size 3.
            Parámetros Hough Transform:
                - rho: 1 píxel de resolución en espacio de parámetros
                - theta: 1 grado de resolución angular
                - threshold: 50 votos mínimos
                - minLineLength: 50 píxeles
                - maxLineGap: 10 píxeles de gap máximo entre segmentos

            Criterios de clasificación por ángulo:
                - Horizontal: ángulo < 15° o > 165°
                - Vertical: 75° < ángulo < 105°
                - Diagonal: todos los demás ángulos

            Una línea se considera "predominante" en su categoría si representa más
            del 30% del total de líneas detectadas.
        """
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        if lines is None:
            return {
                "num_lines": 0,
                "has_diagonals": False,
                "has_horizontals": False,
                "has_verticals": False,
                "dominant_direction": "none"
            }

        horizontals = 0
        verticals = 0
        diagonals = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 15 or angle > 165:
                horizontals += 1
            elif 75 < angle < 105:
                verticals += 1
            else:
                diagonals += 1

        total = len(lines)
        dominant = max([
            ("horizontal", horizontals),
            ("vertical", verticals),
            ("diagonal", diagonals)
        ], key=lambda x: x[1])

        return {
            "num_lines": int(total),
            "has_diagonals": diagonals > total * 0.3,
            "has_horizontals": horizontals > total * 0.3,
            "has_verticals": verticals > total * 0.3,
            "dominant_direction": dominant[0],
            "line_counts": {
                "horizontal": int(horizontals),
                "vertical": int(verticals),
                "diagonal": int(diagonals)
            }
        }

    def _analyze_balance(self, gray: np.ndarray, h: int, w: int) -> Dict:
        """
        Analiza el balance visual mediante distribución de intensidades.

        Evalúa cómo se distribuye el peso visual (suma de valores de intensidad
        de píxeles) a lo largo del frame, tanto en eje horizontal (izquierda-derecha)
        como vertical (división en tercios superior-medio-inferior). Un buen balance
        compositivo distribuye el interés visual de manera equilibrada o con
        asimetría intencional controlada.

        Args:
            gray: Frame en escala de grises como numpy array 2D.
            h: Altura del frame en píxeles.
            w: Ancho del frame en píxeles.

        Returns:
            Diccionario con análisis de balance visual:
                horizontal_balance (float): Ratio de balance entre mitades izquierda
                    y derecha [0.0-1.0] donde 1.0 indica distribución perfectamente
                    equilibrada. Calculado como min(peso_izq, peso_der) / max(...).
                vertical_balance (float): Métrica de balance entre tercios verticales
                    [0.0-1.0] donde valores altos indican mejor distribución uniforme.
                    Calculado invirtiendo coeficiente de variación para que valores
                    altos representen mejor balance.
                is_balanced (bool): Indica si hay balance horizontal significativo
                    según umbral establecido (ratio > 0.7).
                weight_distribution (Dict): Distribución detallada y normalizada de
                    pesos visuales por región con claves:
                        - left, right: pesos relativos de mitades horizontales [0.0-1.0]
                        - top, middle, bottom: pesos absolutos de tercios verticales

        Notes:
            Fórmula de balance horizontal: min(L, R) / max(L, R)
            donde L = suma de intensidades mitad izquierda, R = mitad derecha

            Fórmula de balance vertical: 1 / (1 + CV)
            donde CV = desviación estándar / media de pesos de tercios verticales

            Umbral de balance horizontal significativo: 0.7 (70% de equilibrio)

            El "peso visual" se define como la suma de todos los valores de intensidad
            de píxeles en una región, asumiendo que áreas más brillantes tienen mayor
            peso visual perceptual.
        """
        left_half = gray[:, :w // 2]
        right_half = gray[:, w // 2:]

        left_weight = np.sum(left_half)
        right_weight = np.sum(right_half)
        total_weight = left_weight + right_weight

        horizontal_balance = min(left_weight, right_weight) / max(left_weight, right_weight)

        top_third = gray[:h // 3, :]
        middle_third = gray[h // 3:2 * h // 3, :]
        bottom_third = gray[2 * h // 3:, :]

        top_weight = np.sum(top_third)
        middle_weight = np.sum(middle_third)
        bottom_weight = np.sum(bottom_third)

        vertical_balance = np.std([top_weight, middle_weight, bottom_weight]) / np.mean(
            [top_weight, middle_weight, bottom_weight])

        return {
            "horizontal_balance": float(horizontal_balance),
            "vertical_balance": float(1 / (1 + vertical_balance)),
            "is_balanced": horizontal_balance > 0.7,
            "weight_distribution": {
                "left": float(left_weight / total_weight),
                "right": float(right_weight / total_weight),
                "top": float(top_weight),
                "middle": float(middle_weight),
                "bottom": float(bottom_weight)
            }
        }


def visualize_composition(frame: np.ndarray, show_grid: bool = True,
                          show_points: bool = True) -> np.ndarray:
    """
    Dibuja guías de composición visual sobre el frame.

    Superpone elementos gráficos para visualizar la cuadrícula de regla de tercios
    y los puntos fuertes de intersección, facilitando la evaluación visual directa
    de la composición cinematográfica del frame. Útil para análisis manual y
    verificación de resultados automáticos.

    Args:
        frame: Frame sobre el cual dibujar las guías en formato BGR (OpenCV estándar)
            como numpy array con dimensiones (H, W, 3).
        show_grid: Si True, dibuja la cuadrícula completa de tercios con líneas
            horizontales y verticales. Por defecto True.
        show_points: Si True, dibuja círculos en los cuatro puntos de intersección
            de la cuadrícula (puntos fuertes compositivos). Por defecto True.

    Returns:
        Frame modificado con overlay de guías compositivas aplicado mediante alpha
        blending. El array original no se modifica, se retorna una copia con las
        guías superpuestas con transparencia del 30%.

    Notes:
        Elementos visuales y configuración:
            - Cuadrícula de tercios:
                * Color: amarillo (255, 255, 0) en espacio BGR
                * Grosor: 1 píxel
                * Posiciones: líneas en 1/3 y 2/3 de ancho y alto

            - Puntos de intersección (4 puntos):
                * Círculo interior: cian (0, 255, 255) relleno, radio 8px
                * Círculo exterior: blanco (255, 255, 255) borde, radio 10px, grosor 2px
                * Posiciones: intersecciones de líneas de tercios

            - Transparencia del overlay: 30% guías, 70% frame original

        Las guías se posicionan según la regla de tercios clásica, dividiendo el
        frame en nueve regiones iguales. Los puntos de intersección marcan las
        ubicaciones óptimas para elementos de interés visual según teoría compositiva.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if show_grid:
        h_third = h // 3
        w_third = w // 3
        color = (255, 255, 0)
        thickness = 1

        cv2.line(overlay, (0, h_third), (w, h_third), color, thickness)
        cv2.line(overlay, (0, 2 * h_third), (w, 2 * h_third), color, thickness)
        cv2.line(overlay, (w_third, 0), (w_third, h), color, thickness)
        cv2.line(overlay, (2 * w_third, 0), (2 * w_third, h), color, thickness)

    if show_points:
        h_third = h // 3
        w_third = w // 3

        points = [
            (w_third, h_third), (2 * w_third, h_third),
            (w_third, 2 * h_third), (2 * w_third, 2 * h_third)
        ]

        for px, py in points:
            cv2.circle(overlay, (px, py), 8, (0, 255, 255), -1)
            cv2.circle(overlay, (px, py), 10, (255, 255, 255), 2)

    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return result