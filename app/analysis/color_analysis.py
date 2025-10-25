"""
color_analyzer.py

Módulo para análisis de colores dominantes, temperatura cromática y esquemas
de color en frames de video mediante clustering K-means y análisis HSV.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter
import webcolors
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ColorAnalyzer:
    """
    Analizador de colores en frames de video.

    Realiza análisis cromático mediante clustering K-means para extraer colores
    dominantes, evalúa temperatura de color basándose en balance RGB, clasifica
    esquemas cromáticos mediante análisis HSV, y genera histogramas RGB acumulados
    para visualización estadística.

    Attributes:
        hist_r_accumulated (np.ndarray): Array acumulador de histograma del canal rojo
            con 256 bins correspondientes a niveles de intensidad [0-255].
        hist_g_accumulated (np.ndarray): Array acumulador de histograma del canal verde.
        hist_b_accumulated (np.ndarray): Array acumulador de histograma del canal azul.
        frames_count (int): Contador de frames procesados para normalización de
            histogramas acumulados.
    """

    def __init__(self):
        """
        Inicializa el analizador de colores.

        Configura acumuladores de histograma para los tres canales RGB con 256 bins
        cada uno, inicializados a cero. Establece contador de frames en cero.
        """
        self.hist_r_accumulated = np.zeros(256)
        self.hist_g_accumulated = np.zeros(256)
        self.hist_b_accumulated = np.zeros(256)
        self.frames_count = 0

    def analyze_colors(self, frame: np.ndarray, n_colors: int = 5) -> Dict:
        """
        Analiza los colores dominantes de un frame mediante clustering.

        Ejecuta análisis cromático completo que incluye extracción de colores
        dominantes mediante K-means, evaluación de temperatura de color, y
        clasificación de esquema cromático. Redimensiona el frame para optimizar
        rendimiento del clustering.

        Args:
            frame: Frame a analizar en formato BGR (OpenCV estándar) como numpy
                array con dimensiones (H, W, 3).
            n_colors: Número de colores dominantes a extraer mediante K-means
                clustering. Por defecto 5 colores.

        Returns:
            Diccionario con análisis completo de colores:
                dominant_colors (List[Dict]): Lista de colores dominantes ordenados
                    por frecuencia de aparición, donde cada elemento contiene RGB,
                    hex, porcentaje y nombre del color.
                temperature (Dict): Análisis de temperatura cromática con label
                    descriptivo y valor numérico normalizado.
                color_scheme (Dict): Clasificación del esquema cromático con nombre,
                    descripción y diferencia angular máxima de matices.

        Notes:
            El frame se redimensiona a 150x150 píxeles antes del análisis para
            acelerar el clustering K-means sin pérdida significativa de precisión
            en la detección de colores dominantes.
        """
        frame_small = cv2.resize(frame, (150, 150))
        frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        dominant_colors = self._extract_dominant_colors(frame_rgb, n_colors)
        temperature = self._analyze_temperature(dominant_colors)
        color_scheme = self._analyze_color_scheme(dominant_colors)

        return {
            "dominant_colors": dominant_colors,
            "temperature": temperature,
            "color_scheme": color_scheme
        }

    def _extract_dominant_colors(self, frame_rgb: np.ndarray, n_colors: int) -> List[Dict]:
        """
        Extrae los colores dominantes usando K-means clustering.

        Aplica algoritmo K-means sobre el espacio de píxeles RGB para identificar
        clusters de colores similares. Ordena los colores por frecuencia de aparición
        y los convierte a múltiples representaciones (RGB, hex, nombre CSS3).

        Args:
            frame_rgb: Frame en formato RGB como numpy array con dimensiones (H, W, 3).
            n_colors: Número de clusters K para el algoritmo K-means, determina cuántos
                colores dominantes se extraerán.

        Returns:
            Lista de diccionarios ordenada por frecuencia descendente, donde cada
            elemento contiene:
                rgb (List[int]): Valores RGB como lista [R, G, B] en rango [0, 255].
                hex (str): Representación hexadecimal del color en formato "#RRGGBB".
                percentage (float): Porcentaje de píxeles del frame que pertenecen
                    a este cluster, redondeado a 2 decimales.
                name (str): Nombre del color más cercano según especificación CSS3,
                    capitalizado.

        Notes:
            Configuración K-means:
                - Criterio de parada: 100 iteraciones máximo o epsilon 0.2
                - Inicialización: KMEANS_PP_CENTERS (K-means++)
                - Repeticiones: 10 intentos para encontrar mejor clustering
        """
        pixels = frame_rgb.reshape(-1, 3)
        pixels_float = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels_float, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(labels.flatten())
        total_pixels = len(labels)
        indices = np.argsort(counts)[::-1]

        dominant_colors = []
        for i in indices:
            color_rgb = centers[i].astype(int)
            percentage = (counts[i] / total_pixels) * 100
            hex_color = "#{:02x}{:02x}{:02x}".format(*color_rgb)
            color_name = self._get_color_name(color_rgb)

            dominant_colors.append({
                "rgb": color_rgb.tolist(),
                "hex": hex_color,
                "percentage": round(percentage, 2),
                "name": color_name
            })

        return dominant_colors

    def _get_color_name(self, rgb: np.ndarray) -> str:
        """
        Obtiene el nombre del color más cercano según especificación CSS3.

        Intenta primero un match exacto con colores CSS3 conocidos. Si no existe
        coincidencia exacta, calcula la distancia euclidiana en espacio RGB con
        todos los colores CSS3 y retorna el más cercano.

        Args:
            rgb: Array numpy con valores RGB en formato [R, G, B] donde cada
                componente está en rango [0, 255].

        Returns:
            Nombre del color en formato capitalizado (primera letra mayúscula).
            Retorna "Unknown" si no se puede determinar el nombre.

        Notes:
            Utiliza distancia euclidiana en espacio RGB:
                distance = sqrt((R1-R2)² + (G1-G2)² + (B1-B2)²)
            Compatible con webcolors versión 24.x mediante uso de names() function.
        """
        try:
            color_name = webcolors.rgb_to_name(tuple(rgb), spec='css3')
        except ValueError:
            min_distance = float('inf')
            closest_name = "Unknown"
            css3_names = webcolors.names('css3')

            for name in css3_names:
                try:
                    ref_rgb = webcolors.name_to_rgb(name)
                    distance = np.sqrt(np.sum((np.array(rgb) - np.array(ref_rgb)) ** 2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_name = name
                except Exception:
                    continue

            color_name = closest_name

        return color_name.capitalize()

    def _analyze_temperature(self, colors: List[Dict]) -> Dict:
        """
        Analiza la temperatura de color del frame mediante balance rojo-azul.

        Calcula un score de temperatura cromática basándose en la diferencia entre
        componentes rojas y azules de los colores dominantes, ponderado por su
        frecuencia de aparición. Clasifica el resultado en cinco categorías de
        temperatura.

        Args:
            colors: Lista de colores dominantes con información RGB y porcentaje,
                típicamente el resultado de _extract_dominant_colors().

        Returns:
            Diccionario con análisis de temperatura:
                label (str): Clasificación textual de temperatura. Valores posibles:
                    "Muy Cálido", "Cálido", "Neutral", "Frío", "Muy Frío".
                value (float): Score numérico de temperatura en rango aproximado
                    [-1.0, 1.0] donde valores positivos indican calidez (tonos rojos)
                    y negativos frialdad (tonos azules). Redondeado a 3 decimales.

        Notes:
            Fórmula de temperatura por color:
                temp_score = (R - B) / 255.0
            Temperatura promedio ponderada:
                avg_temp = sum(temp_score * percentage) / sum(percentage)

            Umbrales de clasificación:
                - avg_temp > 0.2: Muy Cálido
                - avg_temp > 0.1: Cálido
                - avg_temp > -0.1: Neutral
                - avg_temp > -0.2: Frío
                - avg_temp <= -0.2: Muy Frío
        """
        if not colors:
            return {"label": "Neutral", "value": 0}

        total_weight = 0
        temp_score = 0

        for color in colors:
            r, g, b = color["rgb"]
            weight = color["percentage"]
            color_temp = (r - b) / 255.0
            temp_score += color_temp * weight
            total_weight += weight

        if total_weight > 0:
            avg_temp = temp_score / total_weight
        else:
            avg_temp = 0

        if avg_temp > 0.2:
            label = "Muy Cálido"
        elif avg_temp > 0.1:
            label = "Cálido"
        elif avg_temp > -0.1:
            label = "Neutral"
        elif avg_temp > -0.2:
            label = "Frío"
        else:
            label = "Muy Frío"

        return {
            "label": label,
            "value": round(avg_temp, 3)
        }

    def _analyze_color_scheme(self, colors: List[Dict]) -> Dict:
        """
        Analiza el esquema cromático del frame mediante diferencias de matiz HSV.

        Convierte colores dominantes a espacio HSV para calcular diferencias angulares
        de matiz. Clasifica el esquema cromático en categorías estándar de teoría del
        color basándose en la máxima diferencia angular encontrada.

        Args:
            colors: Lista de colores dominantes con información RGB, típicamente
                el resultado de _extract_dominant_colors().

        Returns:
            Diccionario con análisis de esquema cromático:
                scheme (str): Clasificación del esquema. Valores posibles:
                    "Monocromático", "Análogo", "Complementario", "Triádico",
                    "Policromático".
                description (str): Descripción textual del esquema cromático.
                max_hue_difference (float): Diferencia angular máxima de matiz
                    encontrada entre colores, en grados [0-180], redondeada a
                    1 decimal.

        Notes:
            Umbrales de clasificación basados en diferencia angular de matiz:
                - < 15°: Monocromático (variaciones de un mismo color)
                - < 30°: Análogo (colores adyacentes en círculo cromático)
                - 60-120°: Complementario (colores opuestos)
                - 40-60°: Triádico (tres colores equidistantes)
                - Otros: Policromático (múltiples colores variados)

            En OpenCV, el canal Hue está en rango [0, 180] en lugar de [0, 360].
            Las diferencias angulares se ajustan para considerar la naturaleza
            circular del espacio de matices.
        """
        if len(colors) < 2:
            return {"scheme": "Monocromático", "description": "Un solo color dominante"}

        hsv_colors = []
        for color in colors:
            rgb = np.uint8([[color["rgb"]]])
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0][0]
            hsv_colors.append(hsv)

        hues = [int(hsv[0]) for hsv in hsv_colors]

        max_diff = 0
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                if diff > 90:
                    diff = 180 - diff
                max_diff = max(max_diff, diff)

        if max_diff < 15:
            scheme = "Monocromático"
            description = "Variaciones de un mismo color"
        elif max_diff < 30:
            scheme = "Análogo"
            description = "Colores adyacentes en el círculo cromático"
        elif 60 < max_diff < 120:
            scheme = "Complementario"
            description = "Colores opuestos en el círculo cromático"
        elif 40 < max_diff < 60:
            scheme = "Triádico"
            description = "Tres colores equidistantes"
        else:
            scheme = "Policromático"
            description = "Múltiples colores variados"

        return {
            "scheme": scheme,
            "description": description,
            "max_hue_difference": round(float(max_diff), 1)
        }

    def accumulate_histogram(self, frame: np.ndarray):
        """
        Acumula histogramas RGB de cada frame para análisis estadístico agregado.

        Calcula histogramas de 256 bins para cada canal RGB usando OpenCV y los
        acumula en arrays globales. Mantiene contador de frames para posterior
        normalización.

        Args:
            frame: Frame en formato BGR (OpenCV estándar) como numpy array con
                dimensiones (H, W, 3).

        Notes:
            Los histogramas se calculan con:
                - 256 bins: uno por cada nivel de intensidad [0-255]
                - Rango completo: [0, 256) para cada canal
                - Sin máscara: considera todos los píxeles del frame

            Los acumuladores permiten generar histograma promedio del video completo
            mediante normalización por frames_count.
        """
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256]).flatten()
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256]).flatten()

        self.hist_b_accumulated += hist_b
        self.hist_g_accumulated += hist_g
        self.hist_r_accumulated += hist_r
        self.frames_count += 1

    def reset_histogram(self):
        """
        Reinicia acumuladores de histograma y contador de frames.

        Limpia todos los arrays acumuladores estableciéndolos a cero y resetea
        el contador de frames. Útil para comenzar análisis de un nuevo video
        sin crear nueva instancia del analizador.

        Notes:
            Después de llamar a reset_histogram(), el próximo frame procesado
            iniciará un nuevo ciclo de acumulación desde cero.
        """
        self.hist_r_accumulated = np.zeros(256)
        self.hist_g_accumulated = np.zeros(256)
        self.hist_b_accumulated = np.zeros(256)
        self.frames_count = 0