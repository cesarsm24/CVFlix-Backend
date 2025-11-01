"""
lighting.py

Análisis de iluminación cinematográfica mediante evaluación de histogramas,
contraste y gradientes espaciales. Clasifica tipo de iluminación, exposición
y dirección de luz dominante en frames de vídeo.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Análisis de histogramas y gradientes
    - numpy: Cálculos estadísticos
    - app.config: Umbrales de iluminación

Usage:
    from app.analysis.lighting import LightingAnalyzer

    analyzer = LightingAnalyzer()
    analysis = analyzer.analyze(frame)

    print(f"Tipo: {analysis['type']}")
    print(f"Brillo: {analysis['brightness']}")
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

from app.config import LIGHTING_CONFIG


class LightingAnalyzer:
    """
    Analizador de iluminación cinematográfica.

    Evalúa características de iluminación mediante análisis de histogramas,
    métricas estadísticas y gradientes espaciales para clasificar el tipo
    de iluminación, estado de exposición y dirección de luz dominante.

    Analiza:
        - Tipo de iluminación (High Key, Low Key, Normal)
        - Brillo promedio
        - Contraste
        - Exposición (subexpuesta, normal, sobreexpuesta)
        - Dirección de luz
        - Distribución de histograma

    Attributes:
        config: Configuración de iluminación desde config.py
    """

    def __init__(self):
        """Inicializa el analizador de iluminación."""
        self.config = LIGHTING_CONFIG

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Realiza análisis completo de iluminación.

        Args:
            frame: Frame BGR a analizar

        Returns:
            Diccionario con análisis completo:
                type: Tipo de iluminación ("High Key", "Low Key", "Normal")
                brightness: Brillo promedio 0-255
                contrast: Contraste 0-255
                exposure: Estado de exposición ("underexposed", "normal", "overexposed")
                light_direction: Dirección dominante de luz ("top", "bottom",
                    "left", "right", "center")
                histogram: Datos del histograma
                dynamic_range: Rango dinámico 0-255

        Notes:
            Análisis basado en conversión a escala de grises. Los umbrales son
            configurables mediante config.py. Retorna valores por defecto en
            caso de error.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = self._calculate_brightness(gray)
        contrast = self._calculate_contrast(gray)
        histogram = self._calculate_histogram(gray)
        dynamic_range = self._calculate_dynamic_range(histogram)

        lighting_type = self._classify_lighting_type(brightness, contrast)
        exposure = self._classify_exposure(brightness)
        light_direction = self._detect_light_direction(gray)

        return {
            "type": lighting_type,
            "brightness": brightness,
            "contrast": contrast,
            "exposure": exposure,
            "light_direction": light_direction,
            "histogram": histogram,
            "dynamic_range": dynamic_range
        }

    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """
        Calcula brillo promedio del frame.

        Args:
            gray: Frame en escala de grises

        Returns:
            Brillo promedio en rango 0-255

        Notes:
            Utiliza valor medio de todos los píxeles:
                - 0 = completamente negro
                - 255 = completamente blanco
                - ~128 = brillo medio
        """
        return float(np.mean(gray))

    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """
        Calcula contraste del frame.

        Args:
            gray: Frame en escala de grises

        Returns:
            Contraste en rango 0-255

        Notes:
            Utiliza desviación estándar como medida de contraste.
            Alto contraste indica gran diferencia entre luces y sombras.
            Bajo contraste indica imagen plana.
        """
        return float(np.std(gray))

    def _calculate_histogram(self, gray: np.ndarray) -> Dict[str, Any]:
        """
        Calcula histograma de luminosidad.

        Args:
            gray: Frame en escala de grises

        Returns:
            Diccionario con datos del histograma:
                values: Array de 256 valores (conteo por nivel)
                peaks: Picos del histograma
                distribution: Distribución (shadows, midtones, highlights)

        Notes:
            Histograma de 256 bins (0-255). Detecta clipping en sombras/luces
            y calcula distribución en tres zonas tonales:
                - Shadows: 0-85 (33% inferior)
                - Midtones: 85-170 (33% medio)
                - Highlights: 170-255 (33% superior)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        peaks = []
        for i in range(1, 255):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                if hist[i] > np.max(hist) * 0.1:
                    peaks.append(i)

        total_pixels = gray.size
        shadows = np.sum(hist[:85]) / total_pixels * 100
        midtones = np.sum(hist[85:170]) / total_pixels * 100
        highlights = np.sum(hist[170:]) / total_pixels * 100

        return {
            "values": hist.tolist(),
            "peaks": peaks,
            "distribution": {
                "shadows": round(shadows, 1),
                "midtones": round(midtones, 1),
                "highlights": round(highlights, 1)
            }
        }

    def _calculate_dynamic_range(self, histogram: Dict[str, Any]) -> float:
        """
        Calcula rango dinámico del frame.

        Mide la diferencia entre el punto más oscuro y más claro con presencia
        significativa de píxeles.

        Args:
            histogram: Datos del histograma

        Returns:
            Rango dinámico en escala 0-255

        Notes:
            Alto rango dinámico indica gran variedad tonal.
            Bajo rango dinámico indica imagen con tonos comprimidos.
            El umbral de significancia se establece en 1% del pico máximo.
        """
        hist_values = np.array(histogram["values"])

        threshold = np.max(hist_values) * 0.01

        min_val = 0
        for i in range(256):
            if hist_values[i] > threshold:
                min_val = i
                break

        max_val = 255
        for i in range(255, -1, -1):
            if hist_values[i] > threshold:
                max_val = i
                break

        return float(max_val - min_val)

    def _classify_lighting_type(self, brightness: float, contrast: float) -> str:
        """
        Clasifica el tipo de iluminación cinematográfica.

        Args:
            brightness: Brillo promedio
            contrast: Contraste

        Returns:
            Tipo de iluminación:
                "High Key": Iluminación brillante con bajo contraste
                "Low Key": Iluminación oscura con alto contraste
                "Normal": Iluminación equilibrada

        Notes:
            High Key: Común en comedias y publicidad. Caracterizado por tonos
                brillantes, bajo contraste y pocas sombras profundas.
            Low Key: Común en thrillers y cine noir. Caracterizado por tonos
                oscuros, alto contraste y sombras dramáticas.
            Los umbrales son configurables mediante config.py.
        """
        high_key_config = self.config.get("high_key", {})
        low_key_config = self.config.get("low_key", {})

        brightness_high = high_key_config.get("brightness_threshold", 160)
        contrast_max = high_key_config.get("contrast_max", 50)

        brightness_low = low_key_config.get("brightness_threshold", 80)
        contrast_min = low_key_config.get("contrast_min", 60)

        if brightness >= brightness_high and contrast < contrast_max:
            return "High Key"
        elif brightness <= brightness_low and contrast > contrast_min:
            return "Low Key"
        else:
            return "Normal"

    def _classify_exposure(self, brightness: float) -> str:
        """
        Clasifica el estado de exposición.

        Args:
            brightness: Brillo promedio

        Returns:
            Estado de exposición:
                "underexposed": Subexpuesta (muy oscura)
                "normal": Exposición correcta
                "overexposed": Sobreexpuesta (muy brillante)

        Notes:
            Clasificación basada únicamente en brillo promedio.
            Los umbrales son configurables mediante config.py.
        """
        exposure_config = self.config.get("exposure", {})

        underexposed_threshold = exposure_config.get("underexposed", 70)
        overexposed_threshold = exposure_config.get("overexposed", 180)

        if brightness < underexposed_threshold:
            return "underexposed"
        elif brightness > overexposed_threshold:
            return "overexposed"
        else:
            return "normal"

    def _detect_light_direction(self, gray: np.ndarray) -> str:
        """
        Detecta dirección dominante de luz mediante análisis de gradientes de Sobel.

        Método que analiza los cambios de intensidad en lugar de
        solo comparar promedios de regiones, reduciendo falsos positivos causados
        por objetos brillantes.
        """
        # Calcular gradientes con Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        # Promediar gradientes (positivo = luz desde ese lado)
        mean_grad_x = np.mean(grad_x)  # + = luz desde derecha, - = desde izquierda
        mean_grad_y = np.mean(grad_y)  # + = luz desde abajo, - = desde arriba

        # Umbral de sensibilidad (configurable)
        threshold = self.config.get("light_direction", {}).get("threshold", 5.0)

        # Determinar dirección dominante
        abs_x = abs(mean_grad_x)
        abs_y = abs(mean_grad_y)

        # Si los gradientes son muy pequeños, luz difusa/centrada
        if abs_x < threshold and abs_y < threshold:
            return "center"

        # Comparar ejes para determinar dirección dominante
        if abs_x > abs_y:
            # Luz lateral dominante
            return "right" if mean_grad_x > 0 else "left"
        else:
            # Luz vertical dominante
            return "bottom" if mean_grad_y > 0 else "top"


def visualize_lighting(
    frame: np.ndarray,
    lighting: Dict[str, Any]
) -> np.ndarray:
    """
    Visualiza información de iluminación sobre el frame.

    Dibuja indicador visual con información del tipo de iluminación, brillo
    y exposición en la esquina superior del frame.

    Args:
        frame: Frame BGR
        lighting: Diccionario con análisis de iluminación

    Returns:
        Frame con indicador de iluminación superpuesto

    Notes:
        Elementos visuales:
            - Rectángulo semi-transparente como fondo
            - Texto con tipo de iluminación
            - Valores de brillo y contraste
            - Color codificado según exposición:
                * Verde: exposición normal
                * Amarillo: subexpuesta
                * Rojo: sobreexpuesta
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    lighting_type = lighting['type']
    brightness = lighting['brightness']
    exposure = lighting['exposure']

    color_map = {
        'normal': (0, 255, 0),
        'underexposed': (0, 255, 255),
        'overexposed': (0, 0, 255)
    }
    color = color_map.get(exposure, (255, 255, 255))

    info_lines = [
        f"Lighting: {lighting_type}",
        f"Brightness: {brightness:.1f}",
        f"Exposure: {exposure}"
    ]

    overlay = frame_copy.copy()
    y_offset = 10
    for line in info_lines:
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        cv2.rectangle(
            overlay,
            (10, y_offset),
            (10 + text_size[0] + 10, y_offset + text_size[1] + 10),
            (0, 0, 0),
            -1
        )
        y_offset += text_size[1] + 20

    cv2.addWeighted(overlay, 0.7, frame_copy, 0.3, 0, frame_copy)

    y_offset = 30
    for line in info_lines:
        cv2.putText(
            frame_copy,
            line,
            (15, y_offset),
            font,
            font_scale,
            color,
            thickness
        )
        y_offset += 30

    return frame_copy