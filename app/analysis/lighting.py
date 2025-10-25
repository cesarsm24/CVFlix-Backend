"""
lighting_analyzer.py

Módulo para análisis cuantitativo de iluminación cinematográfica mediante procesamiento
de histogramas, análisis espacial y cálculo de gradientes. Clasifica tipo de iluminación,
evalúa exposición, contraste y dirección de luz principal.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class LightingType(Enum):
    """
    Enumeración de tipos de iluminación cinematográfica.

    Define categorías estándar de iluminación según teoría cinematográfica clásica,
    basadas en niveles de brillo, contraste y uniformidad de distribución espacial.
    """
    HIGH_KEY = "High Key (Luz Alta)"
    LOW_KEY = "Low Key (Luz Baja)"
    NORMAL = "Iluminación Normal"
    FLAT = "Iluminación Plana"
    DRAMATIC = "Iluminación Dramática"


class LightingAnalyzer:
    """
    Analizador cuantitativo de iluminación cinematográfica.

    Implementa conjunto de métricas para evaluación objetiva de características de
    iluminación en frames de video, incluyendo análisis de exposición mediante
    histogramas, medición de contraste por múltiples métodos, evaluación de
    distribución espacial y determinación de dirección de luz mediante gradientes.

    Notes:
        El análisis se realiza en espacio de grises para simplificar cálculos y
        centrarse en características de luminancia independientes del color. Los
        métodos implementados se basan en técnicas estándar de análisis de imagen
        y teoría fotográfica.
    """

    def __init__(self):
        """
        Inicializa el analizador de iluminación.

        Constructor vacío incluido por consistencia de API y posible expansión futura
        con parámetros de configuración o modelos pre-entrenados.
        """
        pass

    def analyze_lighting(self, frame: np.ndarray) -> Dict:
        """
        Realiza análisis completo de iluminación del frame.

        Ejecuta pipeline de análisis que incluye conversión a espacios de color
        apropiados, evaluación de múltiples características de iluminación, y
        clasificación del tipo de iluminación según métricas calculadas.

        Args:
            frame: Frame a analizar en formato BGR (OpenCV estándar) como numpy array
                con dimensiones (H, W, 3).

        Returns:
            Diccionario con análisis completo de iluminación:
                lighting_type (str): Clasificación del tipo de iluminación según
                    LightingType enum.
                exposure (Dict): Análisis detallado de exposición con brillo medio,
                    distribución de zonas tonales y detección de sobre/subexposición.
                contrast (Dict): Métricas múltiples de contraste (desviación estándar,
                    RMS, Michelson) y clasificación cualitativa.
                distribution (Dict): Análisis de distribución espacial de luz con
                    brillo por cuadrantes, varianza y uniformidad.
                light_direction (Dict): Dirección de luz principal calculada mediante
                    análisis de gradientes con ángulo y clasificación direccional.

        Notes:
            El análisis se realiza primero convirtiendo el frame a escala de grises
            para enfocarse en luminancia. También se genera versión HSV aunque
            actualmente no se utiliza (reservada para expansión futura con análisis
            de temperatura de color).

            Todas las métricas numéricas se redondean apropiadamente para balance
            entre precisión y legibilidad en interfaces de usuario.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        exposure = self._analyze_exposure(gray)
        contrast = self._analyze_contrast(gray)
        distribution = self._analyze_light_distribution(gray)
        lighting_type = self._determine_lighting_type(exposure, contrast, distribution)
        light_direction = self._analyze_light_direction(gray)

        return {
            "lighting_type": lighting_type.value,
            "exposure": exposure,
            "contrast": contrast,
            "distribution": distribution,
            "light_direction": light_direction
        }

    def _analyze_exposure(self, gray: np.ndarray) -> Dict:
        """
        Analiza exposición de la imagen mediante histograma y estadísticas de brillo.

        Calcula distribución tonal clasificando píxeles en zonas de sombras, medios
        tonos y altas luces. Detecta sobre y subexposición mediante conteo de píxeles
        en extremos del rango dinámico. Clasifica nivel de exposición global.

        Args:
            gray: Imagen en escala de grises como numpy array 2D con valores [0, 255].

        Returns:
            Diccionario con análisis de exposición:
                mean_brightness (float): Brillo promedio en rango [0.0, 255.0].
                std_brightness (float): Desviación estándar del brillo, indica
                    variabilidad tonal.
                level (str): Clasificación cualitativa del nivel de exposición.
                    Valores: "Sobreexpuesta", "Brillante", "Normal", "Oscura",
                    "Subexpuesta".
                zones (Dict): Distribución proporcional en zonas tonales:
                    - shadows [0, 85]: tercio inferior del rango dinámico
                    - midtones [85, 170]: tercio medio
                    - highlights [170, 255]: tercio superior
                overexposed_pixels (float): Proporción de píxeles saturados (>=250)
                    en rango [0.0, 1.0].
                underexposed_pixels (float): Proporción de píxeles muy oscuros (<=5)
                    en rango [0.0, 1.0].

        Notes:
            División de zonas tonales basada en sistema de zonas de Ansel Adams:
                - Zona I-III (0-85): Sombras profundas y negros
                - Zona IV-VI (85-170): Medios tonos y grises medios
                - Zona VII-X (170-255): Altas luces y blancos

            Umbrales de clasificación de exposición:
                - Sobreexpuesta: brillo medio > 180
                - Brillante: brillo medio > 140
                - Normal: brillo medio > 100
                - Oscura: brillo medio > 60
                - Subexpuesta: brillo medio <= 60

            Detección de clipping:
                - Sobreexposición: píxeles >= 250 (pérdida de detalle en altas luces)
                - Subexposición: píxeles <= 5 (pérdida de detalle en sombras)

            El histograma se normaliza a proporciones [0.0, 1.0] para independencia
            del tamaño de imagen.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()

        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))

        shadows = np.sum(hist[:85])
        midtones = np.sum(hist[85:170])
        highlights = np.sum(hist[170:])

        overexposed = np.sum(gray >= 250) / gray.size
        underexposed = np.sum(gray <= 5) / gray.size

        if mean_brightness > 180:
            exposure_level = "Sobreexpuesta"
        elif mean_brightness > 140:
            exposure_level = "Brillante"
        elif mean_brightness > 100:
            exposure_level = "Normal"
        elif mean_brightness > 60:
            exposure_level = "Oscura"
        else:
            exposure_level = "Subexpuesta"

        return {
            "mean_brightness": round(mean_brightness, 2),
            "std_brightness": round(std_brightness, 2),
            "level": exposure_level,
            "zones": {
                "shadows": round(float(shadows), 3),
                "midtones": round(float(midtones), 3),
                "highlights": round(float(highlights), 3)
            },
            "overexposed_pixels": round(float(overexposed), 4),
            "underexposed_pixels": round(float(underexposed), 4)
        }

    def _analyze_contrast(self, gray: np.ndarray) -> Dict:
        """
        Analiza contraste mediante múltiples métricas complementarias.

        Implementa tres métodos estándar de medición de contraste que capturan
        diferentes aspectos: variabilidad global (desviación estándar), contraste
        local promedio (RMS) y rango dinámico relativo (Michelson). La combinación
        proporciona evaluación robusta del contraste perceptual.

        Args:
            gray: Imagen en escala de grises como numpy array 2D con valores [0, 255].

        Returns:
            Diccionario con análisis de contraste:
                std_contrast (float): Contraste medido como desviación estándar de
                    intensidades [0.0, ~100.0]. Valores altos indican mayor variabilidad.
                rms_contrast (float): Contraste RMS (Root Mean Square) que mide
                    desviación cuadrática media respecto a brillo medio [0.0, ~100.0].
                michelson_contrast (float): Contraste de Michelson normalizado [0.0, 1.0]
                    calculado como (max-min)/(max+min). Captura rango dinámico completo.
                level (str): Clasificación cualitativa del contraste. Valores:
                    "Muy Alto", "Alto", "Medio", "Bajo", "Muy Bajo".
                is_high_contrast (bool): Bandera que indica si el contraste es alto
                    (std_contrast > 60), útil para clasificación binaria rápida.

        Notes:
            Fórmulas matemáticas:
                1. Contraste por desviación estándar (Weber contrast):
                   std = sqrt(mean((I - mean(I))²))

                2. Contraste RMS (equivalente a std para contraste global):
                   RMS = sqrt(mean((I - mean(I))²))

                3. Contraste de Michelson:
                   C = (L_max - L_min) / (L_max + L_min)
                   donde L_max y L_min son luminancias máxima y mínima

            Umbrales de clasificación por std_contrast:
                - Muy Alto: > 70 (escenas con iluminación dramática)
                - Alto: > 50 (contraste cinematográfico típico)
                - Medio: > 30 (contraste normal para video)
                - Bajo: > 15 (iluminación suave o flat)
                - Muy Bajo: <= 15 (escenas muy uniformes)

            El método de desviación estándar es más sensible a variaciones locales,
            mientras que Michelson captura mejor el rango dinámico global. Para
            imágenes con distribución gaussiana, std_contrast ≈ rms_contrast.
        """
        std_contrast = float(np.std(gray))
        rms_contrast = float(np.sqrt(np.mean((gray - np.mean(gray)) ** 2)))

        max_val = float(np.max(gray))
        min_val = float(np.min(gray))
        michelson = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0

        if std_contrast > 70:
            contrast_level = "Muy Alto"
        elif std_contrast > 50:
            contrast_level = "Alto"
        elif std_contrast > 30:
            contrast_level = "Medio"
        elif std_contrast > 15:
            contrast_level = "Bajo"
        else:
            contrast_level = "Muy Bajo"

        return {
            "std_contrast": round(std_contrast, 2),
            "rms_contrast": round(rms_contrast, 2),
            "michelson_contrast": round(michelson, 3),
            "level": contrast_level,
            "is_high_contrast": std_contrast > 60
        }

    def _analyze_light_distribution(self, gray: np.ndarray) -> Dict:
        """
        Analiza distribución espacial de luz mediante partición en cuadrantes.

        Divide la imagen en cuatro regiones y evalúa brillo promedio en cada una
        para caracterizar cómo se distribuye la luz en el espacio del frame.
        Calcula métricas de uniformidad y detecta concentración de luz.

        Args:
            gray: Imagen en escala de grises como numpy array 2D.

        Returns:
            Diccionario con análisis de distribución espacial:
                quadrant_brightness (Dict[str, float]): Brillo promedio por cuadrante
                    con claves: "top_left", "top_right", "bottom_left", "bottom_right".
                    Valores en rango [0.0, 255.0].
                variance (float): Varianza del brillo entre cuadrantes. Valores altos
                    indican distribución no uniforme.
                uniformity (float): Métrica de uniformidad normalizada [0.0, 1.0] donde
                    1.0 indica distribución perfectamente uniforme.
                brightest_area (str): Identificador del cuadrante más brillante.
                darkest_area (str): Identificador del cuadrante más oscuro.
                brightness_range (float): Diferencia entre cuadrante más brillante y
                    más oscuro [0.0, 255.0].
                is_uniform (bool): Bandera que indica distribución uniforme según
                    umbral (varianza < 500).

        Notes:
            La partición en cuadrantes se realiza dividiendo exactamente por la mitad
            en ambas dimensiones. Para imágenes con dimensiones impares, el píxel
            central se asigna al cuadrante inferior/derecho.

            Cálculo de uniformidad:
                uniformity = 1 - (variance / 10000)
                La normalización por 10000 se basa en varianza esperada máxima para
                imágenes de 8 bits (~100² = 10000 para distribución bimodal extrema).
                Valores se limitan a rango [0.0, 1.0] mediante clipping.

            Umbral de uniformidad:
                - Varianza < 500: distribución considerada uniforme
                - Corresponde a desviación estándar < ~22 en escala 0-255
                - Apropiado para detección de iluminación flat

            Interpretación direccional:
                - Concentración superior → luz cenital o contraluz
                - Concentración lateral → luz lateral o rembrandt
                - Distribución uniforme → iluminación flat o múltiples fuentes

            Útil para detectar esquemas de iluminación específicos y evaluar balance
            visual de la composición.
        """
        h, w = gray.shape

        h_mid = h // 2
        w_mid = w // 2

        quadrants = {
            "top_left": gray[:h_mid, :w_mid],
            "top_right": gray[:h_mid, w_mid:],
            "bottom_left": gray[h_mid:, :w_mid],
            "bottom_right": gray[h_mid:, w_mid:]
        }

        quadrant_brightness = {
            name: float(np.mean(quad))
            for name, quad in quadrants.items()
        }

        brightness_values = list(quadrant_brightness.values())
        variance = float(np.var(brightness_values))

        max_quad = max(quadrant_brightness.items(), key=lambda x: x[1])
        min_quad = min(quadrant_brightness.items(), key=lambda x: x[1])

        brightness_range = max_quad[1] - min_quad[1]
        uniformity = 1 - (variance / 10000)

        return {
            "quadrant_brightness": quadrant_brightness,
            "variance": round(variance, 2),
            "uniformity": round(max(0, min(1, uniformity)), 3),
            "brightest_area": max_quad[0],
            "darkest_area": min_quad[0],
            "brightness_range": round(brightness_range, 2),
            "is_uniform": variance < 500
        }

    def _determine_lighting_type(self, exposure: Dict, contrast: Dict,
                                 distribution: Dict) -> LightingType:
        """
        Determina tipo de iluminación cinematográfica basándose en métricas calculadas.

        Clasifica la iluminación según taxonomía estándar de cinematografía,
        considerando combinación de brillo global, contraste y uniformidad de
        distribución espacial.

        Args:
            exposure: Diccionario con análisis de exposición, requiere clave
                'mean_brightness'.
            contrast: Diccionario con análisis de contraste, requiere clave
                'std_contrast'.
            distribution: Diccionario con análisis de distribución espacial, requiere
                clave 'uniformity'.

        Returns:
            Enum LightingType con clasificación del tipo de iluminación detectado.

        Notes:
            Reglas de clasificación (evaluadas en orden de prioridad):

            1. High Key (Luz Alta):
                - Condiciones: brillo > 150 AND contraste < 40
                - Características: imagen brillante con transiciones suaves
                - Uso típico: comedias, publicidad, programas infantiles
                - Esquema de iluminación: múltiples fuentes suaves, fill alto

            2. Low Key (Luz Baja):
                - Condiciones: brillo < 80 AND contraste > 50
                - Características: predominancia de sombras con altas luces puntuales
                - Uso típico: noir, thriller, terror, drama oscuro
                - Esquema: luz dura, ratio alto entre key y fill

            3. Iluminación Dramática:
                - Condiciones: contraste > 60 (independiente de brillo)
                - Características: alto rango dinámico, transiciones abruptas
                - Uso típico: cine dramático, escenas tensas
                - Esquema: luz lateral o contraluz pronunciado

            4. Iluminación Plana (Flat):
                - Condiciones: uniformidad > 0.8 AND contraste < 30
                - Características: distribución muy uniforme, sombras mínimas
                - Uso típico: entrevistas, documentales, broadcast news
                - Esquema: iluminación frontal difusa, múltiples fuentes suaves

            5. Iluminación Normal:
                - Condiciones: no cumple criterios anteriores
                - Características: balance entre luz y sombra, contraste moderado
                - Uso típico: narrativa general, escenas neutras
                - Esquema: iluminación de tres puntos estándar

            La clasificación sigue orden jerárquico donde características más
            específicas (High/Low Key) tienen prioridad sobre generales (Normal).
            El contraste dramático se evalúa antes que uniformidad por ser más
            distintivo visualmente.
        """
        brightness = exposure["mean_brightness"]
        contrast_val = contrast["std_contrast"]
        uniformity = distribution["uniformity"]

        if brightness > 150 and contrast_val < 40:
            return LightingType.HIGH_KEY

        if brightness < 80 and contrast_val > 50:
            return LightingType.LOW_KEY

        if contrast_val > 60:
            return LightingType.DRAMATIC

        if uniformity > 0.8 and contrast_val < 30:
            return LightingType.FLAT

        return LightingType.NORMAL

    def _analyze_light_direction(self, gray: np.ndarray) -> Dict:
        """
        Analiza dirección de luz principal mediante análisis de gradientes.

        Calcula gradientes de intensidad en direcciones X e Y, determina magnitud
        y ángulo de transiciones tonales dominantes, y clasifica dirección de luz
        en ocho categorías direccionales principales.

        Args:
            gray: Imagen en escala de grises como numpy array 2D.

        Returns:
            Diccionario con análisis de dirección de luz:
                angle_degrees (float): Ángulo de dirección de luz en grados [-180, 180]
                    donde 0° = derecha, 90° = abajo, ±180° = izquierda, -90° = arriba.
                direction (str): Clasificación direccional en octantes. Valores posibles:
                    "Derecha", "Inferior Derecha", "Inferior", "Inferior Izquierda",
                    "Izquierda", "Superior Izquierda", "Superior", "Superior Derecha".
                strength (float): Fuerza promedio de gradientes [0.0, ~255.0]. Valores
                    altos indican transiciones tonales pronunciadas (luz direccional
                    fuerte), valores bajos indican iluminación difusa.

        Notes:
            Algoritmo de análisis:
                1. Cálculo de gradientes Sobel en X e Y con kernel 5x5
                2. Magnitud de gradiente: mag = sqrt(gx² + gy²)
                3. Ángulo de gradiente: angle = arctan2(gy, gx)
                4. Promedio ponderado por magnitud para priorizar bordes fuertes
                5. Conversión a grados y clasificación en octantes de 45°

            Interpretación de ángulos:
                - Gradientes apuntan en dirección de incremento de intensidad
                - Dirección de luz calculada corresponde a zona más brillante
                - Por ejemplo, gradiente hacia derecha indica luz desde derecha

            Clasificación en octantes:
                - [-22.5°, 22.5°): Derecha (horizontal positivo)
                - [22.5°, 67.5°): Inferior Derecha (diagonal positiva)
                - [67.5°, 112.5°): Inferior (vertical positivo)
                - [112.5°, 157.5°): Inferior Izquierda
                - [157.5°, 180°] y [-180°, -157.5°): Izquierda
                - [-157.5°, -112.5°): Superior Izquierda
                - [-112.5°, -67.5°): Superior
                - [-67.5°, -22.5°): Superior Derecha

            La fuerza del gradiente (strength) es indicador de dureza de luz:
                - Valores altos (>50): luz dura con sombras definidas
                - Valores medios (20-50): luz moderada
                - Valores bajos (<20): luz muy difusa o múltiples fuentes

            Limitaciones:
                - Asume fuente de luz principal dominante
                - Múltiples fuentes pueden promediar y cancelar direccionalidad
                - Más efectivo con iluminación lateral o rembrandt que con frontal
        """
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)

        weighted_angle = np.sum(angle * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0

        angle_deg = float(np.degrees(weighted_angle))

        if -22.5 <= angle_deg < 22.5:
            direction = "Derecha"
        elif 22.5 <= angle_deg < 67.5:
            direction = "Inferior Derecha"
        elif 67.5 <= angle_deg < 112.5:
            direction = "Inferior"
        elif 112.5 <= angle_deg < 157.5:
            direction = "Inferior Izquierda"
        elif -67.5 <= angle_deg < -22.5:
            direction = "Superior Derecha"
        elif -112.5 <= angle_deg < -67.5:
            direction = "Superior"
        elif -157.5 <= angle_deg < -112.5:
            direction = "Superior Izquierda"
        else:
            direction = "Izquierda"

        return {
            "angle_degrees": round(angle_deg, 2),
            "direction": direction,
            "strength": round(float(np.mean(magnitude)), 2)
        }


def visualize_lighting(frame: np.ndarray, lighting_info: Dict) -> np.ndarray:
    """
    Dibuja overlay de información de iluminación sobre el frame.

    Renderiza datos de análisis de iluminación como texto con fondo sólido en la
    esquina superior izquierda del frame para visualización en tiempo real o
    debugging.

    Args:
        frame: Frame donde dibujar información en formato BGR (OpenCV) como numpy
            array con dimensiones (H, W, 3). Se modifica in-place.
        lighting_info: Diccionario con resultados de análisis de iluminación,
            típicamente retornado por analyze_lighting(). Debe contener claves:
            'lighting_type', 'exposure', 'contrast', 'light_direction'.

    Returns:
        Frame modificado con overlay de información textual. El array se modifica
        in-place pero también se retorna para encadenamiento de funciones.

    Notes:
        Elementos visuales:
            - Posición: esquina superior izquierda con margen de 10px
            - Espaciado vertical: 30px entre líneas
            - Fondo: rectángulo negro sólido para legibilidad
            - Texto: cian (0, 255, 255) con fuente FONT_HERSHEY_SIMPLEX
            - Escala de fuente: 0.6
            - Grosor: 2 píxeles

        Información mostrada (4 líneas):
            1. Tipo de iluminación cinematográfica
            2. Nivel de exposición (Sobreexpuesta/Normal/Subexpuesta)
            3. Nivel de contraste (Muy Alto/Alto/Medio/Bajo/Muy Bajo)
            4. Dirección de luz principal

        El fondo negro garantiza legibilidad sobre cualquier contenido del frame.
        El color cian se eligió por buen contraste con mayoría de escenas y
        asociación visual con información técnica.

        Útil para:
            - Monitoreo en tiempo real durante captura
            - Debugging de algoritmos de análisis
            - Demostraciones y visualizaciones educativas
            - Validación de resultados de clasificación
    """
    h, w = frame.shape[:2]

    texts = [
        f"Iluminacion: {lighting_info['lighting_type']}",
        f"Exposicion: {lighting_info['exposure']['level']}",
        f"Contraste: {lighting_info['contrast']['level']}",
        f"Direccion: {lighting_info['light_direction']['direction']}"
    ]

    y_offset = 30
    for i, text in enumerate(texts):
        position = (10, y_offset + i * 30)

        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame,
            (position[0] - 5, position[1] - text_h - 5),
            (position[0] + text_w + 5, position[1] + baseline + 5),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame, text, position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )

    return frame