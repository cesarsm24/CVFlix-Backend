"""
shot_analyzer.py

Módulo para análisis y clasificación automática de tipos de plano cinematográfico
mediante detección de rostros y figuras humanas. Implementa taxonomía estándar de
encuadres desde plano detalle hasta gran plano general.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Detección HOG de personas y procesamiento de imágenes
    - numpy: Operaciones con arrays

Usage:
    from app.analysis.shot_analyzer import ShotAnalyzer

    analyzer = ShotAnalyzer()
    result = analyzer.analyze_shot_type(frame, face_boxes)

    print(f"Tipo: {result['shot_type']}")
    print(f"Confianza: {result['confidence']}")
"""

import cv2
import numpy as np
from typing import Dict, Tuple
from enum import Enum


class ShotType(Enum):
    """
    Enumeración de tipos de plano cinematográfico según taxonomía estándar.

    Define clasificación jerárquica de encuadres desde planos muy cerrados
    (detalle, primer plano) hasta planos muy abiertos (plano general), basada
    en convenciones de lenguaje cinematográfico internacional.
    """
    EXTREME_CLOSEUP = "Plano Detalle"
    CLOSEUP = "Primer Plano"
    MEDIUM_CLOSEUP = "Plano Medio Corto"
    MEDIUM_SHOT = "Plano Medio"
    MEDIUM_FULL = "Plano Americano"
    FULL_SHOT = "Plano Entero"
    LONG_SHOT = "Plano General"
    EXTREME_LONG = "Gran Plano General"


class ShotAnalyzer:
    """
    Analizador de tipo de plano cinematográfico mediante visión por computadora.

    Implementa sistema híbrido de clasificación que utiliza detección de rostros
    como método primario (alta precisión) y detección HOG de personas como fallback.
    La clasificación se basa en ratios de tamaño y posición de elementos detectados
    respecto al frame completo.

    Attributes:
        hog: Descriptor HOG (Histogram of Oriented Gradients) configurado con
            detector SVM pre-entrenado para detección de personas. Utiliza modelo
            estándar de Dalal-Triggs entrenado en dataset INRIA
    """

    def __init__(self):
        """
        Inicializa el analizador cargando descriptor HOG para detección de personas.

        Configura HOG descriptor con detector SVM por defecto de OpenCV, entrenado
        en dataset INRIA Person para detección de peatones. El detector es robusto
        ante variaciones de pose y escala.

        Notes:
            Configuración del detector HOG:
                - Tamaño de ventana: 64x128 píxeles
                - Tamaño de celda: 8x8 píxeles
                - Tamaño de bloque: 16x16 píxeles (2x2 celdas)
                - Número de bins: 9 orientaciones
                - Clasificador: SVM lineal pre-entrenado

            El modelo fue entrenado con dataset INRIA Person que contiene 1805
            imágenes positivas de personas en diversas poses y 1218 negativas.
        """
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def analyze_shot_type(self, frame: np.ndarray, face_boxes: list = None) -> Dict:
        """
        Analiza y clasifica el tipo de plano cinematográfico del frame.

        Determina el encuadre utilizando método híbrido: análisis basado en rostros
        si están disponibles (mayor precisión para planos cerrados), o detección de
        personas completas como alternativa (mejor para planos abiertos).

        Args:
            frame: Frame a analizar en formato BGR como numpy array con
                dimensiones (H, W, 3)
            face_boxes: Lista opcional de bounding boxes de rostros detectados en
                formato [(x1, y1, x2, y2), ...]. Si se proporciona, se usa método
                de análisis basado en rostros que es más preciso para planos cerrados.
                None para usar solo detección de personas

        Returns:
            Diccionario con resultados de análisis:
                shot_type: Tipo de plano clasificado según ShotType enum
                confidence: Nivel de confianza de la clasificación [0.0, 1.0]

                Si análisis basado en rostros, incluye adicionalmente:
                    face_height_ratio: Ratio altura_rostro / altura_frame
                    face_area_ratio: Ratio área_rostro / área_frame
                    vertical_position: Posición vertical normalizada [0.0, 1.0]

                Si análisis basado en personas, incluye:
                    person_height_ratio: Ratio altura_persona / altura_frame
                    person_area_ratio: Ratio área_persona / área_frame
                    people_detected: Número de personas detectadas en el frame

        Notes:
            Estrategia de análisis:
                1. Si face_boxes disponible y no vacío → _analyze_from_faces()
                2. Si no hay rostros detectados → _analyze_from_people()
                3. Si no hay detecciones → LONG_SHOT con confianza baja

            El método basado en rostros es preferido porque:
                - Mayor precisión en planos cerrados y medios
                - Rostros son más estables que detección de cuerpo completo
                - Permite diferenciar sutilmente entre tipos de primer plano

            El método basado en personas es útil cuando:
                - Rostros no visibles (ángulo, iluminación, oclusión)
                - Planos más abiertos donde rostro es pequeño
                - Necesidad de contar personas en escena
        """
        h, w = frame.shape[:2]
        frame_area = h * w

        if face_boxes and len(face_boxes) > 0:
            return self._analyze_from_faces(frame, face_boxes, h, w)

        return self._analyze_from_people(frame, h, w, frame_area)

    def _analyze_from_faces(self, frame: np.ndarray, face_boxes: list,
                            frame_h: int, frame_w: int) -> Dict:
        """
        Clasifica tipo de plano basándose en tamaño y posición de rostros detectados.

        Método primario de clasificación que utiliza rostros como referencia
        principal. Analiza el rostro más grande detectado calculando ratios de
        tamaño respecto al frame completo.

        Args:
            frame: Frame completo en formato BGR
            face_boxes: Lista de tuplas con coordenadas de bounding boxes faciales
                en formato (x1, y1, x2, y2)
            frame_h: Altura del frame en píxeles
            frame_w: Ancho del frame en píxeles

        Returns:
            Diccionario con tipo de plano clasificado, confianza y métricas de análisis.
            Si face_boxes está vacía, retorna LONG_SHOT con confianza baja (0.3)

        Notes:
            Métricas calculadas:
                - height_ratio: altura_rostro / altura_frame
                  Métrica principal para clasificación de planos cerrados

                - area_ratio: área_rostro / área_frame
                  Métrica secundaria para refinar clasificación

                - vertical_pos: posición_vertical_centro_rostro / altura_frame
                  Útil para casos ambiguos

            Umbrales de clasificación por height_ratio:
                - > 0.7 (70% del frame): Plano Detalle (EXTREME_CLOSEUP)
                  Muestra parte del rostro. Énfasis expresivo máximo.

                - > 0.4 (40%): Primer Plano (CLOSEUP)
                  Rostro completo llena frame. Conexión emocional intensa.

                - > 0.25 (25%): Plano Medio Corto (MEDIUM_CLOSEUP)
                  Cabeza y hombros. Conversación íntima.

                - > 0.15 (15%): Plano Medio (MEDIUM_SHOT)
                  Desde cintura hacia arriba. Diálogo estándar.

                - > 0.1 (10%): Plano Americano (MEDIUM_FULL)
                  Desde rodillas hacia arriba.

                - <= 0.1: Plano Entero (FULL_SHOT)
                  Cuerpo completo visible. Contexto con acción.

            Los niveles de confianza disminuyen gradualmente en planos más abiertos
            donde la detección facial es menos precisa como indicador único.
        """
        if not face_boxes:
            return {"shot_type": ShotType.LONG_SHOT.value, "confidence": 0.3}

        largest_face = max(face_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = largest_face

        face_h = y2 - y1
        face_w = x2 - x1
        face_area = face_h * face_w
        frame_area = frame_h * frame_w

        area_ratio = face_area / frame_area
        height_ratio = face_h / frame_h
        vertical_pos = (y1 + y2) / 2 / frame_h

        if height_ratio > 0.7 or area_ratio > 0.5:
            shot_type = ShotType.EXTREME_CLOSEUP
            confidence = 0.9
        elif height_ratio > 0.4 or area_ratio > 0.2:
            shot_type = ShotType.CLOSEUP
            confidence = 0.85
        elif height_ratio > 0.25:
            shot_type = ShotType.MEDIUM_CLOSEUP
            confidence = 0.8
        elif height_ratio > 0.15:
            shot_type = ShotType.MEDIUM_SHOT
            confidence = 0.75
        elif height_ratio > 0.1:
            shot_type = ShotType.MEDIUM_FULL
            confidence = 0.7
        else:
            shot_type = ShotType.FULL_SHOT
            confidence = 0.65

        return {
            "shot_type": shot_type.value,
            "confidence": confidence,
            "face_height_ratio": round(height_ratio, 3),
            "face_area_ratio": round(area_ratio, 3),
            "vertical_position": round(vertical_pos, 3)
        }

    def _analyze_from_people(self, frame: np.ndarray, frame_h: int,
                             frame_w: int, frame_area: int) -> Dict:
        """
        Clasifica tipo de plano mediante detección HOG de personas completas.

        Método de fallback cuando no hay rostros detectados. Utiliza HOG+SVM para
        detectar siluetas de personas y clasifica plano basándose en ratio de altura
        de la persona más grande respecto al frame.

        Args:
            frame: Frame completo en formato BGR para análisis HOG
            frame_h: Altura del frame en píxeles
            frame_w: Ancho del frame en píxeles
            frame_area: Área total del frame en píxeles cuadrados

        Returns:
            Diccionario con tipo de plano, confianza y métricas de detección.
            Si no se detectan personas, retorna LONG_SHOT con confianza media (0.5)

        Notes:
            Pipeline de detección:
                1. Redimensionado a 50% de escala para acelerar detección
                2. Detección multi-escala con parámetros optimizados:
                   - winStride=(8,8): paso de ventana deslizante
                   - padding=(4,4): margen de seguridad
                   - scale=1.05: factor de escalado piramidal
                3. Reescalado de coordenadas a dimensiones originales
                4. Selección de persona más grande como sujeto principal
                5. Clasificación basada en height_ratio

            Umbrales de clasificación por height_ratio:
                - > 0.85 (85%): Primer Plano (CLOSEUP)
                  Persona llena casi todo el frame.

                - > 0.65 (65%): Plano Medio (MEDIUM_SHOT)
                  Aproximadamente mitad superior del cuerpo visible.

                - > 0.45 (45%): Plano Americano (MEDIUM_FULL)
                  Desde rodillas hacia arriba.

                - > 0.25 (25%): Plano Entero (FULL_SHOT)
                  Cuerpo completo visible con margen.

                - <= 0.25: Plano General (LONG_SHOT)
                  Persona pequeña en contexto amplio.

            Ventajas del método HOG:
                - Robusto ante variaciones de pose y vestimenta
                - No requiere rostro visible
                - Funciona bien en planos medios y abiertos

            Limitaciones:
                - Menos preciso que rostros para planos muy cerrados
                - Mayor coste computacional
                - Puede fallar con oclusiones parciales
                - Asume personas en posición vertical
        """
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)

        people, weights = self.hog.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )

        if len(people) == 0:
            return {
                "shot_type": ShotType.LONG_SHOT.value,
                "confidence": 0.5,
                "people_detected": 0
            }

        people = people / scale

        largest_person = max(people, key=lambda box: box[2] * box[3])
        x, y, w, h = largest_person

        person_area = w * h
        height_ratio = h / frame_h
        width_ratio = w / frame_w
        area_ratio = person_area / frame_area

        if height_ratio > 0.85:
            shot_type = ShotType.CLOSEUP
            confidence = 0.75
        elif height_ratio > 0.65:
            shot_type = ShotType.MEDIUM_SHOT
            confidence = 0.8
        elif height_ratio > 0.45:
            shot_type = ShotType.MEDIUM_FULL
            confidence = 0.85
        elif height_ratio > 0.25:
            shot_type = ShotType.FULL_SHOT
            confidence = 0.85
        else:
            shot_type = ShotType.LONG_SHOT
            confidence = 0.8

        return {
            "shot_type": shot_type.value,
            "confidence": confidence,
            "person_height_ratio": round(height_ratio, 3),
            "person_area_ratio": round(area_ratio, 3),
            "people_detected": len(people)
        }


def visualize_shot_type(frame: np.ndarray, shot_info: Dict) -> np.ndarray:
    """
    Dibuja overlay con información del tipo de plano sobre el frame.

    Renderiza etiqueta de texto con tipo de plano y porcentaje de confianza en la
    esquina superior izquierda del frame con fondo sólido para garantizar legibilidad.

    Args:
        frame: Frame donde dibujar información en formato BGR como numpy array con
            dimensiones (H, W, 3). Se modifica in-place
        shot_info: Diccionario con resultados de análisis, debe contener claves
            'shot_type' y 'confidence'

    Returns:
        Frame modificado con overlay de información. El array se modifica in-place
        pero también se retorna para encadenamiento de funciones

    Notes:
        Elementos visuales:
            - Posición: (20, 50) píxeles desde esquina superior izquierda
            - Fuente: FONT_HERSHEY_SIMPLEX, escala 0.8
            - Color: amarillo (255, 255, 0) en espacio BGR
            - Grosor: 2 píxeles
            - Fondo: rectángulo negro sólido con padding de 5px

        Formato del texto:
            "{Tipo de Plano} ({confidence}%)"

        El color amarillo se eligió por alto contraste con fondos oscuros y claros.
        Útil para visualización en tiempo real, verificación de clasificaciones
        automáticas y material educativo.
    """
    text = f"{shot_info['shot_type']} ({shot_info['confidence']:.0%})"

    position = (20, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 0)
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        frame,
        (position[0] - 5, position[1] - text_h - 5),
        (position[0] + text_w + 5, position[1] + baseline + 5),
        (0, 0, 0),
        -1
    )

    cv2.putText(frame, text, position, font, font_scale, color, thickness)

    return frame