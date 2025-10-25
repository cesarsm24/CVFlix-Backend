"""
video_processor.py

Procesador principal de video con análisis cinematográfico integral. Orquesta
múltiples analizadores especializados para detección facial, reconocimiento de
actores, clasificación de planos, análisis de composición, iluminación, colores
y movimiento de cámara.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Features:
    - Análisis facial con reconocimiento de actores y detección emocional
    - Tracking inteligente de rostros entre frames consecutivos
    - Clasificación de tipos de plano cinematográfico
    - Evaluación de composición visual (regla de tercios, simetría, balance)
    - Análisis de iluminación y temperatura de color
    - Detección de movimiento de cámara
    - Acumulación de estadísticas para visualizaciones agregadas
    - Manejo robusto de errores con logging detallado
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional
from collections import Counter

from ..analysis.face_detection import FaceDetector, FaceRecognizer, draw_face_box
from ..analysis.shot_analysis import ShotAnalyzer
from ..analysis.composition import CompositionAnalyzer
from ..analysis.lighting import LightingAnalyzer
from ..analysis.color_analysis import ColorAnalyzer
from ..analysis.camera_movement import CameraMovementAnalyzer
from ..analysis.emotion_detection import EmotionDetector, draw_emotion_label
from ..config import (
    USE_FACE_TRACKING,
    TRACKING_THRESHOLD,
    ANALYSIS_CONFIG,
    EMOTION_MODEL_PATH
)
from ..utils.exceptions import VideoProcessingException

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Orquestador principal de análisis cinematográfico de video.

    Coordina múltiples analizadores especializados ejecutando pipeline completo
    de procesamiento por frame: detección facial, reconocimiento de actores,
    análisis emocional, clasificación de planos, evaluación compositiva,
    iluminación, colores y movimiento de cámara. Acumula estadísticas globales
    para generación de reportes agregados.

    Attributes:
        face_detector (FaceDetector): Detector DNN de rostros.
        face_recognizer (FaceRecognizer): Reconocedor de actores por embeddings.
        shot_analyzer (ShotAnalyzer): Clasificador de tipos de plano.
        composition_analyzer (CompositionAnalyzer): Evaluador de composición visual.
        lighting_analyzer (LightingAnalyzer): Analizador de iluminación.
        color_analyzer (ColorAnalyzer): Analizador cromático con histogramas.
        camera_analyzer (CameraMovementAnalyzer): Detector de movimiento de cámara.
        emotion_detector (EmotionDetector): Clasificador de emociones faciales.
        detected_actors (Dict[int, Dict]): Acumulador de actores detectados.
        shot_types_count (Counter): Contador de tipos de plano.
        lighting_types_count (Counter): Contador de tipos de iluminación.
        emotions_count (Counter): Contador de emociones detectadas.
        composition_data (Dict): Datos temporales de métricas compositivas.
        total_frames_analyzed (int): Contador total de frames procesados.
    """

    def __init__(self):
        """
        Inicializa procesador cargando todos los analizadores especializados.

        Raises:
            VideoProcessingException: Si falla inicialización de algún analizador.

        Notes:
            El detector de emociones se carga condicionalmente según existencia
            del archivo de modelo. Si no está disponible, las detecciones
            emocionales utilizarán análisis geométrico como fallback.
        """
        logger.info("Inicializando VideoProcessor...")

        try:
            self.face_detector = FaceDetector()
            self.face_recognizer = FaceRecognizer()
            self.shot_analyzer = ShotAnalyzer()
            self.composition_analyzer = CompositionAnalyzer()
            self.lighting_analyzer = LightingAnalyzer()
            self.color_analyzer = ColorAnalyzer()
            self.camera_analyzer = CameraMovementAnalyzer()

            emotion_model = str(EMOTION_MODEL_PATH) if EMOTION_MODEL_PATH.exists() else None
            self.emotion_detector = EmotionDetector(model_path=emotion_model)

            logger.info("Analizadores cargados correctamente")

        except Exception as e:
            logger.error(f"Error inicializando analizadores: {e}")
            raise VideoProcessingException(
                "Error inicializando procesador de video",
                str(e)
            )

        self.detected_actors: Dict[int, Dict] = {}
        self.last_face_locations = []

        self.shot_types_count = Counter()
        self.lighting_types_count = Counter()
        self.emotions_count = Counter()
        self.color_temperatures_count = Counter()
        self.color_schemes_count = Counter()
        self.global_colors = []
        self.total_frames_analyzed = 0

        self.composition_data = {
            'rule_of_thirds_scores': [],
            'symmetry_scores': [],
            'balance_scores': [],
            'lines_count': []
        }

        logger.info("VideoProcessor inicializado completamente")

    def load_actor_encodings(self, actors_data: List[Dict]):
        """
        Carga encodings faciales de actores conocidos para reconocimiento.

        Args:
            actors_data: Lista de diccionarios con datos de actores incluyendo
                encodings de 128 dimensiones pre-calculados.

        Raises:
            VideoProcessingException: Si falla carga de encodings.

        Notes:
            Los encodings deben haberse generado previamente desde fotografías
            de referencia de cada actor usando face_recognition library.
        """
        try:
            self.face_recognizer.load_actor_encodings(actors_data)
            logger.info(f"{len(actors_data)} encodings de actores cargados")
        except Exception as e:
            logger.error(f"Error cargando encodings: {e}")
            raise VideoProcessingException("Error cargando actores", str(e))

    def process_frame_optimized(
        self,
        frame: np.ndarray,
        frame_number: int,
        detect_faces: bool = True,
        full_analysis: bool = False,
        last_faces: List = None
    ) -> Dict:
        """
        Procesa frame con análisis selectivo optimizado según flags.

        Ejecuta pipeline de análisis configurable permitiendo omitir detecciones
        costosas en frames intermedios. Soporta tracking de rostros entre frames
        consecutivos para reducir overhead de detección.

        Args:
            frame: Frame en formato BGR (OpenCV) a procesar.
            frame_number: Índice del frame en secuencia de video.
            detect_faces: Si ejecutar detección facial en este frame.
            full_analysis: Si ejecutar análisis completo (composición, iluminación).
            last_faces: Rostros detectados en frame anterior para tracking.

        Returns:
            Diccionario con resultados de todos los análisis ejecutados:
                frame_number (int): Índice del frame.
                faces (List[Dict]): Rostros detectados con reconocimiento y emociones.
                shot_type (Dict): Clasificación de tipo de plano.
                composition (Dict): Métricas de composición visual.
                lighting (Dict): Análisis de iluminación.
                colors (Dict): Análisis cromático.
                camera_movement (Dict): Detección de movimiento de cámara.
                emotions (List[Dict]): Emociones detectadas en el frame.

        Notes:
            Estrategia de optimización:
                - detect_faces=False: reutiliza detecciones del frame anterior
                - full_analysis=False: omite análisis de composición e iluminación
                - Tracking facial reduce detecciones DNN costosas

            El análisis de movimiento de cámara se ejecuta siempre para mantener
            estado consistente del analizador entre frames.
        """
        results = {
            "frame_number": frame_number,
            "faces": [],
            "shot_type": None,
            "composition": None,
            "lighting": None,
            "colors": None,
            "camera_movement": None,
            "emotions": []
        }

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_boxes = []
            detect_emotions = ANALYSIS_CONFIG.get("emotion_detection", {}).get("enabled", True) and full_analysis

            # ==================== DETECCIÓN FACIAL ====================
            if detect_faces:
                try:
                    face_locations, face_boxes = self.face_detector.detect_faces(frame)
                    self.last_face_locations = face_boxes

                    if face_locations:
                        face_encodings = self.face_recognizer.extract_encodings(
                            rgb_frame,
                            face_locations
                        )

                        recognized = self.face_recognizer.recognize_faces(face_encodings)

                        for i, (box, recognition) in enumerate(zip(face_boxes, recognized)):
                            face_info = {
                                "box": box,
                                "recognized": recognition is not None
                            }

                            # Detección de emociones
                            emotion_data = None
                            if detect_emotions:
                                try:
                                    x1, y1, x2, y2 = box

                                    if (x1 >= 0 and y1 >= 0 and
                                        x2 <= frame.shape[1] and y2 <= frame.shape[0]):

                                        face_region = frame[y1:y2, x1:x2]

                                        if face_region.size > 0:
                                            emotion_data = self.emotion_detector.detect_emotion(
                                                face_region
                                            )

                                            if emotion_data:
                                                self.emotions_count[emotion_data["emotion"]] += 1
                                                face_info["emotion"] = emotion_data

                                except Exception as e:
                                    logger.warning(f"Error detectando emoción: {e}")

                            # Reconocimiento de actores
                            if recognition:
                                actor_id = recognition["actor_id"]

                                if actor_id not in self.detected_actors:
                                    self.detected_actors[actor_id] = {
                                        "actor_id": actor_id,
                                        "nombre": recognition["nombre"],
                                        "personaje": recognition["personaje"],
                                        "foto_url": recognition["foto_url"],
                                        "detecciones": 0,
                                        "similitudes": [],
                                        "similitud_maxima": 0
                                    }

                                actor = self.detected_actors[actor_id]
                                actor["detecciones"] += 1
                                actor["similitudes"].append(recognition["similitud"])
                                actor["similitud_promedio"] = np.mean(actor["similitudes"])

                                if recognition["similitud"] > actor["similitud_maxima"]:
                                    actor["similitud_maxima"] = recognition["similitud"]

                                face_info.update({
                                    "actor_id": actor_id,
                                    "nombre": recognition["nombre"],
                                    "personaje": recognition["personaje"],
                                    "similitud": round(recognition["similitud"], 1)
                                })

                            results["faces"].append(face_info)

                except Exception as e:
                    logger.warning(f"Error en detección facial frame {frame_number}: {e}")

            else:
                # Reutilizar detecciones del frame anterior
                face_boxes = self.last_face_locations

            # ==================== ANÁLISIS DE TIPO DE PLANO ====================
            if ANALYSIS_CONFIG.get("shot_type", {}).get("enabled", True) and face_boxes:
                try:
                    shot_result = self.shot_analyzer.analyze_shot_type(frame, face_boxes)
                    results["shot_type"] = shot_result
                    if shot_result:
                        self.shot_types_count[shot_result["shot_type"]] += 1
                except Exception as e:
                    logger.warning(f"Error en análisis de plano: {e}")

            # ==================== ANÁLISIS COMPLETO ====================
            if full_analysis:
                # Análisis de composición
                if ANALYSIS_CONFIG.get("composition", {}).get("enabled", True):
                    try:
                        composition_result = self.composition_analyzer.analyze_composition(frame)
                        results["composition"] = composition_result

                        if composition_result:
                            self.composition_data['rule_of_thirds_scores'].append(
                                composition_result['rule_of_thirds']['score']
                            )
                            self.composition_data['symmetry_scores'].append(
                                composition_result['symmetry']['vertical_symmetry']
                            )
                            self.composition_data['balance_scores'].append(
                                composition_result['balance']['horizontal_balance']
                            )
                            self.composition_data['lines_count'].append(
                                composition_result['lines']['num_lines']
                            )
                    except Exception as e:
                        logger.warning(f"Error en análisis de composición: {e}")

                # Análisis de iluminación
                if ANALYSIS_CONFIG.get("lighting", {}).get("enabled", True):
                    try:
                        lighting_result = self.lighting_analyzer.analyze_lighting(frame)
                        results["lighting"] = lighting_result
                        if lighting_result:
                            self.lighting_types_count[lighting_result["lighting_type"]] += 1
                    except Exception as e:
                        logger.warning(f"Error en análisis de iluminación: {e}")

                # Análisis de colores
                if ANALYSIS_CONFIG.get("colors", {}).get("enabled", True):
                    try:
                        color_result = self.color_analyzer.analyze_colors(frame, n_colors=5)
                        results["colors"] = color_result

                        if color_result:
                            self.color_temperatures_count[
                                color_result["temperature"]["label"]
                            ] += 1
                            self.color_schemes_count[
                                color_result["color_scheme"]["scheme"]
                            ] += 1

                            for color in color_result["dominant_colors"]:
                                self.global_colors.append(color["rgb"])

                            self.color_analyzer.accumulate_histogram(frame)

                    except Exception as e:
                        logger.warning(f"Error en análisis de colores: {e}")

            # ==================== ANÁLISIS DE MOVIMIENTO DE CÁMARA ====================
            if ANALYSIS_CONFIG.get("camera_movement", {}).get("enabled", True):
                try:
                    # Verificar dimensiones antes de convertir
                    if len(frame.shape) == 2:
                        # Ya es grayscale (2D)
                        gray = frame
                    elif frame.shape[2] == 1:
                        # Ya es grayscale pero con dimensión extra
                        gray = frame.squeeze()
                    else:
                        # Convertir BGR a grayscale
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    camera_result = self.camera_analyzer.analyze_movement(gray, frame_number)
                    results["camera_movement"] = camera_result
                except Exception as e:
                    logger.warning(f"Error en análisis de movimiento: {e}")

            self.total_frames_analyzed += 1

        except Exception as e:
            logger.error(f"Error procesando frame {frame_number}: {e}")
            raise VideoProcessingException(f"Error procesando frame {frame_number}", str(e))

        return results

    def _calculate_global_palette(self, n_colors: int = 5) -> List[Dict]:
        """
        Calcula paleta de colores global del video mediante K-means clustering.

        Args:
            n_colors: Número de colores dominantes a extraer.

        Returns:
            Lista de diccionarios con colores RGB, hex y nombre.
            Lista vacía si no hay datos de color acumulados.

        Notes:
            Ejecuta K-means sobre todos los colores dominantes acumulados de
            cada frame analizado para identificar la paleta cromática global
            característica del video completo.
        """
        if not self.global_colors:
            return []

        try:
            colors_array = np.array(self.global_colors, dtype=np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            _, labels, centers = cv2.kmeans(
                colors_array,
                n_colors,
                None,
                criteria,
                10,
                cv2.KMEANS_PP_CENTERS
            )

            palette = []
            for center in centers:
                rgb = center.astype(int).tolist()
                hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
                palette.append({
                    "rgb": rgb,
                    "hex": hex_color
                })

            return palette

        except Exception as e:
            logger.warning(f"Error calculando paleta global: {e}")
            return []

    def get_final_results(self) -> Dict:
        """
        Genera resultados finales agregados del procesamiento completo.

        Returns:
            Diccionario con estadísticas globales, actores detectados,
            distribuciones de características cinematográficas y datos
            para visualizaciones (histogramas, timelines).

        Raises:
            VideoProcessingException: Si falla generación de resultados.

        Notes:
            Los resultados incluyen:
                - Lista de actores ordenada por número de detecciones
                - Resúmenes de distribuciones porcentuales
                - Paleta cromática global
                - Histograma RGB acumulado
                - Timeline de movimientos de cámara
                - Métricas promedio de composición
        """
        logger.info("Generando resultados finales...")

        try:
            actors_list = sorted(
                self.detected_actors.values(),
                key=lambda x: x["detecciones"],
                reverse=True
            )

            final_actors = []
            for actor in actors_list:
                final_actors.append({
                    "actor_id": actor["actor_id"],
                    "nombre": actor["nombre"],
                    "personaje": actor["personaje"],
                    "foto_url": actor["foto_url"],
                    "detecciones": actor["detecciones"],
                    "similitud": round(actor["similitud_promedio"], 1),
                    "similitud_maxima": round(actor["similitud_maxima"], 1)
                })

            camera_summary = self.camera_analyzer.get_movement_summary()
            shot_types_summary = self._get_percentage_summary(self.shot_types_count)
            lighting_summary = self._get_percentage_summary(self.lighting_types_count)
            emotions_summary = self._get_percentage_summary(self.emotions_count)
            color_temp_summary = self._get_percentage_summary(self.color_temperatures_count)
            color_scheme_summary = self._get_percentage_summary(self.color_schemes_count)
            global_palette = self._calculate_global_palette(n_colors=5)

            histogram_data = None
            if self.color_analyzer.frames_count > 0:
                histogram_data = {
                    "r": self.color_analyzer.hist_r_accumulated.tolist(),
                    "g": self.color_analyzer.hist_g_accumulated.tolist(),
                    "b": self.color_analyzer.hist_b_accumulated.tolist(),
                    "frames_count": self.color_analyzer.frames_count
                }

            camera_timeline = self.camera_analyzer.get_timeline_data()

            composition_data = None
            if any(self.composition_data.values()):
                composition_data = {
                    "rule_of_thirds_scores": self.composition_data['rule_of_thirds_scores'],
                    "symmetry_scores": self.composition_data['symmetry_scores'],
                    "balance_scores": self.composition_data['balance_scores'],
                    "lines_count": self.composition_data['lines_count']
                }

            composition_summary = None
            if any(self.composition_data.values()):
                total_analyzed = len(self.composition_data['rule_of_thirds_scores'])
                if total_analyzed > 0:
                    composition_summary = {
                        "total_analyzed": total_analyzed,
                        "avg_rule_of_thirds": round(
                            float(np.mean(self.composition_data['rule_of_thirds_scores'])),
                            3
                        ) if self.composition_data['rule_of_thirds_scores'] else 0.0,
                        "avg_symmetry": round(
                            float(np.mean(self.composition_data['symmetry_scores'])),
                            3
                        ) if self.composition_data['symmetry_scores'] else 0.0,
                        "avg_balance": round(
                            float(np.mean(self.composition_data['balance_scores'])),
                            3
                        ) if self.composition_data['balance_scores'] else 0.0,
                        "avg_lines": round(
                            float(np.mean(self.composition_data['lines_count'])),
                            1
                        ) if self.composition_data['lines_count'] else 0.0
                    }

            results = {
                "detected_actors": final_actors,
                "total_actors_detected": len(final_actors),
                "camera_movement_summary": camera_summary,
                "shot_types_summary": {
                    "total_analyzed": sum(self.shot_types_count.values()),
                    "distribution": shot_types_summary,
                    "most_common": self._get_most_common(self.shot_types_count)
                },
                "lighting_summary": {
                    "total_analyzed": sum(self.lighting_types_count.values()),
                    "distribution": lighting_summary,
                    "most_common": self._get_most_common(self.lighting_types_count)
                },
                "emotions_summary": {
                    "total_detected": sum(self.emotions_count.values()),
                    "distribution": emotions_summary,
                    "most_common": self._get_most_common(self.emotions_count)
                },
                "color_analysis_summary": {
                    "temperature_distribution": color_temp_summary,
                    "most_common_temperature": self._get_most_common(self.color_temperatures_count),
                    "color_scheme_distribution": color_scheme_summary,
                    "most_common_scheme": self._get_most_common(self.color_schemes_count),
                    "global_palette": global_palette
                },
                "composition_summary": composition_summary,
                "total_frames_analyzed": self.total_frames_analyzed,
                "histogram_data": histogram_data,
                "camera_timeline": camera_timeline,
                "composition_data": composition_data
            }

            logger.info("Resultados finales generados correctamente")
            return results

        except Exception as e:
            logger.error(f"Error generando resultados finales: {e}")
            raise VideoProcessingException("Error al generar resultados finales", str(e))

    def _get_percentage_summary(self, counter: Counter) -> Dict[str, float]:
        """
        Convierte Counter a diccionario de porcentajes.

        Args:
            counter: Counter con conteos absolutos.

        Returns:
            Diccionario mapeando claves a porcentajes redondeados a 1 decimal.
        """
        total = sum(counter.values())
        if total == 0:
            return {}

        return {
            key: round((count / total) * 100, 1)
            for key, count in counter.items()
        }

    def _get_most_common(self, counter: Counter) -> Optional[str]:
        """
        Obtiene elemento más frecuente de un Counter.

        Args:
            counter: Counter a analizar.

        Returns:
            Elemento más común o None si Counter vacío.
        """
        if not counter:
            return None
        return counter.most_common(1)[0][0]

    def reset(self):
        """
        Reinicia estado del procesador para análisis de nuevo video.

        Limpia todos los acumuladores, contadores y estado de analizadores
        manteniendo los modelos cargados en memoria.

        Raises:
            VideoProcessingException: Si falla reset de algún componente.

        Notes:
            Los encodings de actores no se limpian, permitiendo reutilizar
            el procesador para múltiples videos del mismo contenido sin
            recargar datos de TMDB.
        """
        logger.info("Reseteando VideoProcessor...")

        try:
            self.detected_actors.clear()
            self.last_face_locations = []

            self.shot_types_count.clear()
            self.lighting_types_count.clear()
            self.emotions_count.clear()
            self.color_temperatures_count.clear()
            self.color_schemes_count.clear()
            self.global_colors.clear()
            self.total_frames_analyzed = 0

            self.camera_analyzer.reset()
            self.color_analyzer.reset_histogram()

            self.composition_data = {
                'rule_of_thirds_scores': [],
                'symmetry_scores': [],
                'balance_scores': [],
                'lines_count': []
            }

            logger.info("VideoProcessor reseteado correctamente")

        except Exception as e:
            logger.error(f"Error reseteando VideoProcessor: {e}")
            raise VideoProcessingException("Error al resetear procesador", str(e))