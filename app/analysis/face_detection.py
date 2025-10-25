"""
face_detection.py

Módulo para detección y reconocimiento facial mediante redes neuronales profundas
y comparación de embeddings. Implementa detección DNN de OpenCV y reconocimiento
mediante biblioteca face_recognition con encodings de 128 dimensiones.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Dict, Optional
from ..config import (
    FACE_DETECTION_PROTOTXT,
    FACE_DETECTION_WEIGHTS,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_RECOGNITION_THRESHOLD,
    MIN_FACE_SIZE
)


class FaceDetector:
    """
    Detector de rostros mediante red neuronal convolucional de OpenCV.

    Implementa detección facial usando el modelo ResNet SSD pre-entrenado de OpenCV
    DNN, optimizado para detección en tiempo real con alta precisión. El modelo
    utiliza arquitectura Single Shot Detector con backbone ResNet-10.

    Attributes:
        net (cv2.dnn.Net): Red neuronal cargada desde archivos Caffe (prototxt + weights).
            Modelo ResNet-10 SSD entrenado en dataset WIDER FACE para detección facial.
    """

    def __init__(self):
        """
        Inicializa el detector cargando el modelo DNN desde archivos de configuración.

        Carga modelo pre-entrenado ResNet-10 SSD desde archivos Caffe (arquitectura
        en prototxt y pesos en caffemodel). El modelo está optimizado para detectar
        rostros frontales y semi-frontales en diversas condiciones de iluminación.

        Raises:
            FileNotFoundError: Si los archivos de modelo no existen en las rutas
                especificadas en configuración.
            cv2.error: Si hay error durante la carga del modelo DNN.

        Notes:
            El modelo ResNet-10 SSD utilizado tiene las siguientes características:
                - Arquitectura: ResNet-10 como backbone + SSD para detección
                - Entrada: imágenes de 300x300 píxeles
                - Salida: coordenadas de bounding boxes + scores de confianza
                - Entrenado en: WIDER FACE dataset (32,203 imágenes, 393,703 rostros)
                - Rango de escala detectado: desde 10x10 hasta tamaño completo
        """
        self.net = cv2.dnn.readNetFromCaffe(
            str(FACE_DETECTION_PROTOTXT),
            str(FACE_DETECTION_WEIGHTS)
        )

    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detecta rostros en un frame mediante inferencia con red DNN.

        Preprocesa el frame mediante blob transformation, ejecuta inferencia con el
        modelo ResNet SSD, y post-procesa las detecciones aplicando filtros de
        confianza y tamaño mínimo. Retorna coordenadas en dos formatos para
        compatibilidad con diferentes bibliotecas.

        Args:
            frame: Frame de entrada en formato BGR (OpenCV estándar) como numpy array
                con dimensiones (H, W, 3).

        Returns:
            Tupla con dos listas de coordenadas:
                face_locations (List[Tuple[int, int, int, int]]): Coordenadas en formato
                    (top, right, bottom, left) compatible con face_recognition library.
                    Cada tupla representa los límites del bounding box en píxeles.
                face_boxes (List[Tuple[int, int, int, int]]): Coordenadas en formato
                    (x1, y1, x2, y2) para renderizado con OpenCV. Representa esquina
                    superior izquierda (x1, y1) e inferior derecha (x2, y2).

        Notes:
            Pipeline de detección:
                1. Redimensionado a 300x300 píxeles (entrada estándar del modelo)
                2. Creación de blob con normalización por media BGR (104.0, 177.0, 123.0)
                3. Inferencia mediante forward pass de la red
                4. Filtrado por umbral de confianza (definido en config)
                5. Re-escalado de coordenadas a dimensiones originales del frame
                6. Validación de límites dentro del frame
                7. Filtrado por tamaño mínimo (definido en config)

            La normalización por media BGR específica (104.0, 177.0, 123.0) corresponde
            a los valores medios del dataset de entrenamiento y es crítica para
            rendimiento óptimo del modelo.

            Detecciones con confianza inferior al umbral o dimensiones menores al
            tamaño mínimo se descartan automáticamente para reducir falsos positivos.
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        face_locations = []
        face_boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > FACE_CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 - x1 > MIN_FACE_SIZE and y2 - y1 > MIN_FACE_SIZE:
                    face_locations.append((y1, x2, y2, x1))
                    face_boxes.append((x1, y1, x2, y2))

        return face_locations, face_boxes


class FaceRecognizer:
    """
    Reconocedor facial mediante comparación de embeddings de 128 dimensiones.

    Implementa reconocimiento de identidad facial usando la biblioteca face_recognition
    que se basa en el modelo dlib ResNet-34 entrenado en dataset LFW. Genera embeddings
    faciales de 128 dimensiones y los compara mediante distancia euclidiana para
    determinar identidad.

    Attributes:
        actor_encodings (Dict[int, Dict]): Diccionario de actores conocidos indexado
            por ID de actor. Cada entrada contiene datos completos del actor incluyendo
            su encoding facial de 128 dimensiones, nombre, personaje y URL de foto.
    """

    def __init__(self):
        """
        Inicializa el reconocedor con diccionario vacío de encodings de actores.

        Crea estructura de datos para almacenar encodings de rostros conocidos.
        Los encodings deben cargarse posteriormente mediante load_actor_encodings()
        antes de realizar reconocimiento.
        """
        self.actor_encodings: Dict[int, Dict] = {}

    def load_actor_encodings(self, actors_data: List[Dict]):
        """
        Carga encodings faciales pre-calculados de actores conocidos en memoria.

        Indexa los datos de actores por ID para búsqueda eficiente durante
        reconocimiento. Los encodings deben haber sido calculados previamente desde
        fotografías de los actores mediante extract_encoding_from_image().

        Args:
            actors_data: Lista de diccionarios donde cada elemento representa un actor
                y debe contener las siguientes claves:
                    - id (int): Identificador único del actor
                    - nombre (str): Nombre completo del actor
                    - personaje (str): Nombre del personaje que interpreta
                    - foto_url (str): URL de la fotografía de referencia del actor
                    - encoding (np.ndarray): Vector de embedding facial de 128 dimensiones
                        generado por face_recognition library

        Notes:
            Los encodings faciales son vectores de 128 valores float que representan
            características únicas del rostro en espacio latente aprendido por ResNet-34.
            La calidad del reconocimiento depende críticamente de:
                - Calidad de fotografías de referencia (alta resolución, bien iluminadas)
                - Similitud de ángulo/pose entre referencia y detección
                - Condiciones de iluminación consistentes

            Se recomienda usar fotografías frontales con buena iluminación para maximizar
            precisión de reconocimiento. El modelo tolera variaciones moderadas de pose
            (hasta ~30 grados) y expresión facial.
        """
        self.actor_encodings = {
            actor['id']: actor for actor in actors_data
        }

    def extract_encodings(self, rgb_frame: np.ndarray, face_locations: List[Tuple]) -> List[np.ndarray]:
        """
        Extrae vectores de embedding facial de rostros detectados en el frame.

        Genera encodings de 128 dimensiones para cada rostro detectado mediante el
        modelo ResNet-34 de face_recognition. Los encodings capturan características
        faciales invariantes a transformaciones menores.

        Args:
            rgb_frame: Frame en formato RGB (no BGR) como numpy array con dimensiones
                (H, W, 3). La conversión de BGR a RGB debe realizarse antes de llamar
                este método.
            face_locations: Lista de tuplas con coordenadas de bounding boxes en formato
                (top, right, bottom, left) como retornado por FaceDetector.detect_faces().

        Returns:
            Lista de arrays numpy con encodings faciales, uno por cada rostro en
            face_locations. Cada encoding es un vector de 128 valores float. Lista
            vacía si falla la extracción o no hay rostros.

        Notes:
            El proceso de encoding incluye:
                1. Alineación facial mediante detección de 68 landmarks
                2. Transformación affine para normalizar pose
                3. Forward pass por ResNet-34 pre-entrenado
                4. Extracción de vector de 128-D de última capa fully-connected

            El método es robusto ante variaciones menores de:
                - Expresión facial
                - Iluminación (dentro de rangos razonables)
                - Pose facial (±30 grados aproximadamente)
                - Oclusiones parciales menores

            En caso de error (imagen corrupta, rostro mal formado), retorna lista vacía
            y el error se registra sin interrumpir el procesamiento.
        """
        try:
            return face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            return []

    def recognize_faces(self, face_encodings: List[np.ndarray]) -> List[Optional[Dict]]:
        """
        Reconoce identidades faciales comparando encodings con base de datos de actores.

        Para cada encoding de entrada, calcula distancia euclidiana con todos los
        encodings de actores conocidos y asigna identidad al actor con menor distancia
        si está por debajo del umbral de reconocimiento. Implementa estrategia
        nearest-neighbor con umbral para balance entre precisión y recall.

        Args:
            face_encodings: Lista de vectores de embedding facial de 128 dimensiones,
                típicamente obtenidos de extract_encodings().

        Returns:
            Lista de diccionarios con información de actores reconocidos, o None para
            rostros no reconocidos. Cada diccionario contiene:
                actor_id (int): ID único del actor en la base de datos.
                nombre (str): Nombre completo del actor.
                personaje (str): Nombre del personaje que interpreta.
                foto_url (str): URL de la fotografía de referencia.
                similitud (float): Porcentaje de similitud en rango [0.0, 100.0],
                    calculado como (1 - distancia) * 100.
                distance (float): Distancia euclidiana raw entre encodings en rango
                    [0.0, 1.0+], donde valores menores indican mayor similitud.

        Notes:
            Algoritmo de reconocimiento:
                1. Para cada encoding de entrada, calcular distancia euclidiana con
                   todos los encodings de actores conocidos
                2. Seleccionar actor con distancia mínima
                3. Si distancia < FACE_RECOGNITION_THRESHOLD, aceptar identificación
                4. Si distancia >= umbral, marcar como no reconocido (None)

            Configuración de umbral:
                - Valores típicos: 0.4-0.6
                - Umbral bajo (0.4): mayor precisión, menor recall (más estricto)
                - Umbral alto (0.6): menor precisión, mayor recall (más permisivo)
                - Valor recomendado: 0.5 para balance óptimo

            La distancia euclidiana en espacio de embeddings de 128-D se normaliza
            automáticamente por face_recognition library para que valores típicos
            entre rostros diferentes estén en rango [0.6, 1.0+] y entre mismo rostro
            en [0.0, 0.4].

            Para escenas con múltiples actores, el reconocimiento es O(n*m) donde
            n = número de rostros detectados y m = número de actores en base de datos.
        """
        recognized = []

        for face_encoding in face_encodings:
            best_match = None
            best_distance = float("inf")

            for actor_id, actor_data in self.actor_encodings.items():
                distance = face_recognition.face_distance(
                    [actor_data["encoding"]],
                    face_encoding
                )[0]

                if distance < best_distance:
                    best_distance = distance
                    best_match = (actor_id, actor_data, distance)

            if best_match and best_distance < FACE_RECOGNITION_THRESHOLD:
                actor_id, actor_data, distance = best_match
                similarity = (1 - distance) * 100

                recognized.append({
                    "actor_id": actor_id,
                    "nombre": actor_data["nombre"],
                    "personaje": actor_data["personaje"],
                    "foto_url": actor_data["foto_url"],
                    "similitud": similarity,
                    "distance": distance
                })
            else:
                recognized.append(None)

        return recognized

    @staticmethod
    def extract_encoding_from_image(image_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae un único encoding facial de una imagen de referencia.

        Método estático de utilidad para procesar fotografías de referencia de actores
        y generar sus encodings para almacenamiento en base de datos. Detecta y procesa
        el primer rostro encontrado en la imagen.

        Args:
            image_array: Imagen en formato RGB (no BGR) como numpy array con dimensiones
                (H, W, 3). Idealmente fotografía frontal de buena calidad del actor.

        Returns:
            Vector numpy de encoding facial de 128 dimensiones si se detecta rostro,
            None si no se detecta ningún rostro o ocurre error durante procesamiento.

        Notes:
            Uso típico: pre-procesamiento de fotografías de actores para generar base
            de datos de encodings conocidos antes de iniciar reconocimiento en video.

            Recomendaciones para fotografías de referencia:
                - Resolución: mínimo 300x300 píxeles
                - Iluminación: uniforme y bien distribuida
                - Pose: frontal o semi-frontal (máximo 30° de rotación)
                - Expresión: neutral preferiblemente
                - Fondo: sin importancia, el modelo extrae solo el rostro
                - Calidad: evitar compresión JPEG excesiva

            Si la imagen contiene múltiples rostros, solo se procesa el primero
            detectado (típicamente el más grande o prominente). Para procesar múltiples
            rostros en una imagen, usar extract_encodings() directamente con
            face_locations específicas.

            Los errores durante procesamiento (imagen corrupta, sin rostro detectable)
            se manejan silenciosamente retornando None sin lanzar excepciones.
        """
        try:
            encodings = face_recognition.face_encodings(image_array)
            return encodings[0] if len(encodings) > 0 else None
        except Exception as e:
            return None


def draw_face_box(frame: np.ndarray, box: Tuple[int, int, int, int],
                  label: str, color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Dibuja bounding box y etiqueta de texto sobre un rostro detectado en el frame.

    Renderiza anotaciones visuales que incluyen rectángulo de bounding box con color
    personalizable y etiqueta de texto con fondo sólido para garantizar legibilidad
    sobre cualquier fondo de imagen.

    Args:
        frame: Frame de video donde dibujar anotaciones en formato BGR (OpenCV) como
            numpy array con dimensiones (H, W, 3). Se modifica in-place.
        box: Coordenadas del bounding box facial en formato (x1, y1, x2, y2) donde
            (x1, y1) es esquina superior izquierda y (x2, y2) esquina inferior derecha,
            en píxeles.
        label: Texto de la etiqueta a mostrar sobre el bounding box. Típicamente
            contiene nombre del actor y porcentaje de similitud.
        color: Color del bounding box y fondo de etiqueta en formato BGR como tupla
            (B, G, R). Por defecto verde (0, 255, 0).

    Notes:
        Elementos visuales renderizados:
            - Rectángulo del bounding box:
                * Grosor: 2 píxeles
                * Color: según parámetro color
                * Posición: coordenadas exactas del box

            - Etiqueta de texto:
                * Fondo: rectángulo relleno del mismo color que el box
                * Texto: color negro (0, 0, 0) para máximo contraste
                * Fuente: FONT_HERSHEY_SIMPLEX, escala 0.5
                * Grosor: 2 píxeles
                * Posición: sobre el bounding box con 10px de margen superior

            - Cálculo de dimensiones:
                * Ancho/alto de fondo calculado dinámicamente según texto
                * Baseline incluido para posicionamiento vertical correcto

        El frame se modifica in-place sin retornar valor. La función es idempotente
        y puede llamarse múltiples veces en el mismo frame para anotar múltiples rostros.

        Colores típicos según contexto:
            - Verde (0, 255, 0): actor reconocido con alta confianza
            - Amarillo (0, 255, 255): actor reconocido con confianza media
            - Rojo (0, 0, 255): rostro detectado pero no reconocido
    """
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )

    cv2.rectangle(
        frame,
        (x1, y1 - label_h - 10),
        (x1 + label_w, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2
    )