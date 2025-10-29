"""
face_detection.py

Módulo para detección y reconocimiento facial mediante múltiples algoritmos.
Implementa Viola-Jones (Haar Cascades) para detección rápida clásica y DNN ResNet-10
SSD para detección moderna de alta precisión. Incluye reconocimiento mediante
comparación de embeddings faciales de 128 dimensiones.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Detección facial mediante Haar Cascades y DNN
    - numpy: Operaciones con arrays
    - face_recognition: Generación de encodings y reconocimiento facial

Usage:
    from app.analysis.face_detection import FaceDetector, FaceRecognizer

    detector = FaceDetector()
    face_locations, face_boxes = detector.detect_faces(frame)

    recognizer = FaceRecognizer()
    recognizer.load_actor_encodings(actors_data)
    recognized = recognizer.recognize_faces(face_encodings)
"""

import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Dict, Optional
from ..config import (
    FACE_DETECTION_METHOD,
    FACE_DETECTION_PROTOTXT,
    FACE_DETECTION_WEIGHTS,
    HAAR_CASCADE_PATH,
    HAAR_CASCADE_PROFILE_PATH,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_RECOGNITION_THRESHOLD,
    MIN_FACE_SIZE,
    VIOLA_JONES_SCALE_FACTOR,
    VIOLA_JONES_MIN_NEIGHBORS
)


class FaceDetectorViolaJones:
    """
    Detector de rostros mediante algoritmo clásico de Viola-Jones (2001).

    Implementa detección facial usando Haar Cascades, técnica basada en
    características de Haar-like y clasificador en cascada de AdaBoost.
    Detecta tanto rostros frontales como perfiles laterales.

    Ventajas:
        - Muy rápido (50-100 FPS en CPU moderna)
        - Bajo consumo de memoria (~1MB modelo)
        - No requiere GPU
        - Funciona bien en rostros frontales y perfiles

    Desventajas:
        - Menor precisión que DNN en ángulos oblicuos extremos
        - Sensible a iluminación extrema
        - Más falsos positivos/negativos que métodos modernos

    Attributes:
        cascade_frontal: Clasificador para rostros frontales
        cascade_profile: Clasificador para rostros de perfil
    """

    def __init__(self):
        """
        Inicializa detectores cargando clasificadores Haar Cascade.

        Carga dos modelos pre-entrenados de OpenCV:
        1. haarcascade_frontalface_default.xml - Para rostros frontales
        2. haarcascade_profileface.xml - Para rostros de perfil

        Raises:
            FileNotFoundError: Si algún archivo XML no existe en ruta especificada
            cv2.error: Si hay error durante carga de clasificadores

        Notes:
            Ambos clasificadores utilizan cascadas de AdaBoost con características
            Haar-like. El detector de perfil también funciona para perfiles invertidos
            mediante flip horizontal.
        """
        from ..config import HAAR_CASCADE_PROFILE_PATH

        self.cascade_frontal = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

        if self.cascade_frontal.empty():
            raise FileNotFoundError(
                f"No se pudo cargar Haar Cascade frontal desde {HAAR_CASCADE_PATH}"
            )

        self.cascade_profile = cv2.CascadeClassifier(HAAR_CASCADE_PROFILE_PATH)

        if self.cascade_profile.empty():
            raise FileNotFoundError(
                f"No se pudo cargar Haar Cascade de perfil desde {HAAR_CASCADE_PROFILE_PATH}"
            )

    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detecta rostros frontales y de perfil en frame mediante clasificadores Haar Cascade.

        Aplica tres pasadas de detección:
        1. Rostros frontales
        2. Perfiles izquierdos
        3. Perfiles derechos mediante flip horizontal

        Elimina detecciones duplicadas mediante Non-Maximum Suppression.

        Args:
            frame: Frame de entrada en formato BGR como numpy array con
                dimensiones (H, W, 3)

        Returns:
            Tupla con dos listas de coordenadas:
                face_locations: Coordenadas en formato (top, right, bottom, left)
                    compatible con face_recognition library
                face_boxes: Coordenadas en formato (x1, y1, x2, y2) nativo de OpenCV

        Notes:
            Pipeline de detección multi-ángulo:
                1. Conversión a escala de grises
                2. Ecualización de histograma
                3. Detección de rostros frontales
                4. Detección de perfiles izquierdos
                5. Flip horizontal + detección de perfiles derechos
                6. Non-Maximum Suppression para eliminar duplicados
                7. Conversión a formatos de salida
                8. Filtrado por tamaño mínimo

            El NMS es crucial porque un mismo rostro puede ser detectado por
            múltiples clasificadores.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        all_faces = []

        # Detectar rostros frontales
        faces_frontal = self.cascade_frontal.detectMultiScale(
            gray,
            scaleFactor=VIOLA_JONES_SCALE_FACTOR,
            minNeighbors=VIOLA_JONES_MIN_NEIGHBORS,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces_frontal)

        # Detectar perfiles izquierdos
        faces_profile_left = self.cascade_profile.detectMultiScale(
            gray,
            scaleFactor=VIOLA_JONES_SCALE_FACTOR,
            minNeighbors=VIOLA_JONES_MIN_NEIGHBORS,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces_profile_left)

        # Detectar perfiles derechos mediante flip horizontal
        gray_flipped = cv2.flip(gray, 1)
        faces_profile_right = self.cascade_profile.detectMultiScale(
            gray_flipped,
            scaleFactor=VIOLA_JONES_SCALE_FACTOR,
            minNeighbors=VIOLA_JONES_MIN_NEIGHBORS,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Ajustar coordenadas del flip a coordenadas originales
        for (x, y, w, h) in faces_profile_right:
            x_original = frame.shape[1] - x - w
            all_faces.append((x_original, y, w, h))

        # Eliminar detecciones duplicadas mediante NMS
        all_faces = self._remove_overlapping_faces(all_faces)

        # Convertir a formatos de salida
        face_locations = []
        face_boxes = []

        for (x, y, w, h) in all_faces:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

            if (x2 - x1) > MIN_FACE_SIZE and (y2 - y1) > MIN_FACE_SIZE:
                face_locations.append((y1, x2, y2, x1))
                face_boxes.append((x1, y1, x2, y2))

        return face_locations, face_boxes

    def _remove_overlapping_faces(self, faces: List[Tuple],
                                  overlap_threshold: float = 0.5) -> List[Tuple]:
        """
        Elimina detecciones duplicadas mediante Non-Maximum Suppression (NMS).

        Cuando un mismo rostro es detectado por múltiples clasificadores, este método
        mantiene solo la detección más confiable y elimina las solapadas.

        Args:
            faces: Lista de tuplas (x, y, w, h) con todas las detecciones
            overlap_threshold: Umbral de IoU para considerar overlap. Por defecto 0.5

        Returns:
            Lista filtrada de tuplas (x, y, w, h) sin duplicados

        Notes:
            Algoritmo NMS:
                1. Calcular IoU (Intersection over Union) entre todas las cajas
                2. Ordenar cajas por área (las más grandes primero)
                3. Mantener caja con mayor área
                4. Eliminar todas las cajas con IoU > threshold
                5. Repetir hasta procesar todas

            IoU = Área_Intersección / Área_Unión
                - IoU = 0: cajas no se solapan
                - IoU = 1: cajas idénticas
                - IoU > 0.5: probablemente mismo rostro
        """
        if len(faces) == 0:
            return []

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = y2.argsort()
        keep = []

        while order.size > 0:
            i = order[-1]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[:-1]])
            yy1 = np.maximum(y1[i], y1[order[:-1]])
            xx2 = np.minimum(x2[i], x2[order[:-1]])
            yy2 = np.minimum(y2[i], y2[order[:-1]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[:-1]] - intersection)

            order = order[np.where(iou <= overlap_threshold)[0]]

        return [faces[i] for i in keep]


class FaceDetectorDNN:
    """
    Detector de rostros mediante red neuronal convolucional de OpenCV.

    Implementa detección facial usando el modelo ResNet SSD pre-entrenado de OpenCV
    DNN, optimizado para detección en tiempo real con alta precisión. El modelo
    utiliza arquitectura Single Shot Detector con backbone ResNet-10.

    Attributes:
        net: Red neuronal cargada desde archivos Caffe. Modelo ResNet-10 SSD
            entrenado en dataset WIDER FACE para detección facial
    """

    def __init__(self):
        """
        Inicializa el detector cargando el modelo DNN desde archivos de configuración.

        Carga modelo pre-entrenado ResNet-10 SSD desde archivos Caffe (arquitectura
        en prototxt y pesos en caffemodel). El modelo está optimizado para detectar
        rostros frontales y semi-frontales en diversas condiciones de iluminación.

        Raises:
            FileNotFoundError: Si los archivos de modelo no existen
            cv2.error: Si hay error durante la carga del modelo DNN

        Notes:
            Características del modelo ResNet-10 SSD:
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
        confianza y tamaño mínimo.

        Args:
            frame: Frame de entrada en formato BGR como numpy array con
                dimensiones (H, W, 3)

        Returns:
            Tupla con dos listas de coordenadas:
                face_locations: Coordenadas en formato (top, right, bottom, left)
                    compatible con face_recognition library
                face_boxes: Coordenadas en formato (x1, y1, x2, y2) para
                    renderizado con OpenCV

        Notes:
            Pipeline de detección:
                1. Redimensionado a 300x300 píxeles (entrada estándar del modelo)
                2. Creación de blob con normalización por media BGR (104.0, 177.0, 123.0)
                3. Inferencia mediante forward pass de la red
                4. Filtrado por umbral de confianza
                5. Re-escalado de coordenadas a dimensiones originales del frame
                6. Validación de límites dentro del frame
                7. Filtrado por tamaño mínimo

            La normalización por media BGR específica corresponde a los valores medios
            del dataset de entrenamiento y es crítica para rendimiento óptimo.
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


class FaceDetector:
    """
    Detector de rostros unificado con múltiples backends.

    Proporciona interfaz única para detección facial soportando dos algoritmos:
        - "viola-jones": Haar Cascades clásico (rápido, menos preciso)
        - "dnn": ResNet-10 SSD moderno (lento, más preciso)

    La selección de método se configura mediante variable FACE_DETECTION_METHOD
    en config.py.

    Attributes:
        method: Método de detección activo ("viola-jones" o "dnn")
        detector: Instancia del detector específico según método configurado
    """

    def __init__(self):
        """
        Inicializa detector según método configurado en FACE_DETECTION_METHOD.

        Raises:
            ValueError: Si FACE_DETECTION_METHOD no es válido
            FileNotFoundError: Si archivos de modelo necesarios no existen

        Notes:
            La inicialización carga el modelo en memoria. Para Viola-Jones
            esto implica ~1MB, para DNN ~10MB. Ambos métodos cargan una única
            vez al inicio y reutilizan el modelo para todas las detecciones.
        """
        self.method = FACE_DETECTION_METHOD.lower()

        if self.method == "viola-jones":
            self.detector = FaceDetectorViolaJones()
        elif self.method == "dnn":
            self.detector = FaceDetectorDNN()
        else:
            raise ValueError(
                f"Método de detección no válido: {FACE_DETECTION_METHOD}. "
                f"Opciones: 'viola-jones', 'dnn'"
            )

    def detect_faces(self, frame: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detecta rostros en frame usando método configurado.

        Interfaz unificada que delega a detector específico. La implementación
        interna difiere pero la salida mantiene formato consistente.

        Args:
            frame: Frame BGR como numpy array (H, W, 3)

        Returns:
            Tupla (face_locations, face_boxes) con coordenadas en dos formatos

        Notes:
            Comparación de rendimiento típico (imagen 720p):
                Viola-Jones: ~10-20ms por frame (50-100 FPS)
                DNN: ~50-100ms por frame (10-20 FPS)

            Precisión típica (AP@0.5 en WIDER FACE):
                Viola-Jones: ~70% (frontal), ~40% (oblicuo)
                DNN: ~90% (frontal), ~80% (oblicuo)
        """
        return self.detector.detect_faces(frame)

    def get_method(self) -> str:
        """
        Retorna nombre del método de detección activo.

        Returns:
            String "viola-jones" o "dnn"
        """
        return self.method


class FaceRecognizer:
    """
    Reconocedor facial mediante comparación de embeddings de 128 dimensiones.

    Implementa reconocimiento de identidad facial usando la biblioteca face_recognition
    basada en el modelo dlib ResNet-34 entrenado en dataset LFW. Genera embeddings
    faciales de 128 dimensiones y los compara mediante distancia euclidiana para
    determinar identidad.

    Attributes:
        actor_encodings: Diccionario de actores conocidos indexado por ID de actor.
            Cada entrada contiene datos completos del actor incluyendo su encoding
            facial de 128 dimensiones, nombre, personaje y URL de foto
    """

    def __init__(self):
        """
        Inicializa el reconocedor con diccionario vacío de encodings de actores.

        Crea estructura de datos para almacenar encodings de rostros conocidos.
        Los encodings deben cargarse posteriormente mediante load_actor_encodings()
        antes de realizar reconocimiento.
        """
        self.actor_encodings: Dict[int, Dict] = {}

    def add_known_face(
        self,
        encoding: np.ndarray,
        actor_id: int,
        nombre: str,
        personaje: str,
        foto_url: str
    ):
        """
        Añade un rostro conocido a la base de datos de reconocimiento.

        Método de conveniencia para agregar actores individuales sin necesidad
        de cargar toda una lista mediante load_actor_encodings(). Útil para
        construcción incremental de base de datos durante procesamiento de vídeo.

        Args:
            encoding: Vector de embedding facial de 128 dimensiones generado por
                face_recognition library
            actor_id: Identificador único del actor
            nombre: Nombre completo del actor
            personaje: Nombre del personaje que interpreta en el contenido
            foto_url: URL de la fotografía de referencia del actor

        Notes:
            Si ya existe un actor con el mismo actor_id, se sobrescribirá con
            los nuevos datos.
        """
        self.actor_encodings[actor_id] = {
            'id': actor_id,
            'nombre': nombre,
            'personaje': personaje,
            'foto_url': foto_url,
            'encoding': encoding
        }

    def load_actor_encodings(self, actors_data: List[Dict]):
        """
        Carga encodings faciales pre-calculados de actores conocidos en memoria.

        Indexa los datos de actores por ID para búsqueda eficiente durante
        reconocimiento. Los encodings deben haber sido calculados previamente desde
        fotografías de los actores mediante extract_encoding_from_image().

        Args:
            actors_data: Lista de diccionarios donde cada elemento representa un actor
                y debe contener las siguientes claves:
                    - id: Identificador único del actor
                    - nombre: Nombre completo del actor
                    - personaje: Nombre del personaje que interpreta
                    - foto_url: URL de la fotografía de referencia del actor
                    - encoding: Vector de embedding facial de 128 dimensiones

        Notes:
            Los encodings faciales son vectores de 128 valores float que representan
            características únicas del rostro en espacio latente aprendido por ResNet-34.
            La calidad del reconocimiento depende críticamente de:
                - Calidad de fotografías de referencia (alta resolución, bien iluminadas)
                - Similitud de ángulo/pose entre referencia y detección
                - Condiciones de iluminación consistentes

            Se recomienda usar fotografías frontales con buena iluminación para maximizar
            precisión de reconocimiento.
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
            rgb_frame: Frame en formato RGB (no BGR) como numpy array con
                dimensiones (H, W, 3). La conversión de BGR a RGB debe realizarse
                antes de llamar este método
            face_locations: Lista de tuplas con coordenadas de bounding boxes en
                formato (top, right, bottom, left) como retornado por
                FaceDetector.detect_faces()

        Returns:
            Lista de arrays numpy con encodings faciales, uno por cada rostro en
            face_locations. Cada encoding es un vector de 128 valores float. Lista
            vacía si falla la extracción o no hay rostros

        Notes:
            El proceso de encoding incluye:
                1. Alineación facial mediante detección de 68 landmarks
                2. Transformación affine para normalizar pose
                3. Forward pass por ResNet-34 pre-entrenado
                4. Extracción de vector de 128-D de última capa fully-connected

            El método es robusto ante variaciones menores de expresión facial,
            iluminación, pose facial y oclusiones parciales menores.
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
        si está por debajo del umbral de reconocimiento.

        Args:
            face_encodings: Lista de vectores de embedding facial de 128 dimensiones,
                típicamente obtenidos de extract_encodings()

        Returns:
            Lista de diccionarios con información de actores reconocidos, o None para
            rostros no reconocidos. Cada diccionario contiene:
                actor_id: ID único del actor en la base de datos
                nombre: Nombre completo del actor
                personaje: Nombre del personaje que interpreta
                foto_url: URL de la fotografía de referencia
                similitud: Porcentaje de similitud en rango [0.0, 100.0],
                    calculado como (1 - distancia) * 100
                distance: Distancia euclidiana raw entre encodings en rango
                    [0.0, 1.0+], donde valores menores indican mayor similitud

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
            image_array: Imagen en formato RGB (no BGR) como numpy array con
                dimensiones (H, W, 3). Idealmente fotografía frontal de buena calidad
                del actor

        Returns:
            Vector numpy de encoding facial de 128 dimensiones si se detecta rostro,
            None si no se detecta ningún rostro o ocurre error durante procesamiento

        Notes:
            Recomendaciones para fotografías de referencia:
                - Resolución: mínimo 300x300 píxeles
                - Iluminación: uniforme y bien distribuida
                - Pose: frontal o semi-frontal (máximo 30° de rotación)
                - Expresión: neutral preferiblemente
                - Calidad: evitar compresión JPEG excesiva

            Si la imagen contiene múltiples rostros, solo se procesa el primero
            detectado. Los errores durante procesamiento se manejan silenciosamente
            retornando None sin lanzar excepciones.
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
        frame: Frame de vídeo donde dibujar anotaciones en formato BGR como numpy
            array con dimensiones (H, W, 3). Se modifica in-place
        box: Coordenadas del bounding box facial en formato (x1, y1, x2, y2) donde
            (x1, y1) es esquina superior izquierda y (x2, y2) esquina inferior
            derecha, en píxeles
        label: Texto de la etiqueta a mostrar sobre el bounding box. Típicamente
            contiene nombre del actor y porcentaje de similitud
        color: Color del bounding box y fondo de etiqueta en formato BGR como tupla
            (B, G, R). Por defecto verde (0, 255, 0)

    Notes:
        Elementos visuales renderizados:
            - Rectángulo del bounding box: grosor 2 píxeles
            - Etiqueta de texto: fondo relleno del mismo color que el box
            - Texto: color negro (0, 0, 0) para máximo contraste
            - Fuente: FONT_HERSHEY_SIMPLEX, escala 0.5, grosor 2 píxeles
            - Posición: sobre el bounding box con 10px de margen superior

        El frame se modifica in-place. Colores típicos según contexto:
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