"""
emotion_detector.py

Módulo para detección y clasificación de emociones faciales mediante redes neuronales
convolucionales y análisis geométrico. Clasifica expresiones en 7 categorías emocionales
estándar utilizando modelos FER2013 pre-entrenados con fallback a análisis heurístico.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class EmotionDetector:
    """
    Detector de emociones faciales mediante aprendizaje profundo y análisis geométrico.

    Implementa un sistema híbrido de detección de emociones que utiliza redes neuronales
    convolucionales pre-entrenadas (modelo FER2013) como método principal, con análisis
    geométrico de características faciales como sistema de respaldo cuando el modelo no
    está disponible. Soporta clasificación en 7 categorías emocionales estándar.

    Attributes:
        EMOTIONS (List[str]): Lista de emociones detectables en orden correspondiente
            a las clases del modelo FER2013: ['Enfadado', 'Disgustado', 'Miedo',
            'Feliz', 'Neutral', 'Triste', 'Sorprendido'].
        model (Optional[keras.Model]): Modelo Keras cargado para clasificación mediante
            redes convolucionales. None si no se pudo cargar.
        model_loaded (bool): Indicador de si el modelo fue cargado exitosamente.
        model_num_classes (int): Número de clases de salida del modelo cargado.
            Por defecto 7 para compatibilidad con FER2013.
        input_size (Tuple[int, int]): Dimensiones de entrada requeridas por el modelo
            en formato (width, height). Por defecto (48, 48) píxeles.
    """

    EMOTIONS = ['Enfadado', 'Disgustado', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']

    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el detector de emociones y carga el modelo pre-entrenado.

        Configura el detector intentando cargar un modelo Keras desde la ruta especificada.
        Si el modelo no está disponible o falla la carga, el sistema configura análisis
        geométrico como método alternativo. Detecta automáticamente las dimensiones de
        entrada y número de clases del modelo.

        Args:
            model_path: Ruta completa al archivo del modelo Keras (.h5 o SavedModel).
                Si es None o el archivo no existe, el detector operará exclusivamente
                con análisis geométrico sin capacidades de aprendizaje profundo.

        Notes:
            Requisitos para uso de modelos CNN:
                - TensorFlow >= 2.0 instalado en el entorno
                - Modelo compatible con arquitectura FER2013 (7 clases emocionales)
                - Entrada en escala de grises de dimensiones configurables

            Durante la carga se suprimen warnings de TensorFlow mediante configuración
            del logger. Si el modelo tiene número de clases diferente a 7, se ajusta
            automáticamente la lista de emociones truncándola al número disponible.

            El modelo se carga sin compilación (compile=False) ya que solo se utiliza
            para inferencia, no para entrenamiento adicional.
        """
        self.model = None
        self.model_loaded = False
        self.model_num_classes = 7
        self.input_size = (48, 48)

        if model_path and os.path.exists(model_path):
            try:
                import tensorflow as tf
                tf.get_logger().setLevel('ERROR')

                from tensorflow import keras
                self.model = keras.models.load_model(model_path, compile=False)

                input_shape = self.model.input_shape
                if len(input_shape) >= 3:
                    height = input_shape[1]
                    width = input_shape[2]
                    self.input_size = (width, height)

                output_shape = self.model.output_shape
                self.model_num_classes = output_shape[-1] if isinstance(output_shape, tuple) else 7

                if self.model_num_classes != 7:
                    if self.model_num_classes < 7:
                        self.EMOTIONS = self.EMOTIONS[:self.model_num_classes]

                self.model_loaded = True

            except ImportError:
                pass
            except Exception as e:
                pass

    def detect_emotion(self, face_region: np.ndarray, face_landmarks: Optional[np.ndarray] = None) -> Dict:
        """
        Detecta la emoción predominante en una región facial.

        Realiza clasificación de emociones utilizando el modelo de aprendizaje profundo
        si está disponible, o análisis geométrico como alternativa. Retorna la emoción
        con mayor confianza junto con la distribución completa de probabilidades para
        todas las clases emocionales.

        Args:
            face_region: Imagen recortada de la región facial en formato BGR (OpenCV)
                como numpy array con dimensiones (H, W, 3).
            face_landmarks: Array numpy opcional con coordenadas de puntos faciales
                landmarks para análisis geométrico. Solo utilizado cuando el modelo
                CNN no está disponible. None si no se requiere análisis geométrico.

        Returns:
            Diccionario con resultados completos de detección emocional:
                emotion (str): Emoción detectada con mayor probabilidad.
                confidence (float): Nivel de confianza de la predicción en rango
                    [0.0, 1.0] redondeado a 3 decimales.
                all_emotions (Dict[str, float]): Distribución de probabilidades para
                    todas las emociones posibles, mapeando nombre de emoción a
                    probabilidad redondeada a 3 decimales.
                method (str): Método utilizado para detección, valores posibles:
                    "keras_model" para CNN o "geometric_analysis" para análisis
                    heurístico.

        Notes:
            Si la región facial está vacía (size == 0) o el procesamiento falla,
            retorna emoción "Neutral" con confianza 1.0 por defecto. El método
            selecciona automáticamente entre detección por CNN (si modelo disponible)
            o análisis geométrico basándose en el estado de carga del modelo.
        """
        if self.model_loaded and self.model is not None:
            return self._detect_with_model(face_region)
        else:
            return self._detect_with_geometry(face_region, face_landmarks)

    def _detect_with_model(self, face_region: np.ndarray) -> Dict:
        """
        Detecta emoción mediante red neuronal convolucional pre-entrenada.

        Preprocesa la región facial mediante pipeline estándar (redimensionado a
        dimensiones del modelo, conversión a escala de grises, normalización a
        rango [0.0, 1.0]) y realiza inferencia con el modelo Keras cargado. Incluye
        múltiples validaciones de seguridad para prevenir errores dimensionales.

        Args:
            face_region: Región facial en formato BGR (OpenCV) como numpy array.

        Returns:
            Diccionario con emoción detectada, nivel de confianza, distribución
            completa de probabilidades y método utilizado. En caso de error durante
            inferencia, realiza fallback automático a detección geométrica.

        Notes:
            Pipeline de preprocesamiento:
                1. Redimensionado a self.input_size usando interpolación bilineal
                2. Conversión BGR a escala de grises
                3. Normalización: valores_píxel / 255.0
                4. Reshape a formato batch: (1, height, width, 1)

            La normalización es crítica para compatibilidad con modelos entrenados
            en FER2013 que esperan entrada en rango [0.0, 1.0]. La inferencia se
            ejecuta con verbose=0 para suprimir salida de progreso en consola.

            Validaciones de seguridad:
                - Verificación de dimensiones de predicción vs clases del modelo
                - Validación de índice de emoción dentro de rango válido
                - Manejo de arrays vacíos
        """
        try:
            if face_region.size == 0:
                return self._get_neutral_emotion()

            face_resized = cv2.resize(face_region, self.input_size)
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_normalized = face_gray.astype('float32') / 255.0

            face_reshaped = face_normalized.reshape(1, self.input_size[1], self.input_size[0], 1)

            predictions = self.model.predict(face_reshaped, verbose=0)

            if predictions.shape[-1] != self.model_num_classes:
                return self._detect_with_geometry(face_region)

            emotion_idx = int(np.argmax(predictions[0]))

            if emotion_idx >= len(self.EMOTIONS):
                emotion_idx = len(self.EMOTIONS) - 1

            confidence = float(predictions[0][emotion_idx])

            emotion_scores = {}
            for i in range(min(len(self.EMOTIONS), predictions.shape[-1])):
                emotion_scores[self.EMOTIONS[i]] = float(predictions[0][i])

            for i in range(predictions.shape[-1], len(self.EMOTIONS)):
                emotion_scores[self.EMOTIONS[i]] = 0.0

            return {
                "emotion": self.EMOTIONS[emotion_idx],
                "confidence": round(confidence, 3),
                "all_emotions": {k: round(v, 3) for k, v in emotion_scores.items()},
                "method": "keras_model"
            }

        except Exception as e:
            return self._detect_with_geometry(face_region)

    def _get_neutral_emotion(self) -> Dict:
        """
        Genera respuesta por defecto con emoción neutral.

        Utilizado como valor de retorno seguro cuando no es posible determinar la
        emoción por otros métodos (imagen vacía, errores de procesamiento).

        Returns:
            Diccionario con emoción "Neutral" y confianza 1.0, más distribución
            donde todas las emociones tienen probabilidad 0.0 excepto Neutral con 1.0.

        Notes:
            Este método garantiza siempre un retorno válido y consistente con la
            estructura esperada de resultados de detección emocional.
        """
        return {
            "emotion": "Neutral",
            "confidence": 1.0,
            "all_emotions": {emotion: (1.0 if emotion == "Neutral" else 0.0) for emotion in self.EMOTIONS},
            "method": "default"
        }

    def _detect_with_geometry(self, face_region: np.ndarray,
                              face_landmarks: Optional[np.ndarray] = None) -> Dict:
        """
        Detecta emoción mediante análisis geométrico de características faciales.

        Sistema de fallback que analiza características geométricas y texturales de
        diferentes regiones faciales (boca, ojos, cejas) para inferir emoción mediante
        heurísticas. Menos preciso que CNN pero funciona sin requerir modelos entrenados.

        Args:
            face_region: Región facial en formato BGR.
            face_landmarks: Puntos faciales opcionales (no utilizados actualmente pero
                reservados para expansión futura con análisis de landmarks).

        Returns:
            Diccionario con emoción inferida, confianza estimada basada en certeza
            de las características detectadas, distribución de probabilidades y método
            identificado como "geometric_analysis".

        Notes:
            El análisis geométrico evalúa:
                - Región de boca: curvatura (sonrisa vs ceño), densidad de bordes,
                  apertura estimada
                - Región de ojos: apertura ocular, fruncimiento de cejas mediante
                  gradientes

            Mapeo heurístico de características a emociones:
                - Curvatura positiva boca + ojos abiertos → Feliz
                - Curvatura negativa boca + cejas fruncidas → Enfadado/Triste
                - Alta densidad bordes boca + ojos muy abiertos → Sorprendido
                - Baja apertura ojos + curvatura neutral → Disgustado
                - Ojos muy abiertos + cejas levantadas → Miedo

            La confianza se calcula como función de la magnitud de las características
            detectadas, normalizada a rango [0.0, 1.0]. Típicamente produce confianzas
            entre 0.3-0.7 por la naturaleza heurística del método.
        """
        if face_region.size == 0:
            return self._get_neutral_emotion()

        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            mouth_features = self._analyze_mouth_region(gray)
            eye_features = self._analyze_eyes_region(gray)

            curvature = mouth_features['curvature']
            mouth_openness = mouth_features['openness']
            eye_openness = eye_features['openness']
            brow_furrow = eye_features['brow_furrow']

            scores = {
                'Feliz': 0.0,
                'Triste': 0.0,
                'Enfadado': 0.0,
                'Sorprendido': 0.0,
                'Miedo': 0.0,
                'Disgustado': 0.0,
                'Neutral': 0.3
            }

            if curvature > 0.3:
                scores['Feliz'] = min(1.0, curvature + eye_openness * 0.3)
            elif curvature < -0.3:
                if brow_furrow > 0.5:
                    scores['Enfadado'] = min(1.0, abs(curvature) + brow_furrow * 0.4)
                else:
                    scores['Triste'] = min(1.0, abs(curvature) + (1 - eye_openness) * 0.3)

            if mouth_openness > 0.6 and eye_openness > 0.7:
                scores['Sorprendido'] = min(1.0, mouth_openness * 0.7 + eye_openness * 0.3)

            if eye_openness < 0.3 and abs(curvature) < 0.2:
                scores['Disgustado'] = min(1.0, (1 - eye_openness) * 0.6 + mouth_features['edge_density'] * 0.4)

            if eye_openness > 0.8 and brow_furrow < 0.3:
                scores['Miedo'] = min(1.0, eye_openness * 0.6 + (1 - brow_furrow) * 0.4)

            emotion = max(scores, key=scores.get)
            confidence = scores[emotion]

            all_emotions = {k: round(v, 3) for k, v in scores.items()}

            return {
                "emotion": emotion,
                "confidence": round(confidence, 3),
                "all_emotions": all_emotions,
                "method": "geometric_analysis"
            }

        except Exception as e:
            return self._get_neutral_emotion()

    def _analyze_mouth_region(self, gray: np.ndarray) -> Dict:
        """
        Analiza características geométricas y texturales de la región bucal.

        Evalúa curvatura de sonrisa/ceño mediante análisis de gradientes horizontales,
        calcula densidad de bordes para inferir tensión muscular, y estima apertura
        mediante detección de regiones oscuras (cavidad bucal).

        Args:
            gray: Imagen facial completa en escala de grises como numpy array 2D.

        Returns:
            Diccionario con métricas de región bucal:
                edge_density (float): Densidad normalizada de bordes detectados [0-1].
                    Valores altos indican tensión o expresión marcada.
                curvature (float): Curvatura estimada [-1, 1] donde valores positivos
                    indican sonrisa (comisuras arriba) y negativos ceño (comisuras abajo).
                openness (float): Apertura bucal estimada [0-1] basada en área de
                    regiones oscuras en la mitad inferior de la región.

        Notes:
            La región analizada comprende el 50%-90% de la altura facial (mitad inferior).

            Cálculo de curvatura:
                - Divide región en mitad superior e inferior
                - Compara intensidad de gradientes horizontales entre mitades
                - Gradientes más fuertes arriba → sonrisa (curvatura positiva)
                - Gradientes más fuertes abajo → ceño (curvatura negativa)

            Detección de apertura:
                - Umbralización en 70 para aislar regiones oscuras (cavidad bucal)
                - Mide área de píxeles oscuros en mitad inferior de región
                - Normaliza por tamaño total de región analizada

            La densidad de bordes se calcula mediante detector Canny con umbrales 50-150
            y se normaliza dividiendo entre tamaño de región.
        """
        h, w = gray.shape
        mouth_region = gray[int(h * 0.5):int(h * 0.9), :]

        if mouth_region.size == 0:
            return {'edge_density': 0.0, 'curvature': 0.0, 'openness': 0.0}

        edges = cv2.Canny(mouth_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        grad_x = cv2.Sobel(mouth_region, cv2.CV_64F, 1, 0, ksize=3)

        upper_half = grad_x[:grad_x.shape[0] // 2, :]
        lower_half = grad_x[grad_x.shape[0] // 2:, :]

        upper_intensity = np.mean(np.abs(upper_half))
        lower_intensity = np.mean(np.abs(lower_half))

        curvature = (upper_intensity - lower_intensity) / (upper_intensity + lower_intensity + 1)

        lower_region = mouth_region[mouth_region.shape[0] // 2:, :]
        _, thresh = cv2.threshold(lower_region, 70, 255, cv2.THRESH_BINARY_INV)
        dark_area = np.sum(thresh > 0) / thresh.size if thresh.size > 0 else 0.0

        return {
            'edge_density': float(edge_density),
            'curvature': float(np.clip(curvature, -1, 1)),
            'openness': float(dark_area * 2)
        }

    def _analyze_eyes_region(self, gray: np.ndarray) -> Dict:
        """
        Analiza características geométricas de la región ocular.

        Evalúa apertura ocular mediante detección de áreas oscuras (pupilas y sombras
        oculares) y analiza el fruncimiento de cejas mediante cálculo de ratio entre
        gradientes verticales y horizontales en la región superior del rostro.

        Args:
            gray: Imagen facial completa en escala de grises como numpy array 2D.

        Returns:
            Diccionario con métricas de región ocular:
                openness (float): Apertura ocular estimada [0-1] donde 0 indica ojos
                    cerrados y 1 ojos muy abiertos. Calculado invirtiendo área oscura.
                brow_furrow (float): Intensidad de fruncimiento de cejas [0-1] donde
                    valores altos (>0.5) sugieren enfado, concentración o preocupación.

        Notes:
            La región analizada comprende del 20% al 50% de la altura facial (zona
            superior donde se encuentran ojos y cejas).

            Cálculo de apertura ocular:
                - Umbralización en 70 para detectar regiones oscuras (pupilas)
                - Área oscura normalizada se invierte: apertura = 1 - área_oscura
                - Asume que ojos cerrados tienen más área oscura que ojos abiertos

            Cálculo de fruncimiento de cejas:
                - Gradientes Sobel en direcciones X (vertical) e Y (horizontal)
                - Ratio = gradientes_verticales / (gradientes_horizontales + 1)
                - Valores altos indican líneas verticales prominentes (cejas fruncidas)
                - Se normaliza dividiendo por 2 y limitando a rango [0, 1]

            Si la región está vacía, retorna valores por defecto: apertura 0.5 (neutral)
            y fruncimiento 0.0 (sin fruncir).
        """
        h, w = gray.shape

        eyes_region = gray[int(h * 0.2):int(h * 0.5), :]

        if eyes_region.size == 0:
            return {'openness': 0.5, 'brow_furrow': 0.0}

        _, thresh = cv2.threshold(eyes_region, 70, 255, cv2.THRESH_BINARY_INV)
        dark_area = np.sum(thresh > 0) / thresh.size if thresh.size > 0 else 0.0

        grad_x = cv2.Sobel(eyes_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(eyes_region, cv2.CV_64F, 0, 1, ksize=3)

        vertical_edges = np.mean(np.abs(grad_x))
        horizontal_edges = np.mean(np.abs(grad_y))

        brow_furrow = vertical_edges / (horizontal_edges + 1)

        return {
            'openness': float(1 - dark_area),
            'brow_furrow': float(np.clip(brow_furrow / 2, 0, 1))
        }

    def detect_emotions_batch(self, face_regions: List[np.ndarray]) -> List[Dict]:
        """
        Detecta emociones para múltiples rostros en procesamiento batch.

        Procesa secuencialmente una lista de regiones faciales, aplicando detección
        de emociones a cada una mediante el método configurado (CNN o geométrico).
        Útil para análisis de escenas con múltiples personas.

        Args:
            face_regions: Lista de imágenes recortadas de rostros en formato BGR,
                cada elemento es un numpy array independiente representando una cara.

        Returns:
            Lista de diccionarios con resultados de detección, uno por cada rostro
            en el mismo orden de entrada. Cada diccionario contiene emoción, confianza,
            distribución de probabilidades y método utilizado.

        Notes:
            El procesamiento es secuencial (no paralelo) por simplicidad y para evitar
            problemas de concurrencia con modelos Keras. Para grandes volúmenes de
            rostros (>100) considerar implementación paralela usando multiprocessing
            con múltiples instancias del modelo o procesamiento en GPU con batching
            nativo de TensorFlow.

            Todos los rostros en el batch se procesan con el mismo método (CNN o
            geométrico) según disponibilidad del modelo cargado.
        """
        results = []
        for face_region in face_regions:
            emotion_data = self.detect_emotion(face_region)
            results.append(emotion_data)
        return results


def draw_emotion_label(frame: np.ndarray, box: Tuple[int, int, int, int],
                       emotion_data: Dict) -> np.ndarray:
    """
    Dibuja etiqueta de emoción sobre un rostro detectado en el frame.

    Superpone anotaciones visuales que incluyen un rectángulo de bounding box con
    color codificado según la emoción detectada, y una etiqueta de texto con el
    nombre de la emoción y su porcentaje de confianza. El fondo de la etiqueta
    es sólido para garantizar legibilidad.

    Args:
        frame: Frame de video donde dibujar las anotaciones en formato BGR (OpenCV)
            como numpy array con dimensiones (H, W, 3). Se modifica in-place.
        box: Coordenadas del bounding box facial en formato (x1, y1, x2, y2) donde
            (x1, y1) es la esquina superior izquierda y (x2, y2) la inferior derecha.
        emotion_data: Diccionario con resultados de detección emocional, debe contener
            al menos las claves 'emotion' (str) y 'confidence' (float).

    Returns:
        Frame modificado con anotaciones visuales superpuestas. El array se modifica
        in-place pero también se retorna para encadenamiento de funciones.

    Notes:
        Codificación de colores por emoción en formato BGR:
            - Feliz: verde brillante (0, 255, 0)
            - Triste: azul (255, 0, 0)
            - Enfadado: rojo (0, 0, 255)
            - Sorprendido: amarillo (255, 255, 0)
            - Miedo: magenta (255, 0, 255)
            - Disgustado: púrpura oscuro (128, 0, 128)
            - Neutral: gris claro (200, 200, 200)
            - Default: blanco (255, 255, 255) para emociones no reconocidas

        Elementos visuales:
            - Rectángulo del bounding box: grosor 2 píxeles, color según emoción
            - Etiqueta de texto: formato "Emoción (XX%)" donde XX es confidence
            - Fondo de etiqueta: rectángulo relleno del mismo color que el bounding box
            - Texto: color negro (0, 0, 0) con grosor 2 para contraste
            - Posición: sobre el bounding box con margen de 10 píxeles

        La confianza se formatea como porcentaje entero sin decimales para simplificar
        visualización.
    """
    x1, y1, x2, y2 = box

    emotion = emotion_data['emotion']
    confidence = emotion_data['confidence']
    label = f"{emotion} ({confidence:.0%})"

    color_map = {
        'Feliz': (0, 255, 0),
        'Triste': (255, 0, 0),
        'Enfadado': (0, 0, 255),
        'Sorprendido': (255, 255, 0),
        'Miedo': (255, 0, 255),
        'Disgustado': (128, 0, 128),
        'Neutral': (200, 200, 200)
    }
    color = color_map.get(emotion, (255, 255, 255))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (label_w, label_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame,
        (x1, y1 - label_h - 10),
        (x1 + label_w, y1),
        color,
        -1
    )

    cv2.putText(
        frame, label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 0), 2
    )

    return frame


def visualize_emotion_distribution(emotion_data: Dict, width: int = 300,
                                   height: int = 200) -> np.ndarray:
    """
    Genera visualización de gráfico de barras para distribución de emociones.

    Crea una imagen con gráfico de barras verticales mostrando las probabilidades
    relativas de cada emoción detectada. Las barras se colorean según un gradiente
    que representa nivel de confianza, facilitando identificación visual rápida de
    la emoción predominante y alternativas plausibles.

    Args:
        emotion_data: Diccionario con distribución de emociones. Debe contener la
            clave 'all_emotions' mapeando nombres de emociones a probabilidades.
        width: Ancho de la imagen de visualización en píxeles. Por defecto 300px.
            Se divide equitativamente entre todas las emociones.
        height: Alto de la imagen de visualización en píxeles. Por defecto 200px.
            Las barras se escalan proporcionalmente usando los 20px inferiores para
            etiquetas.

    Returns:
        Imagen RGB con gráfico de barras como numpy array con dimensiones
        (height, width, 3) y dtype uint8. Retorna imagen completamente negra si
        emotion_data no contiene 'all_emotions' o está vacío.

    Notes:
        Características visuales del gráfico:
            - Fondo: negro (0, 0, 0)
            - Barras: ancho = width / num_emociones con separación de 5px
            - Color de barras: gradiente verde→rojo basado en score
                * Score alto (cerca de 1.0): verde brillante
                * Score bajo (cerca de 0.0): rojo
                * Fórmula RGB: (0, 255*score, 255*(1-score))
            - Etiquetas: texto blanco (255, 255, 255), primeros 3 caracteres
              de cada emoción, posicionadas en los 20px inferiores
            - Altura de barras: normalizada respecto a emoción con mayor score
              para maximizar uso del espacio vertical disponible

        Las etiquetas se abrevian a 3 caracteres para optimizar espacio horizontal
        cuando hay muchas emociones. La normalización relativa (no absoluta) mejora
        la visualización al escalar automáticamente las barras al espacio disponible
        independientemente de si los scores son muy bajos o altos.

        Útil para debugging, análisis de confianza del modelo, y visualización en
        tiempo real de distribuciones emocionales ambiguas.
    """
    viz = np.zeros((height, width, 3), dtype=np.uint8)

    if 'all_emotions' not in emotion_data:
        return viz

    emotions = emotion_data['all_emotions']
    num_emotions = len(emotions)

    if num_emotions == 0:
        return viz

    bar_width = width // num_emotions
    max_score = max(emotions.values()) if emotions else 1

    for i, (emotion, score) in enumerate(emotions.items()):
        x = i * bar_width
        bar_height = int((score / max_score) * (height - 30))
        y = height - bar_height - 20

        color = (0, int(255 * score), int(255 * (1 - score)))
        cv2.rectangle(viz, (x + 5, y), (x + bar_width - 5, height - 20), color, -1)

        cv2.putText(
            viz, emotion[:3],
            (x + 10, height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, (255, 255, 255), 1
        )

    return viz