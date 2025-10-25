"""
schemas.py

Modelos Pydantic para validación de datos, serialización de respuestas API
y esquemas de eventos Server-Sent Events del sistema de análisis cinematográfico.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


# ==================== ENUMERACIONES ====================

class ContentType(str, Enum):
    """
    Tipo de contenido en base de datos TMDB.

    Values:
        MOVIE: Película cinematográfica.
        TV: Serie de televisión.
    """
    MOVIE = "movie"
    TV = "tv"


class AnalysisPriority(str, Enum):
    """
    Nivel de prioridad para procesamiento de análisis.

    Determina frecuencia de muestreo y profundidad de análisis:
        HIGH: Análisis de cada frame con máximo detalle.
        MEDIUM: Muestreo cada 2-3 frames con análisis estándar.
        LOW: Muestreo cada 5-10 frames con análisis básico.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MessageType(str, Enum):
    """
    Tipos de eventos en protocolo Server-Sent Events (SSE).

    Values:
        INFO: Información inicial de sesión y configuración.
        FRAME: Datos de análisis de frame individual.
        PROGRESS: Actualización de progreso de procesamiento.
        DONE: Resultados finales y resumen de análisis completo.
        ERROR: Notificación de error durante procesamiento.

    Notes:
        Los eventos SSE se transmiten desde servidor a cliente en un stream
        unidireccional HTTP persistente. Cada evento tiene formato:
            event: {tipo}
            data: {json}
    """
    INFO = "info"
    FRAME = "frame"
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"


# ==================== MODELOS BASE ====================

class VideoInfo(BaseModel):
    """
    Información técnica de archivo de video.

    Attributes:
        width: Ancho del video en píxeles.
        height: Alto del video en píxeles.
        fps: Frames por segundo (frame rate).
        total_frames: Número total de frames en el video.
        duration: Duración total en segundos.
        codec: Código FourCC del codec de video.
    """
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    codec: int

    class Config:
        json_schema_extra = {
            "example": {
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "total_frames": 15000,
                "duration": 500.0,
                "codec": 828601953
            }
        }


class VideoUploadResponse(BaseModel):
    """
    Respuesta tras subida exitosa de archivo de video.

    Attributes:
        success: Flag de éxito de operación.
        path: Ruta del archivo en servidor.
        filename: Nombre del archivo almacenado.
        size: Tamaño del archivo en bytes.
        video_info: Metadatos técnicos extraídos del video.
    """
    success: bool
    path: str
    filename: str
    size: int
    video_info: Optional[VideoInfo] = None


class ContentSearchResponse(BaseModel):
    """
    Respuesta de búsqueda de contenido en base de datos TMDB.

    Attributes:
        found: Indica si se encontró coincidencia.
        id: ID de TMDB del contenido.
        type: Tipo de contenido (película o serie).
        title: Título del contenido.
        overview: Sinopsis descriptiva.
        poster_path: Ruta relativa del poster en TMDB.
        poster_url: URL completa del poster.
        release_date: Fecha de estreno en formato ISO.
    """
    found: bool
    id: Optional[int] = None
    type: Optional[ContentType] = None
    title: Optional[str] = None
    overview: Optional[str] = None
    poster_path: Optional[str] = None
    poster_url: Optional[str] = None
    release_date: Optional[str] = None


# ==================== MODELOS DE ANÁLISIS ====================

class FaceBox(BaseModel):
    """
    Coordenadas de bounding box de rostro detectado.

    Attributes:
        x1: Coordenada X de esquina superior izquierda.
        y1: Coordenada Y de esquina superior izquierda.
        x2: Coordenada X de esquina inferior derecha.
        y2: Coordenada Y de esquina inferior derecha.
    """
    x1: int
    y1: int
    x2: int
    y2: int


class EmotionResult(BaseModel):
    """
    Resultado de clasificación emocional de expresión facial.

    Attributes:
        emotion: Emoción predominante detectada.
        confidence: Nivel de confianza normalizado [0.0, 1.0].
        all_emotions: Distribución completa de probabilidades por emoción.
        method: Método utilizado ("keras_model" o "geometric_analysis").
    """
    emotion: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_emotions: Optional[Dict[str, float]] = None
    method: str = "keras_model"


class FaceDetectionResult(BaseModel):
    """
    Resultado completo de detección y reconocimiento facial.

    Attributes:
        box: Coordenadas del bounding box [x1, y1, x2, y2].
        recognized: Indica si el rostro fue reconocido como actor conocido.
        actor_id: ID del actor si fue reconocido.
        nombre: Nombre del actor reconocido.
        personaje: Personaje que interpreta el actor.
        similitud: Porcentaje de similitud con encoding de referencia [0-100].
        emotion: Resultado de análisis emocional del rostro.
    """
    box: List[int] = Field(min_items=4, max_items=4)
    recognized: bool
    actor_id: Optional[int] = None
    nombre: Optional[str] = None
    personaje: Optional[str] = None
    similitud: Optional[float] = Field(None, ge=0.0, le=100.0)
    emotion: Optional[EmotionResult] = None


class ShotAnalysisResult(BaseModel):
    """
    Resultado de clasificación de tipo de plano cinematográfico.

    Attributes:
        shot_type: Tipo de plano (e.g., "Primer Plano", "Plano Medio").
        confidence: Nivel de confianza de la clasificación [0.0, 1.0].
        face_height_ratio: Ratio altura_rostro/altura_frame.
        face_area_ratio: Ratio área_rostro/área_frame.
        people_detected: Número de personas detectadas en el frame.
    """
    shot_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    face_height_ratio: Optional[float] = None
    face_area_ratio: Optional[float] = None
    people_detected: Optional[int] = None


class CompositionResult(BaseModel):
    """
    Resultado de análisis de composición visual.

    Attributes:
        rule_of_thirds: Evaluación de adherencia a regla de tercios.
        symmetry: Métricas de simetría vertical y horizontal.
        lines: Análisis de líneas dominantes detectadas.
        balance: Evaluación de balance visual y distribución de pesos.
    """
    rule_of_thirds: Dict[str, Any]
    symmetry: Dict[str, Any]
    lines: Dict[str, Any]
    balance: Dict[str, Any]


class LightingResult(BaseModel):
    """
    Resultado de análisis de iluminación cinematográfica.

    Attributes:
        lighting_type: Clasificación del tipo de iluminación.
        exposure: Análisis de exposición y distribución tonal.
        contrast: Métricas de contraste por múltiples métodos.
        distribution: Distribución espacial de luz en cuadrantes.
        light_direction: Dirección de luz principal calculada por gradientes.
    """
    lighting_type: str
    exposure: Dict[str, Any]
    contrast: Dict[str, Any]
    distribution: Dict[str, Any]
    light_direction: Dict[str, Any]


class ColorAnalysisResult(BaseModel):
    """
    Resultado de análisis cromático del frame.

    Attributes:
        dominant_colors: Lista de colores dominantes con RGB, hex y porcentajes.
        temperature: Análisis de temperatura de color (cálido/frío).
        color_scheme: Clasificación de esquema cromático (monocromático,
            complementario, etc.).
    """
    dominant_colors: List[Dict[str, Any]]
    temperature: Dict[str, Any]
    color_scheme: Dict[str, Any]


class CameraMovementResult(BaseModel):
    """
    Resultado de análisis de movimiento de cámara.

    Attributes:
        movement_type: Tipo de movimiento detectado (pan, tilt, zoom, etc.).
        confidence: Nivel de confianza de la clasificación [0.0, 1.0].
        stability: Métricas de estabilidad de cámara.
        is_moving: Flag booleano indicando si hay movimiento significativo.
        intensity: Intensidad del movimiento en escala [0.0, 100.0].
    """
    movement_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    stability: Optional[Dict[str, Any]] = None
    is_moving: bool
    intensity: float = Field(ge=0.0, le=100.0)


# ==================== MODELOS DE RESULTADOS ====================

class ActorInfo(BaseModel):
    """
    Información agregada de actor detectado en el video.

    Attributes:
        actor_id: Identificador único del actor.
        nombre: Nombre completo del actor.
        personaje: Personaje que interpreta.
        foto_url: URL de fotografía de referencia.
        detecciones: Número de frames donde fue detectado.
        similitud: Similitud promedio en todas las detecciones [0-100].
        similitud_maxima: Similitud máxima alcanzada [0-100].
    """
    actor_id: int
    nombre: str
    personaje: str
    foto_url: str
    detecciones: int = 0
    similitud: float = Field(ge=0.0, le=100.0)
    similitud_maxima: float = Field(ge=0.0, le=100.0)


class FrameAnalysisResult(BaseModel):
    """
    Resultado completo de análisis de frame individual.

    Contiene todos los análisis cinematográficos aplicados al frame:
    detección facial, clasificación de plano, composición, iluminación,
    colores y movimiento de cámara.

    Attributes:
        frame_number: Índice del frame en la secuencia.
        faces: Lista de rostros detectados con reconocimiento y emociones.
        shot_type: Clasificación del tipo de plano.
        composition: Análisis de composición visual.
        lighting: Análisis de iluminación cinematográfica.
        colors: Análisis cromático y temperatura de color.
        camera_movement: Análisis de movimiento de cámara.
        emotions: Lista de emociones detectadas en el frame.
    """
    frame_number: int
    faces: List[FaceDetectionResult]
    shot_type: Optional[ShotAnalysisResult] = None
    composition: Optional[CompositionResult] = None
    lighting: Optional[LightingResult] = None
    colors: Optional[ColorAnalysisResult] = None
    camera_movement: Optional[CameraMovementResult] = None
    emotions: List[EmotionResult] = []


class VideoProcessingInfo(BaseModel):
    """
    Información inicial enviada al inicio de procesamiento por SSE.

    Attributes:
        type: Tipo de mensaje ("info").
        total_frames: Número total de frames a procesar.
        fps: Frame rate del video.
        duration: Duración total en segundos.
        actors_loaded: Número de actores cargados para reconocimiento.
        models_ready: Indica si todos los modelos están listos.
        poster_url: URL del poster del contenido si está disponible.
        optimizations: Configuración de optimizaciones aplicadas.
    """
    type: str = "info"
    total_frames: int
    fps: float
    duration: float
    actors_loaded: int
    models_ready: bool
    poster_url: Optional[str] = None
    optimizations: Dict[str, Any]


class ProgressUpdate(BaseModel):
    """
    Actualización de progreso durante procesamiento.

    Attributes:
        type: Tipo de mensaje ("progress").
        frame_number: Frame actual siendo procesado.
        total_frames: Total de frames a procesar.
        progress: Porcentaje de completitud [0.0, 100.0].
    """
    type: str = "progress"
    frame_number: int
    total_frames: int
    progress: float = Field(ge=0.0, le=100.0)


class FinalAnalysisResults(BaseModel):
    """
    Resultados finales y resúmenes agregados del análisis completo.

    Contiene estadísticas globales, actores detectados, distribuciones
    de características cinematográficas y datos para visualizaciones.

    Attributes:
        type: Tipo de mensaje ("done").
        total_frames_processed: Número de frames procesados exitosamente.
        message: Mensaje de finalización.
        detected_actors: Lista de actores detectados con estadísticas.
        total_actors_detected: Conteo único de actores diferentes.
        camera_summary: Resumen de tipos de movimiento de cámara.
        shot_types_summary: Distribución de tipos de plano.
        lighting_summary: Resumen de tipos de iluminación.
        emotions_summary: Distribución de emociones detectadas.
        color_analysis_summary: Resumen de análisis cromático.
        composition_summary: Métricas agregadas de composición.
        poster_url: URL del poster del contenido.
        histogram_data: Datos para gráfico de histograma RGB.
        camera_timeline: Timeline de movimientos de cámara.
        composition_data: Datos temporales de métricas compositivas.
    """
    type: str = "done"
    total_frames_processed: int
    message: str
    detected_actors: List[ActorInfo]
    total_actors_detected: int
    camera_summary: Optional[Dict[str, Any]] = None
    shot_types_summary: Optional[Dict[str, Any]] = None
    lighting_summary: Optional[Dict[str, Any]] = None
    emotions_summary: Optional[Dict[str, Any]] = None
    color_analysis_summary: Optional[Dict[str, Any]] = None
    composition_summary: Optional[Dict[str, Any]] = None
    poster_url: Optional[str] = None
    histogram_data: Optional[Dict[str, Any]] = None
    camera_timeline: Optional[List[Dict[str, Any]]] = None
    composition_data: Optional[Dict[str, Any]] = None


# ==================== MODELOS DE MENSAJES ====================

class ErrorResponse(BaseModel):
    """
    Respuesta estandarizada de error.

    Attributes:
        type: Tipo de mensaje ("error").
        message: Mensaje descriptivo del error.
        details: Detalles adicionales del error.
        error_code: Código de error para identificación programática.
    """
    type: str = "error"
    message: str
    details: Optional[str] = None
    error_code: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """
    Respuesta de endpoint de health check.

    Attributes:
        status: Estado general del sistema ("healthy" o "unhealthy").
        models_loaded: Indica si modelos de IA están cargados.
        services: Estado de servicios individuales.
        uptime_seconds: Tiempo de actividad del servidor.
    """
    status: str
    models_loaded: bool
    services: Dict[str, str]
    uptime_seconds: Optional[int] = None


# ==================== MODELOS DE CONFIGURACIÓN ====================

class ProcessingConfigResponse(BaseModel):
    """
    Configuración de procesamiento para el cliente.

    Define parámetros de optimización y muestreo para procesamiento de video.

    Attributes:
        face_detection_skip: Frames a saltar entre detecciones faciales.
        full_analysis_skip: Frames a saltar entre análisis completos.
        max_frame_width: Ancho máximo para redimensionado de frames.
        jpeg_quality: Calidad JPEG para compresión [1-100].
        compression_enabled: Flag de habilitación de compresión.
        priority: Nivel de prioridad del análisis.
        estimated_time: Estimación de tiempo de procesamiento.
    """
    face_detection_skip: int
    full_analysis_skip: int
    max_frame_width: int
    jpeg_quality: int
    compression_enabled: bool
    priority: str
    estimated_time: Optional[Dict[str, float]] = None