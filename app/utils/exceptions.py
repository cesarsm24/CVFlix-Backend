"""
exceptions.py

Sistema completo de excepciones personalizadas para CVFlix con jerarquía
estructurada, códigos de error únicos, logging automático y serialización
para respuestas HTTP/SSE.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Architecture:
    Jerarquía de excepciones organizada por dominio funcional:
        - CVFlixException (base)
            - ModelNotLoadedException, ModelLoadException
            - VideoException (base vídeos)
                - VideoNotFoundException
                - InvalidVideoException
                - VideoProcessingException
                - VideoCodecException
                - VideoTooLargeException
            - FaceDetectionException (base detección facial)
                - NoFacesDetectedException
                - FaceRecognitionException
                - ActorEncodingException
            - ExternalServiceException (base servicios externos)
                - TMDBException
                    - TMDBContentNotFoundException
                    - TMDBAPIKeyException
            - SSEException (base Server-Sent Events)
                - SSEConnectionException
                - SSEStreamException
            - ConfigurationException
            - MissingDependencyException
            - AnalysisException (base análisis)
                - EmotionDetectionException
                - CompositionAnalysisException
                - LightingAnalysisException
            - CacheException (base caché)
                - CacheReadException
                - CacheWriteException

Usage:
    from app.utils.exceptions import VideoNotFoundException

    if not video_file.exists():
        raise VideoNotFoundException(filename="video.mp4")

    try:
        process_video()
    except CVFlixException as e:
        return JSONResponse(
            status_code=e.status_code,
            content=e.to_dict()
        )
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class CVFlixException(Exception):
    """
    Excepción base del sistema CVFlix con estructura estandarizada.

    Proporciona infraestructura común para todas las excepciones del sistema
    incluyendo logging automático, códigos de error únicos, sugerencias de
    código HTTP y serialización a JSON para respuestas API.

    Attributes:
        message: Mensaje descriptivo del error para el usuario
        details: Detalles técnicos adicionales del error
        error_code: Código único de error para identificación programática
        status_code: Código HTTP sugerido para respuesta (por defecto 500)
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        error_code: Optional[str] = None,
        status_code: int = 500
    ):
        """
        Inicializa excepción con logging automático.

        Args:
            message: Mensaje principal del error orientado al usuario
            details: Información técnica adicional para debugging
            error_code: Código único de error. Por defecto "CVFLIX_ERROR"
            status_code: Código HTTP sugerido para la respuesta

        Notes:
            El error se registra automáticamente en logs al construir la excepción,
            facilitando debugging sin necesidad de logging manual en cada punto
            de lanzamiento.
        """
        self.message = message
        self.details = details
        self.error_code = error_code or "CVFLIX_ERROR"
        self.status_code = status_code

        logger.error(f"[{self.error_code}] {self.message}")
        if self.details:
            logger.error(f"  Detalles: {self.details}")

        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa excepción a diccionario para respuesta JSON.

        Returns:
            Diccionario con estructura estandarizada de error compatible
            con respuestas API y eventos SSE

        Notes:
            El formato retornado es consistente con ErrorResponse schema
            definido en models/schemas.py para validación Pydantic.
        """
        return {
            "type": "error",
            "message": self.message,
            "details": self.details,
            "error_code": self.error_code
        }

    def __str__(self) -> str:
        """Representación string con código de error prefijado."""
        return f"[{self.error_code}] {self.message}"


class ModelNotLoadedException(CVFlixException):
    """
    Se lanza cuando se intenta usar modelo de IA que no está cargado.

    Típicamente ocurre durante inicialización del servidor o tras error
    de carga. El código HTTP 503 indica servicio temporalmente no disponible.
    """

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Modelo '{model_name}' no está cargado",
            details="Los modelos están inicializándose. Intenta de nuevo en unos segundos.",
            error_code="MODEL_NOT_LOADED",
            status_code=503
        )


class ModelLoadException(CVFlixException):
    """
    Se lanza cuando falla carga de modelo de IA desde disco.

    Puede indicar archivo corrupto, falta de memoria o incompatibilidad
    de versiones de librerías (TensorFlow, Keras, PyTorch).
    """

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Error cargando modelo '{model_name}'",
            details=reason,
            error_code="MODEL_LOAD_ERROR",
            status_code=500
        )


class VideoException(CVFlixException):
    """Excepción base para errores relacionados con procesamiento de vídeo."""
    pass


class VideoNotFoundException(VideoException):
    """
    Se lanza cuando no se encuentra archivo de vídeo especificado.

    Código HTTP 404 indica recurso no encontrado. Puede ocurrir por eliminación
    del archivo, ruta incorrecta o permisos insuficientes.
    """

    def __init__(self, filename: str):
        super().__init__(
            message=f"Vídeo '{filename}' no encontrado",
            details="El archivo puede haber sido eliminado o la ruta es incorrecta",
            error_code="VIDEO_NOT_FOUND",
            status_code=404
        )


class InvalidVideoException(VideoException):
    """
    Se lanza cuando vídeo no es válido o está corrupto.

    Código HTTP 400 indica error del cliente. Causas comunes: archivo corrupto,
    formato no soportado, headers inválidos o codificación incorrecta.
    """

    def __init__(self, filename: str, reason: Optional[str] = None):
        super().__init__(
            message=f"El vídeo '{filename}' no es válido",
            details=reason or "El archivo puede estar corrupto o no ser un formato soportado",
            error_code="INVALID_VIDEO",
            status_code=400
        )


class VideoProcessingException(VideoException):
    """
    Se lanza cuando ocurre error durante procesamiento de vídeo.

    Excepción genérica para errores en pipeline de análisis que no tienen
    clasificación más específica. Útil para errores inesperados en análisis.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="VIDEO_PROCESSING_ERROR",
            status_code=500
        )


class VideoCodecException(VideoException):
    """
    Se lanza cuando codec de vídeo no está soportado.

    OpenCV soporta codecs limitados según compilación. Formatos recomendados:
    MP4 (H.264), AVI, MOV, MKV. Codecs problemáticos: VP9, AV1, HEVC sin libs.
    """

    def __init__(self, codec: str):
        super().__init__(
            message="Codec de vídeo no soportado",
            details=f"El codec '{codec}' no está soportado. Usa MP4, AVI, MOV o MKV",
            error_code="UNSUPPORTED_CODEC",
            status_code=400
        )


class VideoTooLargeException(VideoException):
    """
    Se lanza cuando vídeo excede tamaño máximo permitido.

    Código HTTP 413 indica payload demasiado grande. El límite previene
    agotamiento de memoria y tiempos de procesamiento excesivos.
    """

    def __init__(self, size_mb: float, max_size_mb: float):
        super().__init__(
            message="El vídeo es demasiado grande",
            details=f"Tamaño: {size_mb:.1f} MB, Máximo permitido: {max_size_mb:.1f} MB",
            error_code="VIDEO_TOO_LARGE",
            status_code=413
        )


class FaceDetectionException(CVFlixException):
    """Excepción base para errores de detección y reconocimiento facial."""
    pass


class NoFacesDetectedException(FaceDetectionException):
    """
    Se lanza cuando no se detectan rostros en el vídeo.

    El sistema requiere rostros visibles para análisis facial, reconocimiento
    de actores y detección emocional. Vídeos sin personas no son procesables.
    """

    def __init__(self):
        super().__init__(
            message="No se detectaron rostros en el vídeo",
            details="El vídeo debe contener rostros visibles para el análisis",
            error_code="NO_FACES_DETECTED",
            status_code=400
        )


class FaceRecognitionException(FaceDetectionException):
    """
    Se lanza cuando falla reconocimiento facial de actores.

    Puede indicar error en cálculo de encodings, comparación de similitudes
    o acceso a base de datos de actores conocidos.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="FACE_RECOGNITION_ERROR",
            status_code=500
        )


class ActorEncodingException(FaceDetectionException):
    """
    Se lanza cuando falla procesamiento de encodings de actores.

    Puede ocurrir por foto de referencia inválida, rostro no detectable
    en imagen de actor o error en cálculo de embedding facial.
    """

    def __init__(self, actor_name: str, reason: str):
        super().__init__(
            message=f"Error procesando encoding de '{actor_name}'",
            details=reason,
            error_code="ACTOR_ENCODING_ERROR",
            status_code=500
        )


class ExternalServiceException(CVFlixException):
    """Excepción base para errores de servicios externos (TMDB, APIs)."""
    pass


class TMDBException(ExternalServiceException):
    """
    Se lanza cuando hay errores con API de The Movie Database.

    Código HTTP 502 indica Bad Gateway (error del servicio externo).
    Causas: timeout, API caída, límite de rate exceeded.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="TMDB_ERROR",
            status_code=502
        )


class TMDBContentNotFoundException(TMDBException):
    """
    Se lanza cuando no se encuentra contenido en base de datos TMDB.

    Ocurre con búsquedas de películas/series que no existen en TMDB o
    cuando el término de búsqueda es demasiado vago o incorrecto.
    """

    def __init__(self, query: str):
        super().__init__(
            message=f"No se encontró contenido para '{query}'",
            details="Intenta con otro nombre o título más específico",
            error_code="TMDB_CONTENT_NOT_FOUND",
            status_code=404
        )


class TMDBAPIKeyException(TMDBException):
    """
    Se lanza cuando hay problemas con API key de TMDB.

    Código HTTP 401 indica falta de autenticación. Causas: API key inválida,
    expirada, sin permisos suficientes o no configurada en variables de entorno.
    """

    def __init__(self):
        super().__init__(
            message="Error de autenticación con TMDB",
            details="La API key no es válida o ha expirado",
            error_code="TMDB_AUTH_ERROR",
            status_code=401
        )


class SSEException(CVFlixException):
    """Excepción base para errores de comunicación Server-Sent Events."""
    pass


class SSEConnectionException(SSEException):
    """
    Se lanza cuando falla establecimiento o mantenimiento de conexión SSE.

    Causas comunes: timeout de red, cierre abrupto de cliente, error en
    headers HTTP (Accept: text/event-stream) o rechazo del servidor.
    """

    def __init__(self, reason: str):
        super().__init__(
            message="Error en conexión SSE",
            details=reason,
            error_code="SSE_CONNECTION_ERROR",
            status_code=500
        )


class SSEStreamException(SSEException):
    """
    Se lanza cuando hay error en envío de eventos SSE.

    Puede indicar evento malformado, error de serialización JSON,
    buffer lleno o desconexión durante transmisión de eventos.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="SSE_STREAM_ERROR",
            status_code=500
        )


class ConfigurationException(CVFlixException):
    """
    Se lanza cuando hay errores en configuración del sistema.

    Incluye parámetros inválidos, archivos de configuración corruptos,
    variables de entorno faltantes o valores fuera de rango permitido.
    """

    def __init__(self, config_name: str, reason: str):
        super().__init__(
            message=f"Error en configuración '{config_name}'",
            details=reason,
            error_code="CONFIGURATION_ERROR",
            status_code=500
        )


class MissingDependencyException(CVFlixException):
    """
    Se lanza cuando falta dependencia Python requerida.

    Útil para importaciones opcionales que fallan. Proporciona instrucción
    de instalación automáticamente en detalles del error.
    """

    def __init__(self, dependency: str):
        super().__init__(
            message=f"Dependencia faltante: {dependency}",
            details=f"Instala con: pip install {dependency}",
            error_code="MISSING_DEPENDENCY",
            status_code=500
        )


class AnalysisException(CVFlixException):
    """Excepción base para errores en analizadores cinematográficos."""
    pass


class EmotionDetectionException(AnalysisException):
    """
    Se lanza cuando falla detección de emociones faciales.

    Puede indicar modelo no cargado, región facial inválida para análisis
    o error en preprocesamiento de imagen para red neuronal.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="EMOTION_DETECTION_ERROR",
            status_code=500
        )


class CompositionAnalysisException(AnalysisException):
    """
    Se lanza cuando falla análisis de composición visual.

    Errores típicos: frame inválido, falla en detección de líneas,
    error en cálculo de simetría o problemas con análisis de balance.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="COMPOSITION_ANALYSIS_ERROR",
            status_code=500
        )


class LightingAnalysisException(AnalysisException):
    """
    Se lanza cuando falla análisis de iluminación cinematográfica.

    Puede ocurrir por frame completamente negro/blanco, error en cálculo
    de histogramas o falla en detección de dirección de luz.
    """

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message,
            details=details,
            error_code="LIGHTING_ANALYSIS_ERROR",
            status_code=500
        )


class CacheException(CVFlixException):
    """Excepción base para errores del sistema de caché."""
    pass


class CacheReadException(CacheException):
    """
    Se lanza cuando falla lectura de entrada de caché.

    Causas: archivo corrupto, error de deserialización pickle,
    permisos insuficientes o entrada expirada y malformada.
    """

    def __init__(self, key: str, reason: str):
        super().__init__(
            message=f"Error leyendo caché '{key}'",
            details=reason,
            error_code="CACHE_READ_ERROR",
            status_code=500
        )


class CacheWriteException(CacheException):
    """
    Se lanza cuando falla escritura a caché.

    Causas: disco lleno, permisos insuficientes, error de serialización
    pickle o límite de tamaño de caché excedido.
    """

    def __init__(self, key: str, reason: str):
        super().__init__(
            message=f"Error escribiendo caché '{key}'",
            details=reason,
            error_code="CACHE_WRITE_ERROR",
            status_code=500
        )


def handle_exception(exc: Exception) -> Dict[str, Any]:
    """
    Maneja cualquier excepción convirtiéndola a formato estándar.

    Proporciona conversión uniforme de excepciones tanto CVFlix como
    excepciones nativas Python a diccionarios para respuestas JSON.

    Args:
        exc: Excepción a convertir

    Returns:
        Diccionario con estructura estandarizada de error compatible
        con ErrorResponse schema

    Notes:
        Para excepciones CVFlix utiliza to_dict(). Para excepciones
        nativas crea estructura genérica con detalles solo en modo debug
        para evitar exposición de información sensible en producción.
    """
    if isinstance(exc, CVFlixException):
        return exc.to_dict()

    logger.error(f"Excepción no controlada: {type(exc).__name__}: {str(exc)}")

    return {
        "type": "error",
        "message": "Ha ocurrido un error inesperado",
        "details": str(exc) if logger.level == logging.DEBUG else None,
        "error_code": "UNEXPECTED_ERROR"
    }


def raise_for_status(condition: bool, exception: CVFlixException):
    """
    Lanza excepción condicionalmente si se cumple condición.

    Utilidad para validaciones que reduce verbosidad de chequeos con if/raise.
    Similar a assert pero con excepciones tipadas del dominio.

    Args:
        condition: Condición a evaluar. Si True, lanza excepción
        exception: Excepción pre-construida a lanzar

    Notes:
        La excepción debe estar completamente construida antes de la llamada.
        Esto permite mensajes de error detallados con contexto específico.
    """
    if condition:
        raise exception