"""
config.py

Configuración centralizada de la aplicación CVFlix con sistema de logging
profesional, validación de modelos, ajuste adaptativo de parámetros y
optimizaciones específicas para diferentes entornos de despliegue.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Architecture:
    Sistema de configuración modular organizado en componentes:

    Logging:
        - JSONFormatter: logs estructurados para análisis automatizado
        - ColoredConsoleFormatter: output legible con códigos ANSI
        - RotatingFileHandler: rotación automática de archivos de log

    Validación:
        - ModelValidator: verificación de modelos de IA al inicio
        - Reportes de modelos faltantes con rutas específicas

    Procesamiento:
        - ProcessingConfig: configuración adaptativa por duración de video
        - Perfiles automáticos (quality/balanced/speed)
        - Estimación de tiempos de procesamiento

    Optimización:
        - Detección automática de entorno Render
        - Ajuste de memoria y workers para límites de RAM
        - Configuración específica de TensorFlow

Configuration Profiles:
    Quality (videos <2min):
        - face_detection_skip: 10 frames
        - max_frame_width: 960px
        - jpeg_quality: 60
        - Máximo detalle, velocidad secundaria

    Balanced (videos 2-10min):
        - face_detection_skip: 15 frames
        - max_frame_width: 720px
        - jpeg_quality: 50
        - Balance entre calidad y velocidad

    Speed (videos >10min):
        - face_detection_skip: 20 frames
        - max_frame_width: 640px
        - jpeg_quality: 45
        - Prioridad en velocidad de procesamiento

Environment Detection:
    Render Platform:
        - Detectado via variable RENDER
        - Workers reducidos a 2 (vs 6 default)
        - Frame width máximo 640px (vs 720px)
        - Skips aumentados para reducir carga
        - Logs de TensorFlow minimizados

Notes:
    El sistema de logging configura automáticamente tres handlers:
        1. JSON rotativo para procesamiento automatizado
        2. Texto rotativo para lectura humana
        3. Consola con colores para desarrollo

    Los archivos de log rotan automáticamente al alcanzar 10MB, manteniendo
    5 backups para JSON y 3 para texto plano.

    La configuración se valida al importar el módulo, registrando estado
    de modelos y directorios en logs. Modelos faltantes no impiden inicio
    pero limitan funcionalidades disponibles.

    En entorno Render (512MB RAM), los workers se reducen automáticamente
    de 6 a 2 y los skips aumentan para prevenir out-of-memory errors.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
import json
from datetime import datetime

# ==================== CONFIGURACIÓN PARA SERVIDORES SIN GUI ====================
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidores como Render

# ==================== DIRECTORIOS ====================
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios necesarios
for directory in [VIDEOS_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ==================== LOGGING PROFESIONAL ====================

class JSONFormatter(logging.Formatter):
    """
    Formateador JSON para logs estructurados procesables automáticamente.

    Genera logs en formato JSON con timestamp UTC, nivel, módulo, función
    y contexto adicional. Útil para agregación en sistemas como ELK,
    Splunk o CloudWatch.

    Notes:
        Los timestamps se formatean en ISO 8601 con zona UTC explícita.
        Excepciones se incluyen con traceback completo en campo "exception".
        Atributos extra se pueden añadir vía record.extra_data.
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Añadir contexto extra si existe
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        # Añadir excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Formateador con códigos de color ANSI para output de consola legible.

    Coloriza nivel de log para identificación visual rápida durante desarrollo
    y debugging. Los colores se aplican solo al nivel, no al mensaje completo.

    Notes:
        Los códigos ANSI funcionan en terminales Unix/Linux y Windows 10+.
        En entornos sin soporte ANSI, los códigos se muestran como texto.
    """

    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging():
    """
    Configura sistema de logging tri-modal de la aplicación.

    Crea tres handlers con diferentes propósitos:
        1. JSON rotativo: para procesamiento automatizado y análisis
        2. Texto rotativo: para lectura humana y debugging
        3. Consola coloreada: para desarrollo y monitoreo en tiempo real

    Notes:
        Los handlers JSON y texto rotan al alcanzar 10MB. JSON mantiene
        5 backups, texto mantiene 3. La consola muestra solo INFO+ para
        evitar spam en desarrollo.

        Loggers de librerías ruidosas (urllib3, matplotlib, PIL) se
        silencian a WARNING para reducir ruido.
    """

    # Handler para archivo JSON (rotativo)
    json_handler = RotatingFileHandler(
        LOGS_DIR / 'cvflix.json.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JSONFormatter())

    # Handler para archivo de texto legible
    text_handler = RotatingFileHandler(
        LOGS_DIR / 'cvflix.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    text_handler.setLevel(logging.INFO)
    text_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Handler para consola con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredConsoleFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))

    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(json_handler)
    root_logger.addHandler(text_handler)
    root_logger.addHandler(console_handler)

    # Silenciar loggers ruidosos
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


# Inicializar logging
setup_logging()
logger = logging.getLogger(__name__)

# ==================== MODELOS ====================
FACE_DETECTION_PROTOTXT = MODEL_DIR / "deploy.prototxt"
FACE_DETECTION_WEIGHTS = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
EMOTION_MODEL_PATH = MODEL_DIR / "emotion_model.h5"


class ModelValidator:
    """
    Validador de modelos de IA con verificación de existencia y tamaño.

    Verifica que todos los modelos necesarios estén presentes en disco
    y reporta su tamaño para diagnosticar problemas de descarga incompleta.
    """

    @staticmethod
    def validate_models() -> Dict[str, bool]:
        """
        Valida existencia de todos los modelos necesarios.

        Returns:
            Diccionario mapeando nombre de modelo a flag de existencia.

        Notes:
            Registra en logs el estado de cada modelo con tamaño en MB.
            Útil para diagnosticar problemas de inicialización al revisar logs.
        """
        models = {
            "face_detection_prototxt": FACE_DETECTION_PROTOTXT,
            "face_detection_weights": FACE_DETECTION_WEIGHTS,
            "emotion_model": EMOTION_MODEL_PATH
        }

        status = {}
        for model_name, model_path in models.items():
            exists = model_path.exists()
            status[model_name] = exists

            if exists:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"✅ {model_name}: {size_mb:.2f} MB")
            else:
                logger.warning(f"⚠️ {model_name} no encontrado en {model_path}")

        return status

    @staticmethod
    def get_missing_models() -> list:
        """
        Retorna lista de nombres de modelos faltantes.

        Returns:
            Lista de strings con nombres de modelos no encontrados.

        Example:
            >>> missing = ModelValidator.get_missing_models()
            >>> if missing:
            >>>     print(f"Instalar modelos: {', '.join(missing)}")
        """
        status = ModelValidator.validate_models()
        return [name for name, exists in status.items() if not exists]


# ==================== API KEYS ====================
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    logger.error("❌ TMDB_API_KEY no configurado correctamente")


# ==================== CONFIGURACIÓN DE PROCESAMIENTO ====================
@dataclass
class ProcessingSettings:
    """
    Configuración de procesamiento de video con parámetros optimizados.

    Attributes:
        face_detection_skip: Frames a saltar entre detecciones faciales.
        full_analysis_skip: Frames a saltar entre análisis completos.
        progress_update_skip: Frames entre actualizaciones de progreso SSE.
        max_frame_width: Ancho máximo para redimensionamiento.
        jpeg_quality: Calidad de compresión JPEG [1-100].
        priority: Nivel de prioridad (quality/balanced/speed).
    """
    face_detection_skip: int
    full_analysis_skip: int
    progress_update_skip: int
    max_frame_width: int
    jpeg_quality: int
    priority: str


class ProcessingConfig:
    """
    Configuración centralizada y adaptativa de procesamiento de video.

    Proporciona valores por defecto optimizados y métodos para ajuste
    dinámico basado en características del video (duración, resolución).
    """

    # Configuración base
    FACE_DETECTION_SKIP = 15
    FULL_ANALYSIS_SKIP = 45
    PROGRESS_UPDATE_SKIP = 5

    # Umbrales de confianza
    FACE_CONFIDENCE_THRESHOLD = 0.60
    FACE_RECOGNITION_THRESHOLD = 0.60
    MIN_FACE_SIZE = 50

    # Optimización de imágenes
    MAX_FRAME_WIDTH = 720
    JPEG_QUALITY = 50
    RESIZE_INTERPOLATION = "INTER_AREA"

    # Procesamiento paralelo
    MAX_WORKERS = 6
    USE_GPU = True
    BATCH_SIZE = 4

    # Tracking facial
    USE_FACE_TRACKING = True
    TRACKING_THRESHOLD = 50

    @classmethod
    def get_optimal_settings(cls, video_duration: float) -> ProcessingSettings:
        """
        Calcula configuración óptima según duración del video.

        Ajusta parámetros de muestreo y calidad para balance entre
        velocidad de procesamiento y calidad de análisis.

        Args:
            video_duration: Duración del video en segundos.

        Returns:
            ProcessingSettings con parámetros optimizados.

        Notes:
            Estrategia de perfiles:
                - <2min: quality (análisis denso, máximo detalle)
                - 2-10min: balanced (muestreo moderado)
                - >10min: speed (muestreo agresivo para completar rápido)
        """
        if video_duration < 120:  # Videos cortos (< 2 min)
            return ProcessingSettings(
                face_detection_skip=10,
                full_analysis_skip=30,
                progress_update_skip=3,
                max_frame_width=960,
                jpeg_quality=60,
                priority="quality"
            )
        elif video_duration < 600:  # Videos medianos (2-10 min)
            return ProcessingSettings(
                face_detection_skip=cls.FACE_DETECTION_SKIP,
                full_analysis_skip=cls.FULL_ANALYSIS_SKIP,
                progress_update_skip=cls.PROGRESS_UPDATE_SKIP,
                max_frame_width=cls.MAX_FRAME_WIDTH,
                jpeg_quality=cls.JPEG_QUALITY,
                priority="balanced"
            )
        else:  # Videos largos (> 10 min)
            return ProcessingSettings(
                face_detection_skip=20,
                full_analysis_skip=60,
                progress_update_skip=8,
                max_frame_width=640,
                jpeg_quality=45,
                priority="speed"
            )

    @classmethod
    def estimate_processing_time(cls, total_frames: int, fps: float) -> Dict[str, float]:
        """
        Estima tiempo de procesamiento basado en características del video.

        Args:
            total_frames: Número total de frames en el video.
            fps: Frame rate del video.

        Returns:
            Diccionario con estimaciones en segundos y minutos.

        Notes:
            La estimación asume 0.3s por frame analizado en hardware típico.
            Tiempo real varía según número de rostros, resolución y hardware.
        """
        frames_to_analyze = total_frames // cls.FACE_DETECTION_SKIP
        estimated_seconds = frames_to_analyze * 0.3  # 0.3s por frame analizado

        return {
            "total_frames": total_frames,
            "frames_to_analyze": frames_to_analyze,
            "estimated_seconds": round(estimated_seconds, 1),
            "estimated_minutes": round(estimated_seconds / 60, 2),
            "fps": fps
        }


# ==================== CONFIGURACIÓN DE ANÁLISIS ====================
ANALYSIS_CONFIG = {
    "face_detection": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FACE_DETECTION_SKIP,
        "priority": "high",
        "description": "Detección de rostros con DNN"
    },
    "face_recognition": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FACE_DETECTION_SKIP,
        "priority": "high",
        "description": "Reconocimiento facial de actores"
    },
    "emotion_detection": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FULL_ANALYSIS_SKIP,
        "priority": "medium",
        "description": "Detección de emociones faciales"
    },
    "shot_type": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FULL_ANALYSIS_SKIP,
        "priority": "medium",
        "description": "Análisis de tipo de plano"
    },
    "composition": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FULL_ANALYSIS_SKIP * 2,
        "priority": "low",
        "description": "Análisis de composición visual"
    },
    "lighting": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FULL_ANALYSIS_SKIP,
        "priority": "medium",
        "description": "Análisis de iluminación"
    },
    "colors": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FULL_ANALYSIS_SKIP,
        "priority": "medium",
        "description": "Análisis de paleta de colores"
    },
    "camera_movement": {
        "enabled": True,
        "skip_frames": 10,
        "priority": "high",
        "description": "Detección de movimientos de cámara"
    }
}


# ==================== CONFIGURACIÓN DE EMOCIONES ====================
EMOTION_CONFIG = {
    "model_type": "keras",
    "input_size": (48, 48),
    "labels": ['Enfadado', 'Disgustado', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido'],
    "enabled": EMOTION_MODEL_PATH.exists()
}


# ==================== CORS ====================
CORS_ORIGINS = ["*"]


# ==================== RATE LIMITING ====================
RATE_LIMIT_CONFIG = {
    "upload_video": "5/minute",  # 5 uploads por minuto
    "search_content": "30/minute",  # 30 búsquedas por minuto
}


# ==================== CONSTANTES EXPORTADAS ====================
# Exportar para compatibilidad con código existente
FACE_DETECTION_SKIP = ProcessingConfig.FACE_DETECTION_SKIP
FULL_ANALYSIS_SKIP = ProcessingConfig.FULL_ANALYSIS_SKIP
PROGRESS_UPDATE_SKIP = ProcessingConfig.PROGRESS_UPDATE_SKIP
FACE_CONFIDENCE_THRESHOLD = ProcessingConfig.FACE_CONFIDENCE_THRESHOLD
FACE_RECOGNITION_THRESHOLD = ProcessingConfig.FACE_RECOGNITION_THRESHOLD
MIN_FACE_SIZE = ProcessingConfig.MIN_FACE_SIZE
MAX_FRAME_WIDTH = ProcessingConfig.MAX_FRAME_WIDTH
JPEG_QUALITY = ProcessingConfig.JPEG_QUALITY
MAX_WORKERS = ProcessingConfig.MAX_WORKERS
USE_FACE_TRACKING = ProcessingConfig.USE_FACE_TRACKING
TRACKING_THRESHOLD = ProcessingConfig.TRACKING_THRESHOLD


# ==================== VALIDACIÓN AL IMPORTAR ====================
logger.info("=" * 70)
logger.info("🔧 CVFlix Backend v4.0.0 - Validando configuración...")
logger.info("=" * 70)

models_status = ModelValidator.validate_models()
missing = ModelValidator.get_missing_models()

if missing:
    logger.warning(f"⚠️ Modelos faltantes: {', '.join(missing)}")
    logger.warning("   Algunas funcionalidades estarán limitadas")
else:
    logger.info("✅ Todos los modelos disponibles")

logger.info(f"📁 BASE_DIR: {BASE_DIR}")
logger.info(f"📹 VIDEOS_DIR: {VIDEOS_DIR}")
logger.info(f"🤖 MODEL_DIR: {MODEL_DIR}")
logger.info(f"📊 LOGS_DIR: {LOGS_DIR}")
logger.info("=" * 70)

# ==================== CONFIGURACIÓN ESPECÍFICA PARA RENDER ====================
IS_RENDER = os.getenv('RENDER') is not None

if IS_RENDER:
    logger.info("🚀 Modo RENDER detectado - Aplicando optimizaciones...")

    # Reducir uso de memoria para Render (512MB RAM)
    ProcessingConfig.MAX_WORKERS = 2  # Reducido de 6 a 2
    ProcessingConfig.MAX_FRAME_WIDTH = 640  # Reducido de 720 a 640
    ProcessingConfig.FACE_DETECTION_SKIP = 20  # Analizar menos frames
    ProcessingConfig.FULL_ANALYSIS_SKIP = 60  # Analizar menos frames
    ProcessingConfig.PROGRESS_UPDATE_SKIP = 10  # Actualizar menos frecuentemente

    # Configuración de TensorFlow para Render
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TF
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    logger.info("✅ Optimizaciones de Render aplicadas")
    logger.info(f"   - Workers: {ProcessingConfig.MAX_WORKERS}")
    logger.info(f"   - Max frame width: {ProcessingConfig.MAX_FRAME_WIDTH}")
    logger.info(f"   - Face detection skip: {ProcessingConfig.FACE_DETECTION_SKIP}")