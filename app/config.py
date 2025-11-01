"""
config.py

Configuración centralizada de la aplicación CVFlix con sistema de logging
profesional, validación de modelos, ajuste adaptativo de parámetros y
optimizaciones específicas para diferentes entornos de despliegue.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - opencv-python: Procesamiento de vídeo
    - matplotlib: Generación de gráficos
    - logging: Sistema de registro de eventos

Usage:
    from app.config import (
        MODEL_DIR,
        ProcessingConfig,
        ANALYSIS_CONFIG,
        FACE_DETECTION_METHOD
    )
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
import cv2

import matplotlib
matplotlib.use('Agg')

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for directory in [VIDEOS_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """
    Formateador JSON para logs estructurados procesables automáticamente.

    Genera logs en formato JSON con timestamp UTC, nivel, módulo, función
    y contexto adicional. Útil para agregación en sistemas de análisis de logs.

    Notes:
        Los timestamps se formatean en ISO 8601 con zona UTC explícita.
        Excepciones se incluyen con traceback completo en campo "exception".
        Atributos extra se pueden añadir mediante record.extra_data.
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

        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

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

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
        'RESET': '\033[0m'
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

    json_handler = RotatingFileHandler(
        LOGS_DIR / 'cvflix.json.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    json_handler.setLevel(logging.INFO)
    json_handler.setFormatter(JSONFormatter())

    text_handler = RotatingFileHandler(
        LOGS_DIR / 'cvflix.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    text_handler.setLevel(logging.INFO)
    text_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredConsoleFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(json_handler)
    root_logger.addHandler(text_handler)
    root_logger.addHandler(console_handler)

    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)

FACE_DETECTION_PROTOTXT = MODEL_DIR / "deploy.prototxt"
FACE_DETECTION_WEIGHTS = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
EMOTION_MODEL_PATH = MODEL_DIR / "emotion_model.h5"

HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
HAAR_CASCADE_PROFILE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

FACE_DETECTION_METHOD = "dnn"

VIOLA_JONES_SCALE_FACTOR = 1.1
VIOLA_JONES_MIN_NEIGHBORS = 5


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
            Diccionario mapeando nombre de modelo a flag de existencia

        Notes:
            Registra en logs el estado de cada modelo con tamaño en MB.
            Útil para diagnosticar problemas de inicialización al revisar logs.
        """
        models = {
            "emotion_model": EMOTION_MODEL_PATH
        }

        if FACE_DETECTION_METHOD.lower() == "dnn":
            models["face_detection_prototxt"] = FACE_DETECTION_PROTOTXT
            models["face_detection_weights"] = FACE_DETECTION_WEIGHTS
        elif FACE_DETECTION_METHOD.lower() == "viola-jones":
            models["haar_cascade"] = Path(HAAR_CASCADE_PATH)

        status = {}

        logger.info(f"Método de detección facial: {FACE_DETECTION_METHOD.upper()}")

        for model_name, model_path in models.items():
            exists = model_path.exists()
            status[model_name] = exists

            if exists:
                size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"{model_name}: {size_mb:.2f} MB")
            else:
                logger.warning(f"{model_name} no encontrado en {model_path}")

        return status

    @staticmethod
    def get_missing_models() -> list:
        """
        Retorna lista de nombres de modelos faltantes.

        Returns:
            Lista de strings con nombres de modelos no encontrados
        """
        status = ModelValidator.validate_models()
        return [name for name, exists in status.items() if not exists]


TMDB_API_KEY = "2d51820b0b76e3ea8a7d2862af21839a"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    logger.error("TMDB_API_KEY no configurado correctamente")


@dataclass
class ProcessingSettings:
    """
    Configuración de procesamiento de vídeo con parámetros optimizados.

    Attributes:
        face_detection_skip: Frames a saltar entre detecciones faciales
        full_analysis_skip: Frames a saltar entre análisis completos
        progress_update_skip: Frames entre actualizaciones de progreso SSE
        max_frame_width: Ancho máximo para redimensionamiento
        jpeg_quality: Calidad de compresión JPEG [1-100]
        priority: Nivel de prioridad (quality/balanced/speed)
    """
    face_detection_skip: int
    full_analysis_skip: int
    progress_update_skip: int
    max_frame_width: int
    jpeg_quality: int
    priority: str


class ProcessingConfig:
    """
    Configuración centralizada y adaptativa de procesamiento de vídeo.

    Proporciona valores por defecto optimizados y métodos para ajuste
    dinámico basado en características del vídeo (duración, resolución).
    """

    FACE_DETECTION_SKIP = 15
    FULL_ANALYSIS_SKIP = 45
    PROGRESS_UPDATE_SKIP = 5

    FACE_CONFIDENCE_THRESHOLD = 0.60
    FACE_RECOGNITION_THRESHOLD = 0.58
    MIN_FACE_SIZE = 60

    MAX_FRAME_WIDTH = 1080
    JPEG_QUALITY = 50
    RESIZE_INTERPOLATION = "INTER_AREA"

    MAX_WORKERS = 6
    USE_GPU = True
    BATCH_SIZE = 4

    USE_FACE_TRACKING = True
    TRACKING_THRESHOLD = 50

    @classmethod
    def get_optimal_settings(cls, video_duration: float) -> ProcessingSettings:
        """
        Calcula configuración óptima según duración del vídeo.

        Ajusta parámetros de muestreo y calidad para balance entre
        velocidad de procesamiento y calidad de análisis.

        Args:
            video_duration: Duración del vídeo en segundos

        Returns:
            ProcessingSettings con parámetros optimizados

        Notes:
            Estrategia de perfiles:
                - <2min: quality (análisis denso, máximo detalle)
                - 2-10min: balanced (muestreo moderado)
                - >10min: speed (muestreo agresivo para completar rápido)
        """
        if video_duration < 120:
            return ProcessingSettings(
                face_detection_skip=10,
                full_analysis_skip=30,
                progress_update_skip=3,
                max_frame_width=960,
                jpeg_quality=60,
                priority="quality"
            )
        elif video_duration < 600:
            return ProcessingSettings(
                face_detection_skip=cls.FACE_DETECTION_SKIP,
                full_analysis_skip=cls.FULL_ANALYSIS_SKIP,
                progress_update_skip=cls.PROGRESS_UPDATE_SKIP,
                max_frame_width=cls.MAX_FRAME_WIDTH,
                jpeg_quality=cls.JPEG_QUALITY,
                priority="balanced"
            )
        else:
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
        Estima tiempo de procesamiento basado en características del vídeo.

        Args:
            total_frames: Número total de frames en el vídeo
            fps: Frame rate del vídeo

        Returns:
            Diccionario con estimaciones en segundos y minutos

        Notes:
            La estimación asume 0.3s por frame analizado en hardware típico.
            Tiempo real varía según número de rostros, resolución y hardware.
        """
        frames_to_analyze = total_frames // cls.FACE_DETECTION_SKIP
        estimated_seconds = frames_to_analyze * 0.3

        return {
            "total_frames": total_frames,
            "frames_to_analyze": frames_to_analyze,
            "estimated_seconds": round(estimated_seconds, 1),
            "estimated_minutes": round(estimated_seconds / 60, 2),
            "fps": fps
        }


COMPOSITION_CONFIG = {
    "rule_of_thirds": {
        "tolerance": 0.1,
        "grid_color": (255, 255, 0),
        "line_thickness": 2
    },
    "symmetry": {
        "threshold": 0.85
    },
    "balance": {
        "weight_threshold": 0.3
    }
}

LIGHTING_CONFIG = {
    "high_key": {
        "brightness_threshold": 160,
        "contrast_max": 50
    },
    "low_key": {
        "brightness_threshold": 80,
        "contrast_min": 60
    },
    "exposure": {
        "underexposed": 70,
        "overexposed": 180
    },
    "light_direction": {
        "threshold": 5.0,
        "enabled": True
    }
}

CAMERA_MOVEMENT_CONFIG = {
    "motion_threshold": 2.0,
    "pan_threshold": 3.0,
    "tilt_threshold": 3.0,
    "zoom_threshold": 0.05,
    "min_motion_frames": 3
}

EMOTIONS = {
    "angry": "Enfadado/a",
    "disgust": "Disgustado/a",
    "fear": "Miedo",
    "happy": "Feliz",
    "sad": "Triste",
    "surprise": "Sorprendido/a",
    "neutral": "Neutral"
}

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 165, 255),
    "neutral": (128, 128, 128)
}

SHOT_TYPES = {
    "ECU": "Extreme Close-Up",
    "CU": "Close-Up",
    "MCU": "Medium Close-Up",
    "MS": "Medium Shot",
    "MWS": "Medium Wide Shot",
    "WS": "Wide Shot",
    "EWS": "Extreme Wide Shot"
}

SHOT_TYPE_THRESHOLDS = {
    "ECU": 0.7,
    "CU": 0.5,
    "MCU": 0.35,
    "MS": 0.25,
    "MWS": 0.15,
    "WS": 0.1,
    "EWS": 0.0
}

ANALYSIS_CONFIG = {
    "face_detection": {
        "enabled": True,
        "skip_frames": ProcessingConfig.FACE_DETECTION_SKIP,
        "priority": "high",
        "description": f"Detección de rostros con {FACE_DETECTION_METHOD.upper()}"
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

EMOTION_CONFIG = {
    "model_type": "keras",
    "input_size": (48, 48),
    "labels": ['Enfadado', 'Disgustado', 'Miedo', 'Feliz', 'Neutral', 'Triste', 'Sorprendido'],
    "enabled": EMOTION_MODEL_PATH.exists()
}

CORS_ORIGINS = ["*"]

RATE_LIMIT_CONFIG = {
    "upload_video": "5/minute",
    "search_content": "30/minute",
}

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

logger.info("=" * 70)
logger.info("CVFlix Backend v4.0.0 - Validando configuración...")
logger.info("=" * 70)

models_status = ModelValidator.validate_models()
missing = ModelValidator.get_missing_models()

if missing:
    logger.warning(f"Modelos faltantes: {', '.join(missing)}")
    logger.warning("Algunas funcionalidades estarán limitadas")
else:
    logger.info("Todos los modelos disponibles")

logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"VIDEOS_DIR: {VIDEOS_DIR}")
logger.info(f"MODEL_DIR: {MODEL_DIR}")
logger.info(f"LOGS_DIR: {LOGS_DIR}")
logger.info("=" * 70)