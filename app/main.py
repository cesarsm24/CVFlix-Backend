"""
CVFlix API - Backend

API de análisis cinematográfico con IA y OpenCV para procesamiento automático
de vídeo mediante detección facial, reconocimiento de actores, análisis de
emociones, clasificación de planos y evaluación de composición visual.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Core Features:
    - Reconocimiento facial con TMDB
    - Análisis de composición, iluminación y colores
    - Detección de emociones
    - Movimientos de cámara
    - Sistema de caché inteligente
    - Monitoreo de rendimiento
    - Manejo robusto de errores
    - Logging profesional
    - SSE streaming

Dependencies:
    - FastAPI: Framework web asíncrono
    - OpenCV: Procesamiento de vídeo
    - face_recognition: Reconocimiento facial
    - TensorFlow/Keras: Detección de emociones
    - sse_starlette: Server-Sent Events

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import cv2
import os
import numpy as np
import json
import logging
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
import httpx
from pathlib import Path
import uuid
import base64

from sse_starlette.sse import EventSourceResponse

from app.config import (
    VIDEOS_DIR,
    CORS_ORIGINS,
    MAX_WORKERS,
    FACE_DETECTION_SKIP,
    FULL_ANALYSIS_SKIP,
    PROGRESS_UPDATE_SKIP,
    MAX_FRAME_WIDTH,
    JPEG_QUALITY,
    ANALYSIS_CONFIG,
    TMDB_API_KEY,
    TMDB_IMAGE_BASE,
    logger as config_logger
)
from app.services.tmdb_service import TMDBService
from app.services.video_processor import VideoProcessor
from app.analysis.face_detection import FaceRecognizer
from app.utils.image_utils import frame_to_base64
from app.utils.video_utils import get_video_info
from app.utils.exceptions import (
    CVFlixException,
    ModelNotLoadedException,
    VideoProcessingException,
    VideoNotFoundException
)

try:
    from app.core.performance import performance_monitor, FrameTimer
    PERFORMANCE_MONITORING = True
except ImportError:
    config_logger.warning("Monitoreo de rendimiento no disponible")
    PERFORMANCE_MONITORING = False

try:
    from app.core.cache import CacheManager
    cache_manager = CacheManager(cache_dir=Path("cache"), max_size_mb=500)
    CACHE_ENABLED = True
except ImportError:
    config_logger.warning("Sistema de caché no disponible")
    CACHE_ENABLED = False

try:
    from app.middleware.error_handler import (
        ErrorHandlerMiddleware,
        validation_exception_handler,
        http_exception_handler,
        general_exception_handler
    )
    ERROR_HANDLERS = True
except ImportError:
    config_logger.warning("Manejadores de error personalizados no disponibles")
    ERROR_HANDLERS = False


app = FastAPI(
    title="CVFlix API",
    description="API Profesional de Análisis Cinematográfico con IA",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "CVFlix Team",
        "url": "https://github.com/cvflix",
    },
    license_info={
        "name": "MIT License",
    }
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if ERROR_HANDLERS:
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

logger = logging.getLogger(__name__)

tmdb_service = TMDBService()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
models_loaded = False
startup_time = 0
app_start_time = time.time()

global_video_processor: VideoProcessor = None

os.makedirs(VIDEOS_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Encoder JSON personalizado para manejar tipos NumPy."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bytes):
            return obj.decode('utf-8')
        return super().default(obj)


def convert_to_serializable(obj: Any) -> Any:
    """
    Convierte recursivamente objetos NumPy a tipos Python nativos.

    Args:
        obj: Objeto a convertir (puede ser dict, list, tuple o tipo NumPy)

    Returns:
        Objeto con tipos Python nativos serializables a JSON
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8')
    return obj


def should_run_analysis(frame_number: int, analysis_type: str) -> bool:
    """
    Determina si debe ejecutar un análisis específico en el frame actual.

    Args:
        frame_number: Número del frame actual
        analysis_type: Tipo de análisis a verificar

    Returns:
        True si debe ejecutar el análisis, False en caso contrario
    """
    config = ANALYSIS_CONFIG.get(analysis_type, {})
    if not config.get("enabled", True):
        return False
    skip = config.get("skip_frames", FULL_ANALYSIS_SKIP)
    return frame_number % skip == 0


def load_models():
    """
    Precarga modelos de IA al iniciar la aplicación.

    Raises:
        Exception: Si hay error crítico cargando los modelos
    """
    global models_loaded, startup_time, global_video_processor

    logger.info("=" * 70)
    logger.info("Precargando modelos de IA...")
    logger.info("=" * 70)

    start = time.time()

    try:
        logger.info("Cargando VideoProcessor global...")
        global_video_processor = VideoProcessor()
        logger.info("VideoProcessor global cargado")

        logger.info("Validando FaceRecognizer...")
        test_encoding = FaceRecognizer.extract_encoding_from_image(
            np.zeros((100, 100, 3), dtype=np.uint8)
        )
        logger.info("FaceRecognizer validado")

        elapsed = time.time() - start
        startup_time = int(elapsed)
        models_loaded = True

        logger.info(f"Todos los modelos cargados en {elapsed:.2f}s")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def get_processor_instance() -> VideoProcessor:
    """
    Retorna una nueva instancia del procesador para cada análisis.

    Crea una instancia fresca del VideoProcessor para garantizar que los
    encodings de rostros no se comparten entre diferentes contenidos,
    evitando falsos positivos de reconocimiento facial.

    Returns:
        Nueva instancia limpia del VideoProcessor

    Raises:
        ModelNotLoadedException: Si el VideoProcessor no se puede instanciar
    """
    global global_video_processor

    try:
        logger.info("Creando nueva instancia de VideoProcessor")

        global_video_processor = VideoProcessor()

        logger.info("Nueva instancia de VideoProcessor creada exitosamente")
        logger.info("Encodings de rostros: vacíos")
        logger.info("Estado del procesador: limpio")

        return global_video_processor

    except Exception as e:
        logger.error(f"Error creando nueva instancia de VideoProcessor: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise ModelNotLoadedException("VideoProcessor")


@app.get("/")
async def root():
    """
    Endpoint raíz con información de la API.

    Returns:
        Diccionario con información del estado y características de la API
    """
    uptime = int(time.time() - app_start_time)
    return {
        "name": "CVFlix API",
        "version": "4.0.0",
        "status": "online",
        "models_loaded": models_loaded,
        "startup_time_seconds": startup_time,
        "uptime_seconds": uptime,
        "features": {
            "face_recognition": True,
            "emotion_detection": True,
            "shot_analysis": True,
            "lighting_analysis": True,
            "color_analysis": True,
            "camera_movement": True,
            "composition_analysis": True,
            "streaming_sse": True,
            "performance_monitoring": PERFORMANCE_MONITORING,
            "caching": CACHE_ENABLED
        },
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs",
            "process_video": "/api/process-video-sse",
            "search_content": "/search-content",
            "upload_video": "/upload-video",
            "image_proxy": "/image-proxy"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint para verificar estado del servicio.

    Returns:
        Diccionario con estado de salud del servicio
    """
    health_data = {
        "status": "healthy" if models_loaded else "degraded",
        "version": "4.0.0",
        "models_loaded": models_loaded,
        "uptime_seconds": int(time.time() - app_start_time)
    }

    if CACHE_ENABLED:
        health_data["cache"] = {
            "enabled": True,
            "size_mb": round(cache_manager.get_total_size_mb(), 2)
        }

    if PERFORMANCE_MONITORING:
        health_data["performance"] = performance_monitor.get_current_stats()

    return health_data


@app.get("/stats")
async def get_stats():
    """
    Endpoint de estadísticas detalladas del sistema.

    Returns:
        Diccionario con estadísticas de configuración, caché y rendimiento
    """
    stats = {
        "uptime_seconds": int(time.time() - app_start_time),
        "models_loaded": models_loaded,
        "config": {
            "face_detection_skip": FACE_DETECTION_SKIP,
            "full_analysis_skip": FULL_ANALYSIS_SKIP,
            "max_workers": MAX_WORKERS,
            "max_frame_width": MAX_FRAME_WIDTH
        }
    }

    if CACHE_ENABLED:
        stats["cache"] = cache_manager.get_stats()

    if PERFORMANCE_MONITORING:
        stats["performance"] = performance_monitor.get_stats()

    return stats


@app.get("/search-content")
async def search_content(query: str, content_type: str = "movie"):
    """
    Busca contenido en TMDB y retorna información completa.

    Args:
        query: Título a buscar (película o serie)
        content_type: Tipo de contenido ("movie", "tv", "auto")

    Returns:
        Diccionario con información del contenido encontrado o error

    Raises:
        HTTPException: Si la query está vacía o hay error en la búsqueda
    """
    try:
        if not query or len(query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query no puede estar vacío")

        content_id, detected_type = tmdb_service.search_content(query, content_type)

        if not content_id:
            return {
                "success": False,
                "message": "No se encontró contenido",
                "query": query
            }

        poster_url = tmdb_service.get_poster_url(content_id, detected_type)
        cast = tmdb_service.get_cast(content_id, detected_type)

        return {
            "success": True,
            "content_id": content_id,
            "type": detected_type,
            "cast": cast[:10],
            "query": query,
            "poster_url": poster_url
        }

    except Exception as e:
        logger.error(f"Error buscando contenido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/process-video-sse")
async def process_video_sse_get(
        filename: str,
        title: str,
        content_type: str = "auto"
):
    """
    Procesa vídeo previamente subido y envía progreso en tiempo real vía SSE.

    Args:
        filename: Nombre del archivo de vídeo ya subido
        title: Título para buscar en TMDB
        content_type: Tipo de contenido ("movie", "tv", "auto")

    Returns:
        EventSourceResponse con stream de eventos SSE
    """

    async def event_generator():
        """Generador de eventos SSE."""
        video_path = None
        cap = None
        processor = None
        session_id = str(uuid.uuid4())[:8]

        try:
            logger.info(f"[{session_id}] Iniciando procesamiento SSE: {filename}")

            video_path = os.path.join(VIDEOS_DIR, filename)
            if not os.path.exists(video_path):
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": f"Vídeo no encontrado: {filename}"
                    })
                }
                return

            if PERFORMANCE_MONITORING:
                performance_monitor.start_session(session_id)

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "info",
                    "message": f"Iniciando procesamiento de {filename}",
                    "progress": 5
                })
            }

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": "No se pudo abrir el vídeo"
                    })
                }
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            yield {
                "event": "info",
                "data": json.dumps({
                    "type": "info",
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "duration": round(duration, 2),
                    "filename": filename,
                    "optimizations": {
                        "face_detection_skip": FACE_DETECTION_SKIP,
                        "full_analysis_skip": FULL_ANALYSIS_SKIP,
                        "compression_enabled": True,
                        "graph_data_mode": True
                    }
                })
            }

            logger.info(f"[{session_id}] Vídeo - Frames: {total_frames}, FPS: {fps:.2f}, Duración: {duration:.2f}s")

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "tmdb",
                    "message": f"Buscando '{title}' en TMDB...",
                    "progress": 10
                })
            }

            content_id, detected_type = tmdb_service.search_content(title, content_type)
            poster_url = None

            if content_id:
                logger.info(f"[{session_id}] TMDB encontrado - ID: {content_id}")

                content_data = tmdb_service.get_content_data(content_id, detected_type)
                if content_data and content_data.get("poster_path"):
                    poster_url = f"{TMDB_IMAGE_BASE}{content_data['poster_path']}"

                cast_list = tmdb_service.get_cast(content_id, detected_type)
                actors_data = []

                processor = get_processor_instance()

                for i, actor in enumerate(cast_list[:15]):
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "type": "actors",
                            "message": f"Cargando actor: {actor['name']}",
                            "actor_name": actor['name'],
                            "actor_character": actor.get('character', ''),
                            "progress": 15 + (i * 2)
                        })
                    }

                    profile_path = actor.get("profile_path")
                    if profile_path:
                        actor_image = tmdb_service.load_actor_image(profile_path)
                        if actor_image is not None:
                            encoding = FaceRecognizer.extract_encoding_from_image(actor_image)
                            if encoding is not None:
                                processor.add_known_face(
                                    encoding=encoding,
                                    actor_id=actor["id"],
                                    nombre=actor["name"],
                                    personaje=actor.get("character", "Desconocido"),
                                    foto_url=f"{TMDB_IMAGE_BASE}{profile_path}"
                                )
                                actors_data.append({
                                    "id": actor["id"],
                                    "name": actor["name"],
                                    "character": actor.get("character", "Desconocido")
                                })
                                logger.info(f"   {actor['name']} cargado")
                            else:
                                logger.warning(f"   No se pudo extraer encoding: {actor['name']}")
                        else:
                            logger.warning(f"   No se pudo cargar imagen: {actor['name']}")

                logger.info(f"[{session_id}] {len(actors_data)} actores cargados con éxito")
            else:
                logger.warning(f"[{session_id}] No se encontró contenido en TMDB: {title}")
                processor = get_processor_instance()

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "processing",
                    "message": "Analizando vídeo...",
                    "progress": 35
                })
            }

            frame_count = 0
            last_update = 0
            start_time = time.time()
            last_faces = []

            while True:
                frame_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    break

                detect_faces = should_run_analysis(frame_count, "face_detection")
                full_analysis = should_run_analysis(frame_count, "shot_analysis")

                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    executor,
                    processor.process_frame_optimized,
                    frame,
                    frame_count,
                    detect_faces,
                    full_analysis,
                    last_faces
                )

                if results.get("faces"):
                    last_faces = results["faces"]

                if frame_count - last_update >= PROGRESS_UPDATE_SKIP:
                    progress = 35 + int((frame_count / total_frames) * 60)
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0

                    frame_data = None
                    if detect_faces or full_analysis:
                        try:
                            from app.utils.visualization_utils import apply_all_overlays

                            shot_type_value = None
                            if results.get("shot_type"):
                                st = results["shot_type"]
                                if isinstance(st, dict):
                                    shot_type_value = st.get("shot_type", st.get("type"))
                                else:
                                    shot_type_value = str(st)

                            viz_results = {
                                "faces": [],
                                "composition": results.get("composition"),
                                "lighting": results.get("lighting"),
                                "camera_movement": results.get("camera_movement"),
                                "shot_type": shot_type_value,
                                "colors": results.get("colors")
                            }

                            for face_info in results.get("faces", []):
                                face_viz = {
                                    "box": face_info.get("box"),
                                    "recognized": face_info.get("recognized", False)
                                }

                                if face_info.get("recognized"):
                                    face_viz["name"] = f"{face_info['nombre']} ({face_info['similitud']:.0f}%)"

                                if "emotion" in face_info:
                                    emo = face_info["emotion"]
                                    if isinstance(emo, dict):
                                        face_viz["emotion"] = emo.get("emotion", "neutral")
                                        face_viz["emotion_confidence"] = emo.get("confidence", 0.0)
                                    else:
                                        face_viz["emotion"] = str(emo)
                                        face_viz["emotion_confidence"] = 0.0

                                viz_results["faces"].append(face_viz)

                            frame_annotated = apply_all_overlays(
                                frame=frame.copy(),
                                results=viz_results,
                                show_faces=True,
                                show_rule_of_thirds=full_analysis,
                                show_lighting=viz_results["lighting"] is not None,
                                show_camera_movement=viz_results["camera_movement"] is not None,
                                show_shot_type=viz_results["shot_type"] is not None,
                                show_color_palette=viz_results["colors"] is not None,
                                opacity=0.8
                            )

                            frame_data = frame_to_base64(frame_annotated, JPEG_QUALITY, MAX_FRAME_WIDTH)

                        except Exception as e:
                            logger.warning(f"Error aplicando visualizaciones frame {frame_count}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            frame_data = frame_to_base64(frame, JPEG_QUALITY, MAX_FRAME_WIDTH)

                    faces_data = []
                    for face_info in results.get("faces", []):
                        face_dict = {
                            "box": face_info["box"],
                            "recognized": face_info["recognized"]
                        }

                        if face_info["recognized"]:
                            face_dict.update({
                                "actor": face_info["nombre"],
                                "personaje": face_info["personaje"],
                                "similitud": round(face_info["similitud"], 2)
                            })

                        if "emotion" in face_info:
                            face_dict["emotion"] = {
                                "emotion": face_info["emotion"]["emotion"],
                                "confidence": round(face_info["emotion"]["confidence"], 2)
                            }

                        faces_data.append(face_dict)

                    event_data = {
                        "frame_number": frame_count,
                        "total_frames": total_frames,
                        "progress": min(progress, 95),
                        "frame_data": frame_data,
                        "faces_detected": len(results.get("faces", [])),
                        "faces": faces_data,
                        "fps_processing": round(fps_processing, 2),
                        "eta_seconds": round(eta, 1)
                    }

                    for key in ["shot_type", "composition", "lighting", "colors", "camera_movement"]:
                        if key in results and results[key] is not None:
                            event_data[key] = results[key]

                    yield {
                        "event": "frame",
                        "data": json.dumps(convert_to_serializable(event_data))
                    }

                    last_update = frame_count
                    await asyncio.sleep(0.01)

                if PERFORMANCE_MONITORING:
                    frame_time = time.time() - frame_start
                    performance_monitor.record_frame(session_id, frame_count, frame_time)

                frame_count += 1

            cap.release()

            processing_time = time.time() - start_time
            logger.info(f"[{session_id}] Procesamiento completado en {processing_time:.2f}s")

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "finalizing",
                    "message": "Generando resultados finales...",
                    "progress": 98
                })
            }

            final_results = processor.get_final_results()

            yield {
                "event": "complete",
                "data": json.dumps(convert_to_serializable({
                    "progress": 100,
                    "message": "Análisis completado exitosamente",
                    "processing_time": round(processing_time, 2),
                    "total_frames_processed": frame_count,
                    "detected_actors": final_results["detected_actors"],
                    "total_actors_detected": final_results["total_actors_detected"],
                    "camera_summary": final_results.get("camera_movement_summary"),
                    "shot_types_summary": final_results.get("shot_types_summary"),
                    "lighting_summary": final_results.get("lighting_summary"),
                    "emotions_summary": final_results.get("emotions_summary"),
                    "color_analysis_summary": final_results.get("color_analysis_summary"),
                    "composition_summary": final_results.get("composition_summary"),
                    "poster_url": poster_url,
                    "histogram_data": final_results.get("histogram_data"),
                    "camera_timeline": final_results.get("camera_timeline"),
                    "composition_data": final_results.get("composition_data")
                }))
            }

            logger.info(f"[{session_id}] Vídeo procesado exitosamente: {filename}")

            if PERFORMANCE_MONITORING:
                performance_monitor.end_session(session_id)

        except CVFlixException as e:
            logger.error(f"[{session_id}] Error CVFlix: {e.message}")
            yield {
                "event": "error",
                "data": json.dumps(e.to_dict())
            }
            if PERFORMANCE_MONITORING:
                performance_monitor.record_error(session_id)

        except Exception as e:
            logger.error(f"[{session_id}] Error procesando vídeo: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "message": "Error durante el procesamiento",
                    "type": type(e).__name__
                })
            }
            if PERFORMANCE_MONITORING:
                performance_monitor.record_error(session_id)

        finally:
            if cap:
                cap.release()

            if processor:
                processor.reset()

            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"[{session_id}] Vídeo temporal eliminado")
                except Exception as e:
                    logger.warning(f"[{session_id}] Error eliminando vídeo: {e}")

            logger.info(f"[{session_id}] Sesión finalizada")

    return EventSourceResponse(event_generator())


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Sube un vídeo al servidor.

    Args:
        file: Archivo de vídeo

    Returns:
        Información del vídeo subido

    Raises:
        HTTPException: Si el formato no está soportado o hay error al guardar
    """
    try:
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(
                status_code=400,
                detail="Formato no soportado. Use: .mp4, .avi, .mov, .mkv"
            )

        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(VIDEOS_DIR, unique_filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        video_info = get_video_info(file_path)

        logger.info(f"Vídeo subido: {unique_filename}")

        return {
            "success": True,
            "filename": unique_filename,
            "original_filename": file.filename,
            "size_mb": round(len(content) / (1024 * 1024), 2),
            "video_info": video_info
        }

    except Exception as e:
        logger.error(f"Error subiendo vídeo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/image-proxy")
async def image_proxy(url: str):
    """
    Proxy para imágenes de TMDB para evitar problemas de CORS.

    Args:
        url: URL de la imagen

    Returns:
        Response con contenido de la imagen

    Raises:
        HTTPException: Si la imagen no se puede obtener
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            return Response(
                content=response.content,
                media_type=response.headers.get("content-type", "image/jpeg")
            )
    except Exception as e:
        logger.error(f"Error en proxy de imagen: {e}")
        raise HTTPException(status_code=404, detail="Imagen no encontrada")


@app.post("/api/process-video-sse")
async def process_video_sse(
    file: UploadFile = File(...),
    title: str = Form(...),
    content_type: str = Form("movie")
):
    """
    Procesa vídeo y envía progreso en tiempo real vía SSE.

    Args:
        file: Archivo de vídeo
        title: Título para buscar en TMDB
        content_type: Tipo de contenido ("movie", "tv")

    Returns:
        EventSourceResponse con stream de eventos SSE
    """

    async def event_generator():
        """Generador de eventos SSE."""
        video_path = None
        cap = None
        processor = None
        session_id = str(uuid.uuid4())[:8]

        try:
            logger.info(f"[{session_id}] Iniciando procesamiento SSE: {file.filename}")

            if PERFORMANCE_MONITORING:
                performance_monitor.start_session(session_id)

            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": "Formato de vídeo no soportado",
                        "supported": [".mp4", ".avi", ".mov", ".mkv"]
                    })
                }
                return

            file_extension = os.path.splitext(file.filename)[1]
            video_filename = f"{session_id}{file_extension}"
            video_path = os.path.join(VIDEOS_DIR, video_filename)

            with open(video_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "info",
                    "message": f"Vídeo guardado: {file.filename}",
                    "progress": 5
                })
            }

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield {
                    "event": "error",
                    "data": json.dumps({
                        "error": "No se pudo abrir el vídeo"
                    })
                }
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            yield {
                "event": "video_info",
                "data": json.dumps({
                    "total_frames": total_frames,
                    "fps": round(fps, 2),
                    "duration": round(duration, 2),
                    "filename": file.filename
                })
            }

            logger.info(f"[{session_id}] Vídeo - Frames: {total_frames}, FPS: {fps:.2f}, Duración: {duration:.2f}s")

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "tmdb",
                    "message": f"Buscando '{title}' en TMDB...",
                    "progress": 10
                })
            }

            content_id, detected_type = tmdb_service.search_content(title, content_type)
            poster_url = None

            if content_id:
                yield {
                    "event": "tmdb_found",
                    "data": json.dumps({
                        "content_id": content_id,
                        "type": detected_type,
                        "title": title
                    })
                }

                logger.info(f"[{session_id}] TMDB encontrado - ID: {content_id}")

                content_data = tmdb_service.get_content_data(content_id, detected_type)
                if content_data and content_data.get("poster_path"):
                    poster_url = f"{TMDB_IMAGE_BASE}{content_data['poster_path']}"

                cast_list = tmdb_service.get_cast(content_id, detected_type)

                processor = get_processor_instance()

                for i, actor in enumerate(cast_list[:15]):
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "type": "actors",
                            "message": f"Cargando actor: {actor['name']}",
                            "actor_name": actor['name'],
                            "actor_character": actor.get('character', ''),
                            "progress": 15 + (i * 2)
                        })
                    }

                    profile_path = actor.get("profile_path")
                    if profile_path:
                        image = tmdb_service.load_actor_image(profile_path)
                        if image is not None:
                            encoding = FaceRecognizer.extract_encoding_from_image(image)

                            if encoding is not None:
                                processor.add_known_face(
                                    encoding=encoding,
                                    actor_id=actor["id"],
                                    nombre=actor["name"],
                                    personaje=actor.get("character", "Desconocido"),
                                    foto_url=f"{TMDB_IMAGE_BASE}{profile_path}"
                                )
                                logger.debug(f"Encoding añadido: {actor['name']}")
                            else:
                                logger.warning(f"No se detectó cara en la imagen de: {actor['name']}")
                        else:
                            logger.warning(f"No se pudo cargar imagen de: {actor['name']}")

                    await asyncio.sleep(0.01)

                logger.info(f"[{session_id}] Actores cargados en procesador")

            else:
                logger.warning(f"[{session_id}] No se encontró contenido en TMDB: {title}")
                processor = get_processor_instance()

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "processing",
                    "message": "Iniciando análisis de vídeo...",
                    "progress": 35
                })
            }

            frame_count = 0
            last_update = 0
            start_time = time.time()
            last_faces = []
            loop = asyncio.get_event_loop()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_start = time.time()

                detect_faces = should_run_analysis(frame_count, "face_detection")
                full_analysis = should_run_analysis(frame_count, "shot_type")

                results = await loop.run_in_executor(
                    executor,
                    processor.process_frame_optimized,
                    frame,
                    frame_count,
                    detect_faces,
                    full_analysis,
                    last_faces
                )

                if results.get("faces"):
                    last_faces = results["faces"]

                if frame_count - last_update >= PROGRESS_UPDATE_SKIP:
                    progress = 35 + int((frame_count / total_frames) * 60)
                    elapsed = time.time() - start_time
                    fps_processing = frame_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0

                    frame_data = None
                    if detect_faces or full_analysis:
                        try:
                            from app.utils.visualization_utils import apply_all_overlays

                            shot_type_value = None
                            if results.get("shot_type"):
                                st = results["shot_type"]
                                if isinstance(st, dict):
                                    shot_type_value = st.get("shot_type", st.get("type"))
                                else:
                                    shot_type_value = str(st)

                            viz_results = {
                                "faces": [],
                                "composition": results.get("composition"),
                                "lighting": results.get("lighting"),
                                "camera_movement": results.get("camera_movement"),
                                "shot_type": shot_type_value,
                                "colors": results.get("colors")
                            }

                            for face_info in results.get("faces", []):
                                face_viz = {
                                    "box": face_info.get("box"),
                                    "recognized": face_info.get("recognized", False)
                                }

                                if face_info.get("recognized"):
                                    face_viz["name"] = f"{face_info['nombre']} ({face_info['similitud']:.0f}%)"

                                if "emotion" in face_info:
                                    emo = face_info["emotion"]
                                    if isinstance(emo, dict):
                                        face_viz["emotion"] = emo.get("emotion", "neutral")
                                        face_viz["emotion_confidence"] = emo.get("confidence", 0.0)
                                    else:
                                        face_viz["emotion"] = str(emo)
                                        face_viz["emotion_confidence"] = 0.0

                                viz_results["faces"].append(face_viz)

                            frame_annotated = apply_all_overlays(
                                frame=frame.copy(),
                                results=viz_results,
                                show_faces=True,
                                show_rule_of_thirds=full_analysis,
                                show_lighting=viz_results["lighting"] is not None,
                                show_camera_movement=viz_results["camera_movement"] is not None,
                                show_shot_type=viz_results["shot_type"] is not None,
                                show_color_palette=viz_results["colors"] is not None,
                                opacity=0.8
                            )

                            frame_data = frame_to_base64(frame_annotated, JPEG_QUALITY, MAX_FRAME_WIDTH)

                        except Exception as e:
                            logger.warning(f"Error aplicando visualizaciones frame {frame_count}: {e}")
                            import traceback
                            logger.error(traceback.format_exc())
                            frame_data = frame_to_base64(frame, JPEG_QUALITY, MAX_FRAME_WIDTH)

                    faces_data = []
                    for face_info in results.get("faces", []):
                        face_dict = {
                            "box": face_info["box"],
                            "recognized": face_info["recognized"]
                        }

                        if face_info["recognized"]:
                            face_dict.update({
                                "actor": face_info["nombre"],
                                "personaje": face_info["personaje"],
                                "similitud": round(face_info["similitud"], 2)
                            })

                        if "emotion" in face_info:
                            face_dict["emotion"] = {
                                "emotion": face_info["emotion"]["emotion"],
                                "confidence": round(face_info["emotion"]["confidence"], 2)
                            }

                        faces_data.append(face_dict)

                    event_data = {
                        "frame_number": frame_count,
                        "total_frames": total_frames,
                        "progress": min(progress, 95),
                        "frame_data": frame_data,
                        "faces_detected": len(results.get("faces", [])),
                        "faces": faces_data,
                        "fps_processing": round(fps_processing, 2),
                        "eta_seconds": round(eta, 1)
                    }

                    for key in ["shot_type", "composition", "lighting", "colors", "camera_movement"]:
                        if key in results and results[key] is not None:
                            event_data[key] = results[key]

                    yield {
                        "event": "frame",
                        "data": json.dumps(convert_to_serializable(event_data))
                    }

                    last_update = frame_count
                    await asyncio.sleep(0.01)

                if PERFORMANCE_MONITORING:
                    frame_time = time.time() - frame_start
                    performance_monitor.record_frame(session_id, frame_count, frame_time)

                frame_count += 1

            cap.release()

            processing_time = time.time() - start_time
            logger.info(f"[{session_id}] Procesamiento completado en {processing_time:.2f}s")

            yield {
                "event": "progress",
                "data": json.dumps({
                    "type": "finalizing",
                    "message": "Generando resultados finales...",
                    "progress": 98
                })
            }

            final_results = processor.get_final_results()

            yield {
                "event": "complete",
                "data": json.dumps(convert_to_serializable({
                    "progress": 100,
                    "message": "Análisis completado exitosamente",
                    "processing_time": round(processing_time, 2),
                    "total_frames_processed": frame_count,
                    "detected_actors": final_results["detected_actors"],
                    "total_actors_detected": final_results["total_actors_detected"],
                    "camera_summary": final_results.get("camera_movement_summary"),
                    "shot_types_summary": final_results.get("shot_types_summary"),
                    "lighting_summary": final_results.get("lighting_summary"),
                    "emotions_summary": final_results.get("emotions_summary"),
                    "color_analysis_summary": final_results.get("color_analysis_summary"),
                    "composition_summary": final_results.get("composition_summary"),
                    "poster_url": poster_url,
                    "histogram_data": final_results.get("histogram_data"),
                    "camera_timeline": final_results.get("camera_timeline"),
                    "composition_data": final_results.get("composition_data")
                }))
            }

            logger.info(f"[{session_id}] Vídeo procesado exitosamente: {file.filename}")

            if PERFORMANCE_MONITORING:
                performance_monitor.end_session(session_id)

        except CVFlixException as e:
            logger.error(f"[{session_id}] Error CVFlix: {e.message}")
            yield {
                "event": "error",
                "data": json.dumps(e.to_dict())
            }
            if PERFORMANCE_MONITORING:
                performance_monitor.record_error(session_id)

        except Exception as e:
            logger.error(f"[{session_id}] Error procesando vídeo: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "message": "Error durante el procesamiento",
                    "type": type(e).__name__
                })
            }
            if PERFORMANCE_MONITORING:
                performance_monitor.record_error(session_id)

        finally:
            if cap:
                cap.release()

            if processor:
                processor.reset()

            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"[{session_id}] Vídeo temporal eliminado")
                except Exception as e:
                    logger.warning(f"[{session_id}] Error eliminando vídeo: {e}")

            logger.info(f"[{session_id}] Sesión finalizada")

    return EventSourceResponse(event_generator())


@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación."""
    logger.info("=" * 70)
    logger.info("CVFlix API v4.0.0 - Análisis Cinematográfico Profesional")
    logger.info("=" * 70)

    try:
        load_models()
    except Exception as e:
        logger.error(f"Error crítico cargando modelos: {e}")
        raise

    logger.info("Servicios inicializados")
    logger.info(f"Directorio de vídeos: {VIDEOS_DIR}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info(f"Detección facial cada: {FACE_DETECTION_SKIP} frames")
    logger.info(f"Análisis completo cada: {FULL_ANALYSIS_SKIP} frames")
    logger.info("Proxy de imágenes: habilitado")
    logger.info("Streaming SSE: habilitado")

    if PERFORMANCE_MONITORING:
        logger.info("Monitoreo de rendimiento: habilitado")
    if CACHE_ENABLED:
        logger.info("Sistema de caché: habilitado")
    if ERROR_HANDLERS:
        logger.info("Manejo de errores: habilitado")

    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación."""
    logger.info("\n" + "=" * 70)
    logger.info("Cerrando CVFlix API...")
    logger.info("=" * 70)

    executor.shutdown(wait=True)
    logger.info("Executor cerrado")

    logger.info("=" * 70)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )