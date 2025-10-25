"""
==============================================================================
Pytest Configuration - Configuración Global de Tests
==============================================================================
Fixtures y configuración compartida para todos los tests
==============================================================================
"""
import pytest
import asyncio
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

# Imports de la app
from app.main import app
from app.services.video_processor import VideoProcessor
from app.analysis.face_detection import FaceDetector, FaceRecognizer


# ==================== CONFIGURACIÓN DE PYTEST ====================

def pytest_configure(config):
    """Configuración inicial de pytest"""
    config.addinivalue_line(
        "markers", "slow: marca tests lentos que requieren más tiempo"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "unit: marca tests unitarios"
    )


# ==================== FIXTURES DE CLIENTE ====================

@pytest.fixture
def client():
    """Cliente de test para la API"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def async_client():
    """Cliente asíncrono de test"""
    from httpx import AsyncClient

    async def _get_client():
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    return _get_client


# ==================== FIXTURES DE VIDEO ====================

@pytest.fixture
def sample_frame():
    """Frame de ejemplo (640x480 RGB)"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Agregar contenido para que no sea todo negro
    frame[100:200, 100:200] = [128, 128, 128]  # Cuadrado gris
    return frame


@pytest.fixture
def sample_frame_with_face():
    """Frame con región que simula una cara"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Simular región facial (rectángulo más claro)
    frame[150:300, 250:350] = [200, 180, 160]  # Tono piel
    return frame


@pytest.fixture
def sample_video_path(tmp_path):
    """Path a video temporal de prueba"""
    import cv2

    video_path = tmp_path / "test_video.mp4"

    # Crear video de prueba (10 frames, 640x480, 30fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (i * 25, i * 25, i * 25)  # Fade de negro a blanco
        out.write(frame)

    out.release()

    yield video_path

    # Cleanup
    if video_path.exists():
        video_path.unlink()


@pytest.fixture
def mock_video_info():
    """Mock de información de video"""
    return {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "total_frames": 1500,
        "duration": 50.0,
        "codec": 828601953
    }


# ==================== FIXTURES DE MODELOS ====================

@pytest.fixture
def mock_face_detector():
    """Mock de FaceDetector"""
    detector = Mock(spec=FaceDetector)

    # Simular detección exitosa
    detector.detect_faces.return_value = (
        [(100, 300, 200, 200)],  # face_locations
        [(200, 100, 300, 200)]   # face_boxes
    )

    return detector


@pytest.fixture
def mock_face_recognizer():
    """Mock de FaceRecognizer"""
    recognizer = Mock(spec=FaceRecognizer)

    # Simular reconocimiento exitoso
    recognizer.recognize_faces.return_value = [
        {
            "actor_id": 1,
            "nombre": "Test Actor",
            "personaje": "Test Character",
            "similitud": 95.5,
            "foto_url": "http://test.com/photo.jpg"
        }
    ]

    recognizer.extract_encodings.return_value = [np.random.rand(128)]

    return recognizer


@pytest.fixture
def video_processor():
    """Instancia real de VideoProcessor para tests de integración"""
    processor = VideoProcessor()
    yield processor
    processor.reset()


@pytest.fixture
def mock_video_processor():
    """Mock de VideoProcessor"""
    processor = Mock(spec=VideoProcessor)

    # Simular procesamiento de frame
    processor.process_frame_optimized.return_value = {
        "frame_number": 1,
        "faces": [],
        "shot_type": {"shot_type": "Primer Plano", "confidence": 0.8},
        "camera_movement": {"movement_type": "Estático", "intensity": 0},
    }

    # Simular resultados finales
    processor.get_final_results.return_value = {
        "detected_actors": [],
        "total_actors_detected": 0,
        "total_frames_analyzed": 100
    }

    return processor


# ==================== FIXTURES DE ACTORES (TMDB) ====================

@pytest.fixture
def mock_actor_data():
    """Datos de actor de prueba"""
    return {
        "id": 12345,
        "nombre": "John Doe",
        "personaje": "Main Character",
        "foto_url": "http://image.tmdb.org/t/p/w500/test.jpg",
        "profile_path": "/test.jpg",
        "encoding": np.random.rand(128)
    }


@pytest.fixture
def mock_actors_list():
    """Lista de actores de prueba"""
    return [
        {
            "id": 1,
            "name": "Actor One",
            "character": "Character One",
            "profile_path": "/actor1.jpg"
        },
        {
            "id": 2,
            "name": "Actor Two",
            "character": "Character Two",
            "profile_path": "/actor2.jpg"
        }
    ]


# ==================== FIXTURES DE TMDB ====================

@pytest.fixture
def mock_tmdb_service():
    """Mock de TMDBService"""
    from app.services.tmdb_service import TMDBService

    service = Mock(spec=TMDBService)

    # Simular búsqueda exitosa
    service.search_content.return_value = (12345, "movie")

    # Simular obtención de reparto
    service.get_cast.return_value = [
        {
            "id": 1,
            "name": "Actor Name",
            "character": "Character Name",
            "profile_path": "/path.jpg"
        }
    ]

    # Simular carga de imagen
    service.load_actor_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    return service


# ==================== FIXTURES DE ARCHIVOS TEMPORALES ====================

@pytest.fixture
def temp_videos_dir(tmp_path):
    """Directorio temporal para videos"""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    yield videos_dir
    shutil.rmtree(videos_dir, ignore_errors=True)


# ==================== FIXTURES DE CONFIGURACIÓN ====================

@pytest.fixture
def mock_config(monkeypatch):
    """Mock de configuración"""
    monkeypatch.setenv("TMDB_API_KEY", "test_api_key")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")  # Reducir logs en tests


# ==================== FIXTURES DE WEBSOCKET ====================

@pytest.fixture
def mock_websocket():
    """Mock de WebSocket"""
    ws = Mock()
    ws.accept = asyncio.coroutine(lambda: None)
    ws.send_json = asyncio.coroutine(lambda x: None)
    ws.receive_json = asyncio.coroutine(lambda: {"filename": "test.mp4"})
    ws.close = asyncio.coroutine(lambda: None)
    return ws


# ==================== UTILIDADES ====================

@pytest.fixture
def assert_frame_valid():
    """Helper para validar frames"""
    def _assert(frame):
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3  # RGB
        assert frame.dtype == np.uint8

    return _assert


@pytest.fixture
def create_test_video():
    """Factory para crear videos de prueba"""
    def _create(path, num_frames=10, width=640, height=480, fps=30):
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            out.write(frame)

        out.release()
        return path

    return _create


# ==================== HOOKS ====================

@pytest.fixture(scope="session")
def event_loop():
    """Event loop para tests asíncronos"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()