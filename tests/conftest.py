"""
conftest.py

Configuración global de tests para CVFlix con fixtures compartidas y
configuración de pytest.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Usage:
    pytest tests/
    pytest tests/test_api.py -v
    pytest tests/ -m "not slow"
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, MagicMock
import tempfile
import shutil

from app.main import app
from app.services.video_processor import VideoProcessor
from app.analysis.face_detection import FaceDetector, FaceRecognizer


def pytest_configure(config):
    """Configuración inicial de pytest."""
    config.addinivalue_line(
        "markers", "slow: marca tests lentos que requieren más tiempo"
    )
    config.addinivalue_line(
        "markers", "integration: marca tests de integración"
    )
    config.addinivalue_line(
        "markers", "unit: marca tests unitarios"
    )


@pytest.fixture
def client():
    """Cliente de test para la API."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def async_client():
    """Cliente asíncrono de test."""
    from httpx import AsyncClient

    async def _get_client():
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac

    return _get_client


@pytest.fixture
def sample_frame():
    """Frame de ejemplo (640x480 RGB)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:200, 100:200] = [128, 128, 128]
    return frame


@pytest.fixture
def sample_frame_with_face():
    """Frame con región que simula una cara."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:300, 250:350] = [200, 180, 160]
    return frame


@pytest.fixture
def sample_video_path(tmp_path):
    """Path a vídeo temporal de prueba."""
    import cv2

    video_path = tmp_path / "test_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (i * 25, i * 25, i * 25)
        out.write(frame)

    out.release()

    yield video_path

    if video_path.exists():
        video_path.unlink()


@pytest.fixture
def mock_video_info():
    """Mock de información de vídeo."""
    return {
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "total_frames": 1500,
        "duration": 50.0,
        "codec": 828601953
    }


@pytest.fixture
def mock_face_detector():
    """Mock de FaceDetector."""
    detector = Mock(spec=FaceDetector)

    detector.detect_faces.return_value = (
        [(100, 300, 200, 200)],
        [(200, 100, 300, 200)]
    )

    return detector


@pytest.fixture
def mock_face_recognizer():
    """Mock de FaceRecognizer."""
    recognizer = Mock(spec=FaceRecognizer)

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
    """Instancia real de VideoProcessor para tests de integración."""
    processor = VideoProcessor()
    yield processor
    processor.reset()


@pytest.fixture
def mock_video_processor():
    """Mock de VideoProcessor."""
    processor = Mock(spec=VideoProcessor)

    processor.process_frame_optimized.return_value = {
        "frame_number": 1,
        "faces": [],
        "shot_type": {"shot_type": "Primer Plano", "confidence": 0.8},
        "camera_movement": {"movement_type": "Estático", "intensity": 0},
    }

    processor.get_final_results.return_value = {
        "detected_actors": [],
        "total_actors_detected": 0,
        "total_frames_analyzed": 100
    }

    return processor


@pytest.fixture
def mock_actor_data():
    """Datos de actor de prueba."""
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
    """Lista de actores de prueba."""
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


@pytest.fixture
def mock_tmdb_service():
    """Mock de TMDBService."""
    from app.services.tmdb_service import TMDBService

    service = Mock(spec=TMDBService)

    service.search_content.return_value = (12345, "movie")

    service.get_cast.return_value = [
        {
            "id": 1,
            "name": "Actor Name",
            "character": "Character Name",
            "profile_path": "/path.jpg"
        }
    ]

    service.load_actor_image.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    return service


@pytest.fixture
def temp_videos_dir(tmp_path):
    """Directorio temporal para vídeos."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    yield videos_dir
    shutil.rmtree(videos_dir, ignore_errors=True)


@pytest.fixture
def mock_config(monkeypatch):
    """Mock de configuración."""
    monkeypatch.setenv("TMDB_API_KEY", "test_api_key")
    monkeypatch.setenv("LOG_LEVEL", "ERROR")


@pytest.fixture
def mock_websocket():
    """Mock de WebSocket."""
    ws = Mock()
    ws.accept = asyncio.coroutine(lambda: None)
    ws.send_json = asyncio.coroutine(lambda x: None)
    ws.receive_json = asyncio.coroutine(lambda: {"filename": "test.mp4"})
    ws.close = asyncio.coroutine(lambda: None)
    return ws


@pytest.fixture
def assert_frame_valid():
    """Helper para validar frames."""
    def _assert(frame):
        assert isinstance(frame, np.ndarray)
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8

    return _assert


@pytest.fixture
def create_test_video():
    """Factory para crear vídeos de prueba."""
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


@pytest.fixture(scope="session")
def event_loop():
    """Event loop para tests asíncronos."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()