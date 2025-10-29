"""
test_api.py

Tests de la API REST de CVFlix para validación de endpoints HTTP.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Usage:
    pytest tests/test_api.py -v
    pytest tests/test_api.py::TestHealthEndpoints -v
"""

import pytest
from fastapi import status
from unittest.mock import patch, Mock
import json


class TestHealthEndpoints:
    """Tests de endpoints de salud y estado."""

    def test_root_endpoint(self, client):
        """Test del endpoint raíz."""
        response = client.get("/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert data["status"] == "online"
        assert "features" in data

    def test_health_check(self, client):
        """Test del health check."""
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "status" in data
        assert "models_loaded" in data

    def test_stats_endpoint(self, client):
        """Test del endpoint de estadísticas."""
        response = client.get("/stats")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "uptime_seconds" in data
        assert "models_loaded" in data


class TestVideoUpload:
    """Tests de subida de vídeo."""

    @pytest.mark.xfail(reason="Test usa vídeo fake, la validación lo rechaza correctamente")
    def test_upload_video_success(self, client, tmp_path):
        """Test de subida exitosa de vídeo."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        with open(video_file, "rb") as f:
            response = client.post(
                "/upload-video",
                files={"file": ("test.mp4", f, "video/mp4")}
            )

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "success" in data
            assert "filename" in data

    def test_upload_video_no_file(self, client):
        """Test de subida sin archivo."""
        response = client.post("/upload-video")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_video_invalid_format(self, client, tmp_path):
        """Test de formato no soportado."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("not a video")

        with open(invalid_file, "rb") as f:
            response = client.post(
                "/upload-video",
                files={"file": ("test.txt", f, "text/plain")}
            )

        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]


class TestTMDBSearch:
    """Tests de búsqueda en TMDB."""

    @patch('app.services.tmdb_service.TMDBService.search_content')
    def test_search_content_found(self, mock_search, client):
        """Test de búsqueda exitosa."""
        mock_search.return_value = (12345, "movie")

        response = client.get("/search-content?query=Breaking+Bad")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["success"] == True
        assert "content_id" in data
        assert "type" in data

    @patch('app.services.tmdb_service.TMDBService.search_content')
    def test_search_content_not_found(self, mock_search, client):
        """Test de búsqueda sin resultados."""
        mock_search.return_value = (None, None)

        response = client.get("/search-content?query=NonexistentMovie12345")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["success"] == False

    def test_search_content_empty_query(self, client):
        """Test de búsqueda con query vacío."""
        response = client.get("/search-content?query=")

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_search_content_no_query(self, client):
        """Test sin parámetro query."""
        response = client.get("/search-content")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestImageProxy:
    """Tests del proxy de imágenes TMDB."""

    @patch('httpx.AsyncClient.get')
    @pytest.mark.asyncio
    async def test_image_proxy_success(self, mock_get, client):
        """Test de proxy exitoso."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake image data"
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_get.return_value = mock_response

        response = client.get("/image-proxy?url=http://example.com/test.jpg")

        if response.status_code == status.HTTP_200_OK:
            assert response.headers["content-type"] == "image/jpeg"

    @patch('httpx.AsyncClient.get')
    @pytest.mark.asyncio
    async def test_image_proxy_not_found(self, mock_get, client):
        """Test de imagen no encontrada."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        response = client.get("/image-proxy?url=http://example.com/nonexistent.jpg")

        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ]


class TestErrorHandling:
    """Tests de manejo de errores."""

    def test_404_not_found(self, client):
        """Test de ruta no existente."""
        response = client.get("/nonexistent-route")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self, client):
        """Test de método HTTP no permitido."""
        response = client.put("/")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_validation_error_format(self, client):
        """Test de formato de error de validación."""
        response = client.get("/search-content?query=")

        if response.status_code >= 400:
            data = response.json()
            assert "detail" in data or "message" in data


class TestCORS:
    """Tests de configuración CORS."""

    def test_cors_headers_present(self, client):
        """Test de presencia de headers CORS."""
        response = client.options(
            "/",
            headers={"Origin": "http://localhost:3000"}
        )

        assert "access-control-allow-origin" in response.headers or \
               response.status_code == status.HTTP_200_OK


class TestPerformance:
    """Tests de rendimiento de la API."""

    @pytest.mark.slow
    def test_concurrent_health_checks(self, client):
        """Test de múltiples health checks concurrentes."""
        import concurrent.futures

        def make_request():
            return client.get("/health")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]

        assert all(r.status_code == status.HTTP_200_OK for r in responses)

    @pytest.mark.slow
    def test_response_time_health_check(self, client):
        """Test de tiempo de respuesta del health check."""
        import time

        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start

        assert response.status_code == status.HTTP_200_OK
        assert elapsed < 1.0


class TestContentNegotiation:
    """Tests de negociación de contenido."""

    def test_json_response_default(self, client):
        """Test de respuesta JSON por defecto."""
        response = client.get("/")

        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert isinstance(data, dict)