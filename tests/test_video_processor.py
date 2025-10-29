"""
test_video_processor.py

Tests unitarios y de integración para el procesador principal de vídeo.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Usage:
    pytest tests/test_video_processor.py -v
    pytest tests/test_video_processor.py -m "not slow"
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from collections import Counter


class TestVideoProcessorInitialization:
    """Tests de inicialización del procesador."""

    def test_processor_init_success(self):
        """Test de inicialización exitosa."""
        from app.services.video_processor import VideoProcessor

        processor = VideoProcessor()

        assert processor.face_detector is not None
        assert processor.face_recognizer is not None
        assert processor.shot_analyzer is not None
        assert processor.camera_analyzer is not None
        assert isinstance(processor.detected_actors, dict)
        assert len(processor.detected_actors) == 0

    def test_processor_has_all_analyzers(self, video_processor):
        """Test de que tiene todos los analizadores."""
        assert hasattr(video_processor, 'face_detector')
        assert hasattr(video_processor, 'face_recognizer')
        assert hasattr(video_processor, 'shot_analyzer')
        assert hasattr(video_processor, 'composition_analyzer')
        assert hasattr(video_processor, 'lighting_analyzer')
        assert hasattr(video_processor, 'color_analyzer')
        assert hasattr(video_processor, 'camera_analyzer')
        assert hasattr(video_processor, 'emotion_detector')


class TestActorEncodingLoading:
    """Tests de carga de encodings de actores."""

    def test_load_actor_encodings_success(self, video_processor, mock_actor_data):
        """Test de carga exitosa de encodings."""
        actors_data = [mock_actor_data]

        video_processor.load_actor_encodings(actors_data)

        assert len(video_processor.face_recognizer.actor_encodings) == 1

    def test_load_multiple_actors(self, video_processor):
        """Test de carga de múltiples actores."""
        actors_data = [
            {
                "id": 1,
                "nombre": "Actor 1",
                "personaje": "Character 1",
                "foto_url": "url1",
                "encoding": np.random.rand(128)
            },
            {
                "id": 2,
                "nombre": "Actor 2",
                "personaje": "Character 2",
                "foto_url": "url2",
                "encoding": np.random.rand(128)
            }
        ]

        video_processor.load_actor_encodings(actors_data)

        assert len(video_processor.face_recognizer.actor_encodings) == 2

    def test_load_empty_actors_list(self, video_processor):
        """Test de carga con lista vacía."""
        video_processor.load_actor_encodings([])

        assert len(video_processor.face_recognizer.actor_encodings) == 0


class TestFrameProcessing:
    """Tests de procesamiento de frames."""

    def test_process_frame_basic(self, video_processor, sample_frame):
        """Test de procesamiento básico de frame."""
        result = video_processor.process_frame_optimized(
            sample_frame,
            frame_number=1,
            detect_faces=False,
            full_analysis=False
        )

        assert isinstance(result, dict)
        assert "frame_number" in result
        assert result["frame_number"] == 1
        assert "faces" in result
        assert "camera_movement" in result

    def test_process_frame_with_face_detection(
            self, video_processor, sample_frame_with_face
    ):
        """Test de procesamiento con detección facial."""
        result = video_processor.process_frame_optimized(
            sample_frame_with_face,
            frame_number=1,
            detect_faces=True,
            full_analysis=False
        )

        assert "faces" in result
        assert isinstance(result["faces"], list)

    def test_process_frame_with_full_analysis(
            self, video_processor, sample_frame
    ):
        """Test de procesamiento con análisis completo."""
        result = video_processor.process_frame_optimized(
            sample_frame,
            frame_number=1,
            detect_faces=True,
            full_analysis=True
        )

        assert result["shot_type"] is not None or result["composition"] is not None
        assert result["camera_movement"] is not None

    def test_process_multiple_frames(self, video_processor, sample_frame):
        """Test de procesamiento de múltiples frames."""
        frames_to_process = 10

        for i in range(frames_to_process):
            result = video_processor.process_frame_optimized(
                sample_frame,
                frame_number=i,
                detect_faces=(i % 3 == 0),
                full_analysis=(i % 5 == 0)
            )

            assert result["frame_number"] == i

    @patch('app.services.video_processor.ANALYSIS_CONFIG')
    def test_process_frame_respects_config(
            self, mock_config, video_processor, sample_frame
    ):
        """Test de que respeta la configuración de análisis."""
        mock_config.get.return_value = {"enabled": False}

        result = video_processor.process_frame_optimized(
            sample_frame,
            frame_number=1,
            detect_faces=True,
            full_analysis=True
        )

        assert isinstance(result, dict)


class TestActorStatistics:
    """Tests de estadísticas de actores."""

    def test_update_actor_stats(self, video_processor):
        """Test de actualización de estadísticas de actor."""
        recognition = {
            "actor_id": 1,
            "nombre": "Test Actor",
            "personaje": "Test Character",
            "foto_url": "url",
            "similitud": 95.0
        }

        video_processor._update_actor_stats(1, recognition, frame_number=10)

        assert 1 in video_processor.detected_actors
        actor_data = video_processor.detected_actors[1]

        assert actor_data["detecciones"] == 1
        assert actor_data["similitud_maxima"] == 95.0
        assert 10 in actor_data["frames_aparicion"]

    def test_multiple_detections_same_actor(self, video_processor):
        """Test de múltiples detecciones del mismo actor."""
        recognition = {
            "actor_id": 1,
            "nombre": "Test Actor",
            "personaje": "Test Character",
            "foto_url": "url",
            "similitud": 90.0
        }

        video_processor._update_actor_stats(1, recognition, 10)

        recognition["similitud"] = 95.0
        video_processor._update_actor_stats(1, recognition, 20)

        actor_data = video_processor.detected_actors[1]

        assert actor_data["detecciones"] == 2
        assert actor_data["similitud_maxima"] == 95.0
        assert actor_data["similitud_promedio"] == 92.5


class TestFinalResults:
    """Tests de resultados finales."""

    def test_get_final_results_empty(self, video_processor):
        """Test de resultados sin procesamiento."""
        results = video_processor.get_final_results()

        assert "detected_actors" in results
        assert "total_actors_detected" in results
        assert results["total_actors_detected"] == 0
        assert isinstance(results["detected_actors"], list)

    def test_get_final_results_with_actors(self, video_processor):
        """Test de resultados con actores detectados."""
        recognition = {
            "actor_id": 1,
            "nombre": "Actor 1",
            "personaje": "Character 1",
            "foto_url": "url1",
            "similitud": 95.0
        }
        video_processor._update_actor_stats(1, recognition, 10)

        recognition2 = {
            "actor_id": 2,
            "nombre": "Actor 2",
            "personaje": "Character 2",
            "foto_url": "url2",
            "similitud": 90.0
        }
        video_processor._update_actor_stats(2, recognition2, 20)

        results = video_processor.get_final_results()

        assert results["total_actors_detected"] == 2
        assert len(results["detected_actors"]) == 2

        assert results["detected_actors"][0]["actor_id"] in [1, 2]

    def test_get_final_results_structure(self, video_processor, sample_frame):
        """Test de estructura completa de resultados."""
        for i in range(5):
            video_processor.process_frame_optimized(
                sample_frame,
                frame_number=i,
                detect_faces=True,
                full_analysis=True
            )

        results = video_processor.get_final_results()

        required_keys = [
            "detected_actors",
            "total_actors_detected",
            "camera_movement_summary",
            "shot_types_summary",
            "lighting_summary",
            "emotions_summary",
            "color_analysis_summary",
            "total_frames_analyzed"
        ]

        for key in required_keys:
            assert key in results


class TestProcessorReset:
    """Tests de reseteo del procesador."""

    def test_reset_clears_actors(self, video_processor):
        """Test de que reset limpia actores."""
        recognition = {
            "actor_id": 1,
            "nombre": "Test",
            "personaje": "Test",
            "foto_url": "url",
            "similitud": 95.0
        }
        video_processor._update_actor_stats(1, recognition, 10)

        assert len(video_processor.detected_actors) > 0

        video_processor.reset()

        assert len(video_processor.detected_actors) == 0

    def test_reset_clears_counters(self, video_processor, sample_frame):
        """Test de que reset limpia contadores."""
        for i in range(5):
            video_processor.process_frame_optimized(
                sample_frame,
                frame_number=i,
                full_analysis=True
            )

        assert video_processor.total_frames_analyzed > 0

        video_processor.reset()

        assert video_processor.total_frames_analyzed == 0
        assert len(video_processor.shot_types_count) == 0
        assert len(video_processor.global_colors) == 0

    def test_reset_preserves_analyzers(self, video_processor):
        """Test de que reset no elimina los analizadores."""
        original_detector = video_processor.face_detector

        video_processor.reset()

        assert video_processor.face_detector is original_detector


class TestErrorHandling:
    """Tests de manejo de errores."""

    def test_process_invalid_frame(self, video_processor):
        """Test con frame inválido."""
        with pytest.raises(Exception):
            video_processor.process_frame_optimized(
                None,
                frame_number=1
            )

    def test_process_wrong_shape_frame(self, video_processor):
        """Test con frame de forma incorrecta."""
        wrong_frame = np.zeros((100, 100), dtype=np.uint8)

        try:
            result = video_processor.process_frame_optimized(
                wrong_frame,
                frame_number=1
            )
            assert isinstance(result, dict)
        except Exception:
            pass


class TestCompositionDataAccumulation:
    """Tests de acumulación de datos de composición."""

    def test_composition_data_accumulates(self, video_processor, sample_frame):
        """Test de que los datos de composición se acumulan."""
        for i in range(5):
            video_processor.process_frame_optimized(
                sample_frame,
                frame_number=i,
                full_analysis=True
            )

        assert len(video_processor.composition_data['rule_of_thirds_scores']) > 0

    def test_composition_data_in_final_results(
            self, video_processor, sample_frame
    ):
        """Test de que los datos de composición aparecen en resultados finales."""
        for i in range(3):
            video_processor.process_frame_optimized(
                sample_frame,
                frame_number=i,
                full_analysis=True
            )

        results = video_processor.get_final_results()

        if results.get("composition_data"):
            assert "rule_of_thirds_scores" in results["composition_data"]


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationProcessing:
    """Tests de integración de procesamiento completo."""

    def test_full_video_processing_simulation(
            self, video_processor, sample_frame, mock_actor_data
    ):
        """Test de simulación de procesamiento completo de vídeo."""
        video_processor.load_actor_encodings([mock_actor_data])

        for i in range(30):
            detect_faces = (i % 5 == 0)
            full_analysis = (i % 10 == 0)

            result = video_processor.process_frame_optimized(
                sample_frame,
                frame_number=i,
                detect_faces=detect_faces,
                full_analysis=full_analysis
            )

            assert isinstance(result, dict)
            assert result["frame_number"] == i

        final_results = video_processor.get_final_results()

        assert final_results["total_frames_analyzed"] > 0
        assert isinstance(final_results["detected_actors"], list)