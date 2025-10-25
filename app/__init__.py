"""
CVFlix - Sistema de Análisis Cinematográfico con IA

Plataforma de análisis automático de video mediante computer vision y deep learning.
Implementa detección facial, reconocimiento de actores, análisis de emociones,
clasificación de planos, evaluación de composición, iluminación y color.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Faculty: Escuela Politécnica de Cáceres (EPCC)
Universidad: Universidad de Extremadura (UEX)
Version: 4.0.0

Core Technologies:
    - FastAPI (API REST y Server-Sent Events)
    - OpenCV (procesamiento de video)
    - face_recognition (reconocimiento facial)
    - TensorFlow/Keras (detección de emociones)

Main Components:
    - VideoProcessor: orquestador principal de análisis
    - TMDBService: integración con The Movie Database
    - Analizadores cinematográficos (planos, composición, iluminación, color)
    - Sistema de caché LRU y monitoreo de rendimiento
"""

__version__ = "4.0.0"
__author__ = "César Sánchez Montes"
__course__ = "Imagen Digital"
__year__ = "2025"
__faculty__ = "Escuela Politécnica de Cáceres (EPCC)"
__university__ = "Universidad de Extremadura (UEX)"