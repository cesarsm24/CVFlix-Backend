"""
CVFlix - Sistema de Análisis Cinematográfico con IA

Plataforma de análisis automático de video mediante computer vision y deep learning.
Implementa detección facial, reconocimiento de actores, análisis de emociones,
clasificación de planos, evaluación de composición, iluminación y color.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Faculty: Escuela Politécnica de Cáceres
Universidad: Universidad de Extremadura
Version: 4.0.0

Core Technologies:
    - FastAPI: API REST y Server-Sent Events
    - OpenCV: Procesamiento de video
    - face_recognition: Reconocimiento facial
    - TensorFlow/Keras: Detección de emociones

Main Components:
    - VideoProcessor: Orquestador principal de análisis
    - TMDBService: Integración con The Movie Database
    - Analizadores cinematográficos: Planos, composición, iluminación, color
    - Sistema de caché LRU y monitoreo de rendimiento
"""

__version__ = "4.0.0"
__author__ = "César Sánchez Montes"
__course__ = "Imagen Digital"
__year__ = "2025"
__faculty__ = "Escuela Politécnica de Cáceres"
__university__ = "Universidad de Extremadura"