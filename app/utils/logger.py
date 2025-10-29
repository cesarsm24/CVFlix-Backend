"""
logger.py

Sistema de logging centralizado para CVFlix con utilidades para logging
estructurado de procesamiento de vídeo y eventos del sistema.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Usage:
    from app.utils.logger import CVFlixLogger

    logger = CVFlixLogger.get_logger(__name__)
    logger.info("Procesamiento iniciado")

    CVFlixLogger.log_processing_start(logger, "video.mp4", 15000, 500.0)
    CVFlixLogger.log_processing_end(logger, "video.mp4", 5000, 250.5)
"""

import logging
from typing import Optional
from pathlib import Path


class CVFlixLogger:
    """
    Logger centralizado con gestión de instancias y utilidades de logging.

    Implementa patrón singleton por nombre de logger para evitar duplicación
    de handlers y configuración. Proporciona métodos de conveniencia para
    logging estructurado de eventos de procesamiento de vídeo.

    Attributes:
        _loggers: Cache de loggers por nombre para reutilización de instancias

    Notes:
        Los loggers se crean bajo demanda y se almacenan en cache para
        prevenir creación redundante de handlers. La jerarquía de nombres
        sigue convención Python (módulos separados por punto).
    """

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Obtiene logger existente o crea uno nuevo con configuración estándar.

        Args:
            name: Nombre del logger, típicamente __name__ del módulo llamante

        Returns:
            Instancia de logging.Logger configurada

        Notes:
            Los loggers se cachean por nombre para evitar duplicación. El
            nombre debería seguir convención de módulos Python para aprovechar
            jerarquía de logging. El logger retornado hereda configuración de
            handlers y formatters del logger raíz de la aplicación.
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger

        return cls._loggers[name]

    @classmethod
    def log_processing_start(cls, logger: logging.Logger, filename: str,
                             total_frames: int, duration: float):
        """
        Registra inicio de procesamiento de vídeo con formato estructurado.

        Args:
            logger: Instancia de logger donde registrar el evento
            filename: Nombre del archivo de vídeo a procesar
            total_frames: Número total de frames en el vídeo
            duration: Duración del vídeo en segundos

        Notes:
            Formato con separadores para identificación visual rápida en logs.
            La información de frames y duración es crítica para estimar tiempo
            de procesamiento y configurar estrategias de sampling.

            Los números se formatean con separadores de miles para legibilidad.
            La duración se redondea a 2 decimales para precisión suficiente.
        """
        logger.info("=" * 70)
        logger.info(f"Iniciando procesamiento: {filename}")
        logger.info(f"   Total frames: {total_frames:,}")
        logger.info(f"   Duración: {duration:.2f}s")
        logger.info("=" * 70)

    @classmethod
    def log_processing_end(cls, logger: logging.Logger, filename: str,
                           frames_processed: int, elapsed_time: float):
        """
        Registra finalización de procesamiento con métricas de rendimiento.

        Args:
            logger: Instancia de logger donde registrar el evento
            filename: Nombre del archivo de vídeo procesado
            frames_processed: Número de frames efectivamente procesados
            elapsed_time: Tiempo transcurrido en segundos desde inicio

        Notes:
            Calcula automáticamente FPS de procesamiento (frames/segundo) como
            métrica clave de rendimiento. FPS típicos:
                - 1-5 fps: procesamiento pesado (análisis completo por frame)
                - 10-30 fps: procesamiento optimizado con sampling
                - >30 fps: procesamiento ligero o hardware acelerado

            El cálculo de FPS maneja división por cero retornando 0 si
            elapsed_time es 0. Los separadores visuales facilitan parsing
            de logs para extracción de métricas y análisis de rendimiento.
        """
        fps = frames_processed / elapsed_time if elapsed_time > 0 else 0
        logger.info("=" * 70)
        logger.info(f"Procesamiento completado: {filename}")
        logger.info(f"   Frames procesados: {frames_processed:,}")
        logger.info(f"   Tiempo total: {elapsed_time:.2f}s")
        logger.info(f"   Velocidad: {fps:.2f} fps")
        logger.info("=" * 70)