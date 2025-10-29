"""
Middleware Module - Middleware de CVFlix

Capa de middleware para aplicación FastAPI que proporciona manejo centralizado
de errores, logging de requests y gestión de excepciones HTTP.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Modules:
    error_handler: Sistema centralizado de captura y manejo de excepciones
        con logging estructurado y respuestas HTTP estandarizadas

Usage:
    from fastapi import FastAPI
    from app.middleware import setup_middleware

    app = FastAPI()
    setup_middleware(app)

Notes:
    El middleware se carga condicionalmente para manejo robusto de dependencias.
    Si faltan componentes, la aplicación continúa funcionando sin middleware.
"""

from typing import Optional
from fastapi import FastAPI
import logging

__version__ = "4.0.0"
__author__ = "César Sánchez Montes"

logger = logging.getLogger(__name__)

try:
    from .error_handler import (
        ErrorHandlerMiddleware,
        validation_exception_handler,
        http_exception_handler,
        general_exception_handler,
        setup_error_handlers
    )

    ERROR_HANDLER_AVAILABLE = True
except ImportError as e:
    ERROR_HANDLER_AVAILABLE = False
    logger.warning(f"Error handler not available: {e}")

__all__ = [
    "ErrorHandlerMiddleware",
    "validation_exception_handler",
    "http_exception_handler",
    "general_exception_handler",
    "setup_error_handlers",
    "setup_middleware",
    "ERROR_HANDLER_AVAILABLE",
]


def setup_middleware(app: FastAPI, enable_error_handlers: bool = True):
    """
    Configura middleware completo para la aplicación FastAPI.

    Registra middleware de manejo de errores HTTP y configura exception handlers
    personalizados para validación, errores HTTP y excepciones generales.

    Args:
        app: Instancia de FastAPI donde registrar middleware
        enable_error_handlers: Flag para activar/desactivar manejadores de error.
            Por defecto True. Útil para deshabilitar en testing

    Notes:
        El middleware se registra en el siguiente orden:
            1. ErrorHandlerMiddleware - Captura excepciones no manejadas
            2. Exception handlers específicos - Validación, HTTP, generales

        Si ERROR_HANDLER_AVAILABLE es False, la configuración se omite sin
        generar error, permitiendo arranque degradado de la aplicación.

        Los errores durante configuración se registran pero no interrumpen
        el startup de la aplicación para prevenir fallos catastróficos.
    """
    if enable_error_handlers and ERROR_HANDLER_AVAILABLE:
        try:
            app.middleware("http")(ErrorHandlerMiddleware(app))
            setup_error_handlers(app)
            logger.info("Middleware de errores configurado")
        except Exception as e:
            logger.error(f"Error configurando middleware: {e}")
    else:
        logger.warning("Middleware de errores no disponible")


def get_middleware_info() -> dict:
    """
    Obtiene información sobre disponibilidad de middleware.

    Returns:
        Diccionario con estado de componentes de middleware:
            version: Versión del módulo middleware
            error_handler_available: Disponibilidad del sistema de
                manejo de errores

    Notes:
        Útil para diagnóstico de dependencias y verificación de configuración
        en endpoints de health check o admin panels.
    """
    return {
        "version": __version__,
        "error_handler_available": ERROR_HANDLER_AVAILABLE,
    }