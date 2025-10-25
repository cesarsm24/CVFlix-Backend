"""
==============================================================================
Core Module - Módulos Centrales de CVFlix
==============================================================================

Paquete central de infraestructura para sistema de análisis cinematográfico.
Proporciona caché inteligente con política LRU y monitoreo de rendimiento.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Modules:
    cache: Sistema de caché inteligente con política LRU para optimización
        de análisis repetitivos.
    performance: Monitoreo de rendimiento y métricas de procesamiento.

==============================================================================
"""

from pathlib import Path

__version__ = "4.0.0"
__author__ = "César Sánchez Montes"

# Importaciones condicionales para manejo robusto de dependencias opcionales
try:
    from .cache import CacheManager, cached

    CACHE_AVAILABLE = True

    # Instancia singleton global: directorio ./cache con límite de 500MB
    cache_manager = CacheManager(
        cache_dir=Path("cache"),
        max_size_mb=500
    )
except ImportError as e:
    CACHE_AVAILABLE = False
    cache_manager = None

try:
    from .performance import (
        PerformanceMonitor,
        FrameTimer,
        performance_monitor
    )

    PERFORMANCE_AVAILABLE = True
except ImportError as e:
    PERFORMANCE_AVAILABLE = False
    performance_monitor = None

# API pública del módulo
__all__ = [
    "CacheManager",
    "cached",
    "cache_manager",
    "CACHE_AVAILABLE",
    "PerformanceMonitor",
    "FrameTimer",
    "performance_monitor",
    "PERFORMANCE_AVAILABLE",
]


def get_module_info() -> dict:
    """
    Obtiene información sobre disponibilidad de módulos core.

    Returns:
        Diccionario con estado de módulos:
            version (str): Versión del módulo.
            cache_available (bool): Disponibilidad del sistema de caché.
            performance_available (bool): Disponibilidad del monitor de rendimiento.
            cache_instance (bool): Estado de instancia cache_manager.
            performance_instance (bool): Estado de instancia performance_monitor.

    Notes:
        Útil para diagnóstico de dependencias y verificación de configuración.
    """
    return {
        "version": __version__,
        "cache_available": CACHE_AVAILABLE,
        "performance_available": PERFORMANCE_AVAILABLE,
        "cache_instance": cache_manager is not None,
        "performance_instance": performance_monitor is not None
    }