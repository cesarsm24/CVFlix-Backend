"""
cache.py

Sistema de caché inteligente con política LRU para optimización de procesamiento
de vídeo. Implementa almacenamiento persistente con serialización pickle, gestión
automática de expiración TTL y limpieza por tamaño con estrategia LRU.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - pickle: Serialización de objetos Python
    - json: Almacenamiento de metadatos
    - hashlib: Generación de claves hash

Usage:
    from app.core.cache import CacheManager, cached

    cache = CacheManager(cache_dir=Path("cache"), max_size_mb=500)

    result = cache.get("key")
    if result is None:
        result = compute_expensive_operation()
        cache.set("key", result, ttl_hours=24)

    @cached(cache, key_prefix="encoding", ttl_hours=48)
    def expensive_function(data):
        return processed_data
"""

import hashlib
import pickle
import logging
from typing import Any, Optional, Dict, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gestor de caché persistente con política LRU y expiración TTL.

    Implementa sistema de almacenamiento en disco para resultados de procesamiento
    computacionalmente costosos. Utiliza serialización pickle para objetos Python
    arbitrarios, gestión automática de tamaño mediante política LRU (Least Recently
    Used), y expiración basada en tiempo de vida (TTL).

    Attributes:
        cache_dir: Directorio raíz donde se almacenan archivos de caché
        max_size_mb: Límite máximo de tamaño del caché en megabytes
        metadata_file: Archivo JSON con metadatos de todas las entradas
        metadata: Diccionario en memoria con información de cada entrada
            cacheada incluyendo timestamps, hits, tamaño y metadatos personalizados
    """

    def __init__(self, cache_dir: Path, max_size_mb: int = 500):
        """
        Inicializa el gestor de caché.

        Args:
            cache_dir: Directorio donde almacenar archivos de caché. Se crea
                automáticamente si no existe
            max_size_mb: Tamaño máximo del caché en megabytes. Al superar este
                límite se activa limpieza automática LRU. Por defecto 500MB

        Notes:
            El directorio de caché se crea con parents=True para generar toda
            la jerarquía necesaria. Los metadatos se cargan automáticamente
            desde cache_metadata.json si existe.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """
        Carga metadatos persistentes del caché desde disco.

        Returns:
            Diccionario con metadatos de todas las entradas, o diccionario vacío
            si el archivo no existe o está corrupto

        Notes:
            Los errores de carga se registran pero no interrumpen la inicialización,
            permitiendo arranque limpio del sistema incluso con metadatos corruptos.
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error cargando metadata: {e}")
        return {}

    def _save_metadata(self):
        """
        Persiste metadatos del caché a disco en formato JSON.

        Notes:
            Utiliza indentación para legibilidad del archivo JSON. Los errores
            se registran pero no se propagan para evitar interrumpir operaciones
            de caché por fallos de persistencia de metadatos.
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error guardando metadata: {e}")

    def _generate_key(self, *args, **kwargs) -> str:
        """
        Genera clave hash única para conjunto de parámetros.

        Args:
            *args: Argumentos posicionales a incluir en la clave
            **kwargs: Argumentos nombrados a incluir en la clave

        Returns:
            Hash MD5 hexadecimal de 32 caracteres representando los parámetros

        Notes:
            Los kwargs se ordenan alfabéticamente para garantizar consistencia
            independiente del orden de paso de parámetros. MD5 es suficiente
            para prevenir colisiones en espacio de caché típico.
        """
        key_str = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """
        Calcula ruta del archivo de caché para una clave dada.

        Args:
            key: Identificador único de la entrada de caché

        Returns:
            Path completo al archivo .cache correspondiente
        """
        return self.cache_dir / f"{key}.cache"

    def get(self, key: str) -> Optional[Any]:
        """
        Recupera valor del caché si existe y no ha expirado.

        Args:
            key: Identificador único de la entrada a recuperar

        Returns:
            Objeto deserializado si existe en caché y es válido, None si no existe,
            expiró o hubo error de deserialización

        Notes:
            Actualiza automáticamente estadísticas de hits y timestamp de último
            acceso. Las entradas expiradas se eliminan automáticamente. Los errores
            de deserialización resultan en eliminación de la entrada corrupta.
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        if key in self.metadata:
            expires_at = datetime.fromisoformat(self.metadata[key]['expires_at'])
            if datetime.now() > expires_at:
                logger.info(f"Caché expirado: {key}")
                self.delete(key)
                return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            if key in self.metadata:
                self.metadata[key]['hits'] += 1
                self.metadata[key]['last_accessed'] = datetime.now().isoformat()
                self._save_metadata()

            logger.debug(f"Caché hit: {key}")
            return data

        except Exception as e:
            logger.error(f"Error leyendo caché {key}: {e}")
            self.delete(key)
            return None

    def set(
            self,
            key: str,
            value: Any,
            ttl_hours: int = 24,
            metadata: Optional[Dict] = None
    ):
        """
        Almacena valor en caché con tiempo de vida especificado.

        Args:
            key: Identificador único para la entrada
            value: Objeto Python a cachear (debe ser serializable con pickle)
            ttl_hours: Tiempo de vida en horas antes de expiración. Por defecto 24h
            metadata: Diccionario opcional con metadatos personalizados para la entrada

        Notes:
            Serializa el valor con pickle y almacena metadatos en JSON separado.
            Calcula tamaño del archivo para gestión de límites. Activa limpieza
            automática si se excede el tamaño máximo del caché.
        """
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)

            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            expires_at = datetime.now() + timedelta(hours=ttl_hours)

            self.metadata[key] = {
                'created_at': datetime.now().isoformat(),
                'expires_at': expires_at.isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'size_mb': file_size_mb,
                'hits': 0,
                'metadata': metadata or {}
            }

            self._save_metadata()
            logger.debug(f"Caché guardado: {key} ({file_size_mb:.2f} MB)")

            self._cleanup_if_needed()

        except Exception as e:
            logger.error(f"Error guardando caché {key}: {e}")

    def delete(self, key: str):
        """
        Elimina entrada del caché incluyendo archivo y metadatos.

        Args:
            key: Identificador de la entrada a eliminar

        Notes:
            Maneja gracefully el caso donde el archivo o metadatos no existen.
            Los errores se registran pero no se propagan.
        """
        cache_path = self._get_cache_path(key)

        try:
            if cache_path.exists():
                cache_path.unlink()

            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()

            logger.debug(f"Caché eliminado: {key}")

        except Exception as e:
            logger.error(f"Error eliminando caché {key}: {e}")

    def clear_all(self):
        """
        Elimina todas las entradas del caché.

        Notes:
            Itera sobre todos los archivos .cache en el directorio y limpia
            completamente el diccionario de metadatos. Útil para mantenimiento
            o reset del sistema.
        """
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Error eliminando {cache_file}: {e}")

        self.metadata.clear()
        self._save_metadata()
        logger.info("Caché completamente limpiado")

    def _cleanup_if_needed(self):
        """
        Ejecuta limpieza LRU si el caché excede el tamaño máximo.

        Notes:
            Política de limpieza:
                - Ordena entradas por timestamp de último acceso (LRU)
                - Elimina entradas más antiguas hasta reducir a 80% del límite
                - Deja 20% de margen para evitar limpiezas frecuentes

            La limpieza es automática y transparente, ejecutándose después de
            cada operación set() que cause exceso de tamaño.
        """
        total_size = sum(m['size_mb'] for m in self.metadata.values())

        if total_size > self.max_size_mb:
            logger.warning(f"Caché excede límite ({total_size:.1f}/{self.max_size_mb} MB)")

            sorted_keys = sorted(
                self.metadata.keys(),
                key=lambda k: self.metadata[k]['last_accessed']
            )

            for key in sorted_keys:
                if total_size <= self.max_size_mb * 0.8:
                    break

                size = self.metadata[key]['size_mb']
                self.delete(key)
                total_size -= size
                logger.info(f"Eliminado por límite: {key} ({size:.2f} MB)")

    def get_total_size_mb(self) -> float:
        """
        Calcula tamaño total actual del caché en megabytes.

        Returns:
            Suma de tamaños de todas las entradas en MB, o 0.0 si está vacío
        """
        if not self.metadata:
            return 0.0
        return sum(m['size_mb'] for m in self.metadata.values())

    def get_stats(self) -> Dict:
        """
        Genera estadísticas agregadas del estado del caché.

        Returns:
            Diccionario con métricas del caché:
                total_entries: Número de entradas cacheadas
                total_size_mb: Tamaño total en megabytes
                max_size_mb: Límite máximo configurado
                usage_percent: Porcentaje de uso del límite
                total_hits: Suma de hits de todas las entradas
                entries: Lista de claves de entradas existentes

        Notes:
            Útil para monitoreo, debugging y dashboards de métricas del sistema.
        """
        if not self.metadata:
            return {
                "total_entries": 0,
                "total_size_mb": 0,
                "hit_rate": 0
            }

        total_size = self.get_total_size_mb()
        total_hits = sum(m['hits'] for m in self.metadata.values())

        return {
            "total_entries": len(self.metadata),
            "total_size_mb": round(total_size, 2),
            "max_size_mb": self.max_size_mb,
            "usage_percent": round((total_size / self.max_size_mb) * 100, 1),
            "total_hits": total_hits,
            "entries": list(self.metadata.keys())
        }


def cached(
        cache_manager: CacheManager,
        key_prefix: str = "",
        ttl_hours: int = 24
):
    """
    Decorador para cachear automáticamente resultados de funciones.

    Args:
        cache_manager: Instancia de CacheManager a utilizar
        key_prefix: Prefijo opcional para distinguir contextos de caché
        ttl_hours: Tiempo de vida en horas para entradas generadas

    Returns:
        Decorador que envuelve funciones para agregar comportamiento de caché

    Notes:
        La clave de caché se genera combinando:
            - Prefijo especificado
            - Nombre de la función
            - Representación string de todos los argumentos

        Argumentos nombrados se ordenan alfabéticamente para garantizar
        consistencia de claves. El resultado se hashea con MD5.

        Útil para funciones puras con operaciones costosas (análisis de imagen,
        encoding facial, procesamiento de IA) donde resultados son deterministas
        dado mismo input.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key_parts = [key_prefix, func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])

            cache_key = hashlib.md5("_".join(key_parts).encode()).hexdigest()

            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            result = func(*args, **kwargs)

            cache_manager.set(
                cache_key,
                result,
                ttl_hours=ttl_hours,
                metadata={
                    "function": func.__name__,
                    "prefix": key_prefix
                }
            )

            return result

        return wrapper

    return decorator