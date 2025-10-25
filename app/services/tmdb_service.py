"""
tmdb_service.py

Servicio de integración con API de The Movie Database (TMDB) para búsqueda
de contenido cinematográfico, obtención de reparto y descarga de imágenes
de actores para reconocimiento facial.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0
"""

import requests
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import io
from ..config import TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE


class TMDBService:
    """
    Cliente HTTP para consumir API REST de The Movie Database.

    Implementa métodos para búsqueda de contenido (películas y series),
    obtención de reparto, descarga de metadatos y recuperación de imágenes
    de actores en formato numpy array para procesamiento con OpenCV.

    Attributes:
        api_key (str): Clave API de TMDB para autenticación.
        base_url (str): URL base de la API v3 de TMDB.
        image_base (str): URL base del CDN de imágenes de TMDB.
    """

    def __init__(self):
        """
        Inicializa servicio con credenciales desde configuración.

        Notes:
            Las credenciales se cargan desde módulo config. Es necesario
            tener una API key válida de TMDB para que el servicio funcione.
        """
        self.api_key = TMDB_API_KEY
        self.base_url = TMDB_BASE_URL
        self.image_base = TMDB_IMAGE_BASE

    def search_content(self, query: str, content_type: str = "movie") -> Tuple[Optional[int], Optional[str]]:
        """
        Busca contenido en TMDB y retorna el mejor resultado por puntuación.

        Realiza búsqueda de películas y/o series según parámetro content_type,
        filtra resultados por mínimo de votos para evitar contenido oscuro,
        y selecciona el mejor puntuado considerando vote_average y popularity.

        Args:
            query: Término de búsqueda (título de película o serie).
            content_type: Tipo de contenido a buscar. Valores válidos:
                - "movie": solo películas
                - "tv": solo series
                - "auto": ambos tipos (por defecto)

        Returns:
            Tupla (content_id, detected_type) donde:
                - content_id: ID de TMDB del contenido, None si no se encuentra
                - detected_type: "movie" o "tv", None si no se encuentra

        Notes:
            Algoritmo de selección:
                1. Busca películas y/o series según content_type
                2. Filtra resultados con menos de 100 votos (evita contenido oscuro)
                3. Ordena por vote_average descendente, luego por popularity
                4. Retorna el primer resultado (mejor puntuado)

            El umbral de 100 votos previene que películas/series con un solo
            voto de 10/10 sean consideradas mejor puntuadas que contenido
            popular con 8.5/10 promedio de miles de votos.

            En caso de error durante la búsqueda, registra el traceback completo
            y retorna (None, None) sin interrumpir el flujo de la aplicación.
        """
        try:
            movies = []
            tv_shows = []

            if content_type in ["movie", "auto"]:
                movies = self._search_movies(query)

            if content_type in ["tv", "auto"]:
                tv_shows = self._search_tv_shows(query)

            if not movies and not tv_shows:
                return None, None

            all_results = []

            for movie in movies:
                vote_average = movie.get("vote_average", 0)
                vote_count = movie.get("vote_count", 0)

                if vote_count >= 100:
                    all_results.append({
                        "id": movie["id"],
                        "type": "movie",
                        "title": movie.get("title", "Sin título"),
                        "year": movie.get("release_date", "")[:4] if movie.get("release_date") else "N/A",
                        "vote_average": vote_average,
                        "vote_count": vote_count,
                        "popularity": movie.get("popularity", 0)
                    })

            for tv in tv_shows:
                vote_average = tv.get("vote_average", 0)
                vote_count = tv.get("vote_count", 0)

                if vote_count >= 100:
                    all_results.append({
                        "id": tv["id"],
                        "type": "tv",
                        "title": tv.get("name", "Sin título"),
                        "year": tv.get("first_air_date", "")[:4] if tv.get("first_air_date") else "N/A",
                        "vote_average": vote_average,
                        "vote_count": vote_count,
                        "popularity": tv.get("popularity", 0)
                    })

            if not all_results:
                return None, None

            all_results.sort(key=lambda x: (x["vote_average"], x["popularity"]), reverse=True)

            best = all_results[0]
            return best["id"], best["type"]

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None

    def _search_movies(self, query: str) -> List[Dict]:
        """
        Realiza búsqueda de películas en API de TMDB.

        Args:
            query: Término de búsqueda para películas.

        Returns:
            Lista de diccionarios con datos de películas encontradas.
            Lista vacía si no hay resultados o hay error.

        Notes:
            Utiliza endpoint /search/movie con idioma español (es-ES) para
            obtener títulos y sinopsis traducidas. El timeout de 10 segundos
            previene bloqueos indefinidos en caso de problemas de red.
        """
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                "api_key": self.api_key,
                "query": query,
                "language": "es-ES"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            return []

    def _search_tv_shows(self, query: str) -> List[Dict]:
        """
        Realiza búsqueda de series de TV en API de TMDB.

        Args:
            query: Término de búsqueda para series.

        Returns:
            Lista de diccionarios con datos de series encontradas.
            Lista vacía si no hay resultados o hay error.

        Notes:
            Utiliza endpoint /search/tv con los mismos parámetros que la
            búsqueda de películas para consistencia de idioma y formato.
        """
        try:
            url = f"{self.base_url}/search/tv"
            params = {
                "api_key": self.api_key,
                "query": query,
                "language": "es-ES"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            return []

    def get_cast(self, content_id: int, content_type: str, limit: int = 15) -> List[Dict]:
        """
        Obtiene lista de reparto principal del contenido.

        Args:
            content_id: ID de TMDB del contenido (película o serie).
            content_type: Tipo de contenido ("movie" o "tv").
            limit: Número máximo de actores a retornar. Por defecto 15.

        Returns:
            Lista de diccionarios con información de actores ordenados por
            importancia (billing order). Lista vacía si hay error.

        Notes:
            La API de TMDB retorna el reparto ordenado por billing order
            (orden de créditos), donde las primeras posiciones corresponden
            a actores principales. El límite de 15 es balance entre cobertura
            de personajes principales y rendimiento de procesamiento.
        """
        try:
            url = f"{self.base_url}/{content_type}/{content_id}/credits"
            params = {"api_key": self.api_key}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            cast = data.get("cast", [])[:limit]
            return cast
        except Exception as e:
            return []

    def get_content_details(self, content_id: int, content_type: str) -> Optional[Dict]:
        """
        Obtiene metadatos completos del contenido.

        Args:
            content_id: ID de TMDB del contenido.
            content_type: Tipo de contenido ("movie" o "tv").

        Returns:
            Diccionario con metadatos completos del contenido incluyendo título,
            sinopsis, fecha de estreno, géneros, runtime, etc. None si hay error.

        Notes:
            Los metadatos se obtienen en idioma español (es-ES) para consistencia
            con la interfaz de usuario. Incluye campos como overview, poster_path,
            genres, runtime, vote_average, entre otros.
        """
        try:
            url = f"{self.base_url}/{content_type}/{content_id}"
            params = {"api_key": self.api_key, "language": "es-ES"}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return None

    def get_content_data(self, content_id: int, content_type: str) -> Optional[Dict]:
        """
        Alias de get_content_details para compatibilidad con código legacy.

        Args:
            content_id: ID de TMDB del contenido.
            content_type: Tipo de contenido ("movie" o "tv").

        Returns:
            Mismo resultado que get_content_details().

        Notes:
            Método mantenido para retrocompatibilidad con versiones anteriores
            del código que utilizaban este nombre de método.
        """
        return self.get_content_details(content_id, content_type)

    def get_poster_url(self, content_id: int, content_type: str) -> Optional[str]:
        """
        Construye URL completa del poster en alta resolución.

        Args:
            content_id: ID de TMDB del contenido.
            content_type: Tipo de contenido ("movie" o "tv").

        Returns:
            URL completa del poster en resolución w500 (500px ancho).
            None si el contenido no tiene poster o hay error.

        Notes:
            TMDB ofrece posters en múltiples resoluciones. Se utiliza w500
            como balance entre calidad visual y tamaño de descarga. Otras
            opciones disponibles: w92, w154, w185, w342, w500, w780, original.
        """
        try:
            details = self.get_content_details(content_id, content_type)
            if details and details.get("poster_path"):
                return f"{self.image_base}{details['poster_path']}"
            return None
        except Exception as e:
            return None

    def load_actor_image(self, profile_path: str) -> Optional[np.ndarray]:
        """
        Descarga y convierte imagen de actor a numpy array RGB.

        Args:
            profile_path: Ruta relativa de imagen en CDN de TMDB (e.g., "/abc.jpg").

        Returns:
            Array numpy en formato RGB (H, W, 3) listo para procesamiento con
            OpenCV/face_recognition. None si no hay imagen o falla descarga.

        Notes:
            Pipeline de procesamiento:
                1. Construye URL completa concatenando image_base + profile_path
                2. Descarga imagen con timeout de 10 segundos
                3. Abre con PIL.Image desde bytes en memoria
                4. Convierte a RGB si está en otro modo (RGBA, L, etc.)
                5. Convierte a numpy array para compatibilidad con OpenCV

            La conversión a RGB es crítica porque face_recognition requiere
            imágenes en formato RGB de 3 canales. Imágenes RGBA (con canal
            alpha) o en escala de grises fallarían en el encoding facial.
        """
        if not profile_path:
            return None
        try:
            url = f"{self.image_base}{profile_path}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(io.BytesIO(response.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return np.array(img)
            return None
        except Exception as e:
            return None

    def load_cast_images(self, cast: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Descarga imágenes de múltiples actores en batch.

        Args:
            cast: Lista de diccionarios de actores con campo profile_path.

        Returns:
            Diccionario mapeando actor_id a numpy array RGB de su fotografía.
            Solo incluye actores cuyas imágenes se descargaron exitosamente.

        Notes:
            Procesa secuencialmente la lista de actores llamando a
            load_actor_image() para cada uno. Actores sin profile_path o
            cuya descarga falle se omiten del resultado sin interrumpir
            el procesamiento de los demás.

            Para optimización, podría paralelizarse con ThreadPoolExecutor
            pero la implementación actual prioriza simplicidad y evita
            problemas de rate limiting de la API de TMDB.
        """
        actor_images = {}
        for actor in cast:
            if not actor.get("profile_path"):
                continue
            img = self.load_actor_image(actor["profile_path"])
            if img is not None:
                actor_images[actor["id"]] = img
        return actor_images