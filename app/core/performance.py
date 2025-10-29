"""
performance.py

Sistema de monitoreo de rendimiento y métricas en tiempo real para procesamiento
de vídeo. Implementa tracking de FPS, uso de memoria, tiempos por frame y
estadísticas agregadas por sesión.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - psutil: Monitoreo de CPU y memoria del sistema
    - dataclasses: Estructuras de datos para métricas

Usage:
    from app.core.performance import performance_monitor, FrameTimer

    performance_monitor.start_session("video_123", total_frames=1000)

    with FrameTimer("Face Detection") as timer:
        faces = detect_faces(frame)

    performance_monitor.record_frame(
        session_id="video_123",
        frame_number=42,
        processing_time=timer.get_elapsed()
    )

    stats = performance_monitor.get_current_stats()
    print(f"FPS: {stats['current_fps']}")
"""

import time
import psutil
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """
    Métricas de procesamiento para un frame individual.

    Attributes:
        frame_number: Índice del frame en la secuencia de vídeo
        processing_time: Tiempo total de procesamiento en segundos
        face_detection_time: Tiempo específico de detección facial
        recognition_time: Tiempo de reconocimiento de identidades
        analysis_time: Tiempo de análisis cinematográfico
        memory_usage_mb: Uso de memoria RAM en megabytes
        timestamp: Momento de captura de las métricas
    """
    frame_number: int
    processing_time: float
    face_detection_time: Optional[float] = None
    recognition_time: Optional[float] = None
    analysis_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SessionMetrics:
    """
    Métricas agregadas para sesión completa de procesamiento.

    Attributes:
        session_id: Identificador único de la sesión
        start_time: Timestamp de inicio de procesamiento
        end_time: Timestamp de finalización, None si activa
        total_frames: Número total de frames a procesar
        frames_processed: Contador de frames procesados hasta el momento
        total_processing_time: Suma acumulada de tiempos de procesamiento
        average_fps: FPS promedio calculado como frames/tiempo_total
        peak_memory_mb: Pico máximo de uso de memoria durante la sesión
        errors_count: Contador de errores durante procesamiento
        frame_metrics: Histórico de métricas por frame
    """
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_frames: int = 0
    frames_processed: int = 0
    total_processing_time: float = 0.0
    average_fps: float = 0.0
    peak_memory_mb: float = 0.0
    errors_count: int = 0
    frame_metrics: List[FrameMetrics] = field(default_factory=list)


class PerformanceMonitor:
    """
    Monitor de rendimiento del sistema con tracking de sesiones.

    Implementa sistema de monitoreo en tiempo real que rastrea métricas de
    procesamiento por frame y por sesión completa. Utiliza psutil para métricas
    del sistema (CPU, memoria) y mantiene histórico de tiempos de frame en
    deque con tamaño limitado para eficiencia de memoria.

    Attributes:
        max_history: Tamaño máximo del buffer de histórico de tiempos
        frame_times: Buffer circular con tiempos recientes de frames
        sessions: Diccionario de todas las sesiones
        current_session: ID de la sesión activa actualmente
        process: Handle del proceso actual para monitoreo
    """

    def __init__(self, max_history: int = 100):
        """
        Inicializa el monitor de rendimiento.

        Args:
            max_history: Tamaño del buffer circular para histórico de tiempos
                de frames. Limita uso de memoria manteniendo solo datos recientes.
                Por defecto 100 frames

        Notes:
            Utiliza deque con maxlen para implementación eficiente de buffer
            circular O(1) para append y popleft. El handle de psutil.Process
            se crea una vez para evitar overhead de creación repetida.
        """
        self.max_history = max_history
        self.frame_times = deque(maxlen=max_history)
        self.sessions: Dict[str, SessionMetrics] = {}
        self.current_session: Optional[str] = None
        self.process = psutil.Process()

    def start_session(self, session_id: str, total_frames: int = 0) -> SessionMetrics:
        """
        Inicia nueva sesión de monitoreo de procesamiento.

        Args:
            session_id: Identificador único para la sesión
            total_frames: Número total de frames que se procesarán. 0 si es desconocido

        Returns:
            Objeto SessionMetrics recién creado para la sesión

        Notes:
            Establece la nueva sesión como current_session para registro automático.
            Múltiples sesiones pueden coexistir pero solo una es actual a la vez.
        """
        session = SessionMetrics(
            session_id=session_id,
            start_time=datetime.now(),
            total_frames=total_frames
        )
        self.sessions[session_id] = session
        self.current_session = session_id

        logger.info(f"Sesión iniciada: {session_id} ({total_frames} frames)")
        return session

    def record_frame(
            self,
            session_id: str,
            frame_number: int,
            processing_time: float,
            **kwargs
    ):
        """
        Registra métricas de procesamiento de un frame individual.

        Args:
            session_id: ID de la sesión a la que pertenece el frame
            frame_number: Índice del frame en la secuencia
            processing_time: Tiempo total de procesamiento en segundos
            **kwargs: Métricas adicionales opcionales (face_detection_time,
                recognition_time, analysis_time)

        Notes:
            Actualiza automáticamente:
                - Contador de frames procesados
                - Tiempo total acumulado
                - Uso de memoria actual y pico máximo
                - FPS promedio recalculado
                - Buffer circular de tiempos recientes

            Si la sesión no existe, registra warning pero no falla para
            prevenir interrupciones del pipeline de procesamiento.
        """
        if session_id not in self.sessions:
            logger.warning(f"Sesión {session_id} no encontrada")
            return

        session = self.sessions[session_id]

        memory_mb = self.process.memory_info().rss / (1024 * 1024)

        metrics = FrameMetrics(
            frame_number=frame_number,
            processing_time=processing_time,
            memory_usage_mb=memory_mb,
            **kwargs
        )

        session.frames_processed += 1
        session.total_processing_time += processing_time
        session.frame_metrics.append(metrics)

        if memory_mb > session.peak_memory_mb:
            session.peak_memory_mb = memory_mb

        if session.total_processing_time > 0:
            session.average_fps = session.frames_processed / session.total_processing_time

        self.frame_times.append(processing_time)

    def end_session(self, session_id: str) -> Optional[SessionMetrics]:
        """
        Finaliza sesión de monitoreo y genera resumen de métricas.

        Args:
            session_id: ID de la sesión a finalizar

        Returns:
            SessionMetrics con datos completos de la sesión, o None si no existe

        Notes:
            Calcula duración total y registra resumen con duración, frames procesados,
            FPS promedio, tiempo total y pico de memoria.

            La sesión permanece en el diccionario de sesiones para consulta
            posterior hasta que se ejecute cleanup_old_sessions().
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        session.end_time = datetime.now()

        duration = (session.end_time - session.start_time).total_seconds()

        logger.info("=" * 70)
        logger.info(f"Sesión finalizada: {session_id}")
        logger.info(f"   Duración: {duration:.2f}s")
        logger.info(f"   Frames procesados: {session.frames_processed}/{session.total_frames}")
        logger.info(f"   FPS promedio: {session.average_fps:.2f}")
        logger.info(f"   Tiempo total: {session.total_processing_time:.2f}s")
        logger.info(f"   Memoria pico: {session.peak_memory_mb:.1f} MB")
        logger.info("=" * 70)

        return session

    def get_current_stats(self) -> Dict:
        """
        Obtiene snapshot de estadísticas actuales del sistema.

        Returns:
            Diccionario con métricas instantáneas:
                cpu_percent: Uso de CPU del sistema
                memory_total_gb: RAM total del sistema
                memory_available_gb: RAM disponible
                memory_percent: Porcentaje de RAM usado
                process_memory_mb: Memoria del proceso actual
                active_sessions: Número de sesiones activas
                avg_frame_time: Tiempo promedio de frames recientes
                min_frame_time: Tiempo mínimo observado
                max_frame_time: Tiempo máximo observado
                current_fps: FPS estimado de frames recientes

        Notes:
            Utiliza interval=0.1 en cpu_percent para balance entre precisión
            y overhead. Las estadísticas de frames se calculan solo sobre el
            buffer circular reciente.
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info().rss / (1024 * 1024)

        stats = {
            "cpu_percent": round(cpu_percent, 1),
            "memory_total_gb": round(memory.total / (1024 ** 3), 2),
            "memory_available_gb": round(memory.available / (1024 ** 3), 2),
            "memory_percent": round(memory.percent, 1),
            "process_memory_mb": round(process_memory, 1),
            "active_sessions": len([s for s in self.sessions.values() if s.end_time is None])
        }

        if self.frame_times:
            stats.update({
                "avg_frame_time": round(sum(self.frame_times) / len(self.frame_times), 3),
                "min_frame_time": round(min(self.frame_times), 3),
                "max_frame_time": round(max(self.frame_times), 3),
                "current_fps": round(len(self.frame_times) / sum(self.frame_times), 2) if sum(
                    self.frame_times) > 0 else 0
            })

        return stats

    def get_stats(self) -> Dict:
        """
        Obtiene estadísticas completas del monitor incluyendo sesiones.

        Returns:
            Diccionario extendido con estadísticas del sistema más información
            detallada de sesiones activas y completadas

        Notes:
            Combina resultados de get_current_stats() con agregados de sesiones
            y detalles de la sesión actual si existe. Útil para dashboards y
            APIs de monitoreo.
        """
        stats = self.get_current_stats()

        stats["sessions"] = {
            "total": len(self.sessions),
            "active": len([s for s in self.sessions.values() if s.end_time is None]),
            "completed": len([s for s in self.sessions.values() if s.end_time is not None])
        }

        if self.current_session and self.current_session in self.sessions:
            current = self.sessions[self.current_session]
            stats["current_session"] = {
                "session_id": current.session_id,
                "frames_processed": current.frames_processed,
                "total_frames": current.total_frames,
                "average_fps": round(current.average_fps, 2),
                "peak_memory_mb": round(current.peak_memory_mb, 1)
            }

        return stats

    def get_session_report(self, session_id: str) -> Optional[Dict]:
        """
        Genera reporte detallado de métricas de una sesión específica.

        Args:
            session_id: ID de la sesión para generar reporte

        Returns:
            Diccionario con reporte completo o None si sesión no existe.
            Incluye estadísticas agregadas, tasas de completitud y análisis
            de tiempos de procesamiento

        Notes:
            Calcula estadísticas descriptivas (promedio, mínimo, máximo) de
            tiempos de procesamiento sobre todos los frames de la sesión.
            Útil para análisis post-procesamiento y optimización de rendimiento.
        """
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        processing_times = [m.processing_time for m in session.frame_metrics]

        report = {
            "session_id": session.session_id,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "total_frames": session.total_frames,
            "frames_processed": session.frames_processed,
            "completion_rate": round(
                (session.frames_processed / session.total_frames * 100) if session.total_frames > 0 else 0,
                2
            ),
            "total_time": round(session.total_processing_time, 2),
            "average_fps": round(session.average_fps, 2),
            "peak_memory_mb": round(session.peak_memory_mb, 1),
            "errors": session.errors_count
        }

        if processing_times:
            report.update({
                "avg_processing_time": round(sum(processing_times) / len(processing_times), 3),
                "min_processing_time": round(min(processing_times), 3),
                "max_processing_time": round(max(processing_times), 3)
            })

        return report

    def record_error(self, session_id: str):
        """
        Registra error durante procesamiento de sesión.

        Args:
            session_id: ID de la sesión donde ocurrió el error

        Notes:
            Incrementa contador de errores para análisis de tasa de fallos.
            No interrumpe procesamiento, solo registra para métricas.
        """
        if session_id in self.sessions:
            self.sessions[session_id].errors_count += 1

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """
        Limpia sesiones finalizadas que exceden edad máxima.

        Args:
            max_age_hours: Edad máxima en horas para retener sesiones.
                Por defecto 24 horas

        Notes:
            Solo elimina sesiones finalizadas (end_time != None). Las sesiones
            activas nunca se eliminan independiente de su edad. Útil para
            prevenir crecimiento ilimitado del diccionario de sesiones en
            sistemas de larga duración.
        """
        now = datetime.now()
        to_remove = []

        for session_id, session in self.sessions.items():
            if session.end_time:
                age = (now - session.end_time).total_seconds() / 3600
                if age > max_age_hours:
                    to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"Sesión antigua eliminada: {session_id}")


class FrameTimer:
    """
    Context manager para medición de tiempo de operaciones.

    Registra automáticamente warnings para operaciones lentas (>0.5s).
    Útil para identificar cuellos de botella en pipeline de procesamiento.

    Attributes:
        name: Etiqueta descriptiva para la operación
        start_time: Timestamp de inicio de medición
        elapsed: Tiempo transcurrido en segundos
    """

    def __init__(self, name: str = "Operation"):
        """
        Inicializa timer con nombre descriptivo.

        Args:
            name: Etiqueta descriptiva para la operación a medir
        """
        self.name = name
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        """Inicia medición de tiempo."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Finaliza medición y registra si es operación lenta.

        Notes:
            Umbrales de logging:
                - >1.0s: Warning por operación muy lenta
                - >0.5s: Info por operación moderadamente lenta
        """
        self.elapsed = time.time() - self.start_time

        if self.elapsed > 1.0:
            logger.warning(f"{self.name} tomó {self.elapsed:.2f}s (lento)")
        elif self.elapsed > 0.5:
            logger.info(f"{self.name}: {self.elapsed:.2f}s")

    def get_elapsed(self) -> float:
        """
        Obtiene tiempo transcurrido.

        Returns:
            Tiempo en segundos. Si el timer está activo, retorna tiempo
            actual. Si finalizó, retorna tiempo total medido
        """
        if self.elapsed is None:
            return time.time() - self.start_time if self.start_time else 0
        return self.elapsed


performance_monitor = PerformanceMonitor()