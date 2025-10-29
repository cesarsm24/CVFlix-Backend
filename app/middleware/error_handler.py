"""
error_handler.py

Middleware de manejo centralizado de errores HTTP con logging estructurado,
formateo consistente de respuestas y medición de tiempo de procesamiento.

Author: César Sánchez Montes
Course: Imagen Digital
Year: 2025
Version: 4.0.0

Dependencies:
    - fastapi: Framework web y excepciones HTTP
    - starlette: Excepciones base de Starlette

Usage:
    from fastapi import FastAPI
    from app.middleware.error_handler import setup_error_handlers

    app = FastAPI()
    setup_error_handlers(app)
"""

import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware:
    """
    Middleware ASGI para captura y formateo centralizado de errores HTTP.

    Intercepta todas las excepciones no manejadas en el pipeline de procesamiento
    de requests, las registra con logging estructurado y retorna respuestas JSON
    estandarizadas. Además mide tiempo de procesamiento agregando header
    X-Process-Time a todas las respuestas.

    Attributes:
        app: Aplicación ASGI envuelta por el middleware

    Notes:
        Implementa protocolo ASGI completo mediante __call__ con scope, receive
        y send. Solo procesa requests HTTP (scope["type"] == "http"), ignorando
        otros tipos como WebSocket.
    """

    def __init__(self, app):
        """
        Inicializa middleware envolviendo aplicación ASGI.

        Args:
            app: Aplicación ASGI (típicamente instancia FastAPI) a envolver
        """
        self.app = app

    async def __call__(self, scope, receive, send):
        """
        Procesa request ASGI con manejo de errores y medición de tiempo.

        Args:
            scope: Diccionario con información del request (tipo, path, headers)
            receive: Callable asíncrono para recibir eventos del cliente
            send: Callable asíncrono para enviar eventos al cliente

        Notes:
            Pipeline de procesamiento:
                1. Verifica que sea request HTTP, si no pasa directo
                2. Inicia timer para medición de tiempo
                3. Envuelve send para agregar header X-Process-Time
                4. Ejecuta aplicación dentro de try-catch
                5. Captura excepciones no manejadas y retorna error 500

            El header X-Process-Time se agrega en segundos con 4 decimales,
            útil para monitoreo de performance y detección de endpoints lentos.

            Las excepciones capturadas aquí son las que escaparon de todos los
            exception handlers específicos, típicamente errores inesperados o
            bugs en el código de la aplicación.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{time.time() - start_time:.4f}".encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            logger.error(f"Error no manejado: {exc}")
            logger.error(traceback.format_exc())

            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "type": "error",
                    "message": "Error interno del servidor",
                    "details": str(exc) if logger.level == logging.DEBUG else None,
                    "error_code": "INTERNAL_SERVER_ERROR"
                }
            )
            await response(scope, receive, send)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Maneja errores de validación de Pydantic en request bodies y parámetros.

    Args:
        request: Objeto Request de FastAPI con información del request
        exc: Excepción RequestValidationError conteniendo detalles de validación

    Returns:
        JSONResponse con código 422 y detalles estructurados de errores de validación

    Notes:
        Transforma los errores de Pydantic en formato amigable:
            - field: Path del campo con error
            - message: Mensaje descriptivo del error
            - type: Tipo de error de validación de Pydantic

        Código HTTP 422 Unprocessable Entity es el estándar para errores de
        validación según especificación WebDAV y adoptado por FastAPI.

        Los errores se registran como warnings, no errors, porque son esperables
        con input de usuarios y no indican bugs en el código.
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })

    logger.warning(f"Error de validación: {errors}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "type": "error",
            "message": "Error de validación de datos",
            "details": errors,
            "error_code": "VALIDATION_ERROR"
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Maneja excepciones HTTP estándar lanzadas explícitamente en el código.

    Args:
        request: Objeto Request de FastAPI
        exc: Excepción HTTPException con status_code y detail

    Returns:
        JSONResponse con código de estado de la excepción y mensaje formateado

    Notes:
        Maneja excepciones lanzadas con raise HTTPException(), típicamente:
            - 400 Bad Request
            - 401 Unauthorized
            - 403 Forbidden
            - 404 Not Found
            - 409 Conflict

        El error_code se genera dinámicamente como HTTP_{status_code} para
        facilitar parsing programático en clientes.

        Se registra como warning porque son errores controlados que indican
        problemas con el request del usuario, no bugs del servidor.
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": "error",
            "message": exc.detail,
            "error_code": f"HTTP_{exc.status_code}"
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """
    Maneja cualquier excepción no capturada por handlers específicos.

    Args:
        request: Objeto Request de FastAPI
        exc: Excepción de tipo genérico Exception

    Returns:
        JSONResponse con error 500 y mensaje genérico. Detalles solo en modo debug

    Notes:
        Handler de último recurso para excepciones inesperadas que escaparon
        de los otros handlers. Típicamente indica bugs en el código.

        Seguridad:
            - En producción (log level != DEBUG): mensaje genérico sin detalles
            - En desarrollo (DEBUG): incluye traceback completo

        Evita exposición de información sensible del sistema en producción
        (rutas de archivos, variables internas, estructura de código).

        El traceback completo se registra siempre en logs para debugging,
        solo se oculta en la respuesta HTTP al cliente.
    """
    logger.error(f"Error inesperado: {exc}")
    logger.error(traceback.format_exc())

    is_dev = logger.level == logging.DEBUG

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "type": "error",
            "message": "Ha ocurrido un error inesperado",
            "details": str(exc) if is_dev else "Contacta al administrador",
            "error_code": "UNEXPECTED_ERROR"
        }
    )


def setup_error_handlers(app):
    """
    Registra todos los exception handlers en la aplicación FastAPI.

    Args:
        app: Instancia de FastAPI donde registrar los handlers

    Notes:
        Orden de registro es importante: handlers más específicos primero.
        FastAPI busca handlers en orden de registro y usa el primero que matchea.

        Jerarquía de manejo:
            1. RequestValidationError - Errores de validación Pydantic
            2. StarletteHTTPException - Errores HTTP explícitos
            3. Exception - Catch-all para errores inesperados

        Los handlers se ejecutan antes que el ErrorHandlerMiddleware, que solo
        captura excepciones que escaparon de todos los handlers registrados.
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    logger.info("Manejadores de errores configurados")