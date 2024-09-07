import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def timer_decorator(func):
    """Decorador para medir el tiempo de ejecución de una función."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Función {func.__name__} ejecutada en {end_time - start_time:.2f} segundos")
        return result
    return wrapper

def log_decorator(func):
    """Decorador para registrar la entrada y salida de una función."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Llamando a la función: {func.__name__}")
        result = func(*args, **kwargs)
        logger.info(f"Función {func.__name__} completada")
        return result
    return wrapper

def error_handler(func):
    """Decorador para manejar y registrar errores."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error en la función {func.__name__}: {str(e)}")
            raise
    return wrapper
