from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar

# --- MDC-style context (like Logback's MDC) ---
trace_id_var: ContextVar[str] = ContextVar("trace_id", default = "-")
span_id_var: ContextVar[str] = ContextVar("span_id", default = "-")


class ContextFilter(logging.Filter):
    """
    Фильтр для добавления идентификаторов трассировки в записи журналов.

    Добавляет в запись журнала два поля:
    - traceId: идентификатор трассировки
    - spanId: идентификатор спана
    """

    def filter( self, record: logging.LogRecord ) -> bool:
        record.traceId = trace_id_var.get()
        record.spanId = span_id_var.get()
        return True


def _make_formatter( use_colors: bool ) -> logging.Formatter:
    """
    Создает форматировщик для вывода журналов.

    Args:
        use_colors: Если True, будет использоваться цветной вывод (если доступен)

    Returns:
        Настроенный форматировщик для журналов
    """
    if use_colors:
        try:
            from colorlog import ColoredFormatter  # type: ignore
        except Exception as e:
            print("Не удалось загрузить библиотеку colorlog")
            use_colors = False

    if use_colors:
        try:
            import colorama  # type: ignore
            colorama.just_fix_windows_console()
        except Exception as e:
            print("Не удалось загрузить библиотеку colorama")
            pass

        return ColoredFormatter(
            "%(white)s%(asctime)s.%(msecs)03d%(reset)s "
            "%(log_color)s%(levelname)-5s%(reset)s "
            "[%(purple)s%(threadName).15s%(reset)s] "
            "[%(traceId)s, %(spanId)s] "
            "%(cyan)s%(name).36s:%(lineno)d%(reset)s - %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
            reset = True,
            log_colors = {
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

    return logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-5s "
        "[%(threadName).15s] "
        "[%(traceId)s, %(spanId)s] "
        "%(name).36s:%(lineno)d - %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )


def init_logging( level: str | int | None = None, use_colors: bool | None = None ) -> None:
    """
    Инициализирует систему ведения журналов с цветным консольным форматированием.

    Настраивает корневой логгер для вывода в стандартный поток вывода с цветным
    форматированием (если доступно) и поддержкой идентификаторов трассировки.

    Args:
        level: Уровень логгирования (по умолчанию: INFO или значение из LOG_LEVEL)
        use_colors: Использовать ли цвета (по умолчанию: определяется автоматически)

    Environment Variables:
        LOG_LEVEL: Переопределяет уровень логгирования (по умолчанию: INFO)
        LOG_COLOR: Управление цветом - "1" (всегда), "0" (никогда), "auto" (по умолчанию: только в TTY)
    """
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    if use_colors is None:
        env_color = os.getenv("LOG_COLOR", "1").lower()
        use_colors = True if env_color == "1" else False if env_color == "0" else sys.stdout.isatty()

    handler = logging.StreamHandler(stream = sys.stdout)
    handler.setFormatter(_make_formatter(use_colors))
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers[:] = []  # prevent duplicate logs on reload
    root.addHandler(handler)
    root.setLevel(lvl)

    # Make Uvicorn loggers flow through our root formatter
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
        lg.setLevel(lvl)


# --- helpers to set correlation ids wherever you are ---
def set_trace_context( trace_id: str = "-", span_id: str = "-" ) -> None:
    """
    Устанавливает контекст трассировки для текущего потока выполнения.

    Args:
        trace_id: Идентификатор трассировки (по умолчанию: "-")
        span_id: Идентификатор спана (по умолчанию: "-")
    """
    trace_id_var.set(trace_id)
    span_id_var.set(span_id)


class trace_context:
    """
    Контекстный менеджер для временной установки идентификаторов трассировки.

    Позволяет временно установить идентификаторы трассировки в блоке кода,
    автоматически восстанавливая предыдущие значения при выходе из блока.

    Example:
        with trace_context("trace-123", "span-456"):
            # В этом блоке все логи будут содержать traceId="trace-123" и spanId="span-456"
            logger.info("Сообщение в контексте трассировки")
    """

    def __init__( self, trace_id: str = "-", span_id: str = "-" ):
        self._t = None
        self._s = None
        self.trace_id = trace_id
        self.span_id = span_id

    def __enter__( self ):
        self._t = trace_id_var.set(self.trace_id)
        self._s = span_id_var.set(self.span_id)

    def __exit__( self, exc_type, exc, tb ):
        if self._t is not None:
            trace_id_var.reset(self._t)
        if self._s is not None:
            span_id_var.reset(self._s)
