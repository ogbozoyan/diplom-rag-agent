# =========================
# Logging
# =========================
from __future__ import annotations

import logging
import os
from logging import Logger


def setup_logging( ) -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    fmt = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    logging.basicConfig(level = level, format = fmt)
    # reduce noisy libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("pypdf._reader").setLevel(logging.INFO)

    return logging.getLogger("agent_rag")


logger: Logger = setup_logging()
