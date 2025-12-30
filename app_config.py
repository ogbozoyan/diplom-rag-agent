# =========================
# Config
# =========================

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# --- Silence HF tokenizers fork warning (best practice) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --- Optional .env support ---
@dataclass(frozen = True)
class AppConfig:
    docs_dir: Path
    context_dir: Optional[Path]
    enable_web: bool

    pg_url: str
    schema: str
    docs_table: str
    ctx_table: str

    chunk_size: int
    chunk_overlap: int

    doc_top_k: int
    ctx_top_k: int

    vector_size: int  # 0 means "probe"
    reindex: bool

    chat_model: str
    temperature: float

    # stable namespace for uuid5 ids (do NOT change after first ingest)
    uuid_namespace: uuid.UUID

    @staticmethod
    def from_env( ) -> "AppConfig":
        docs_dir = Path(os.getenv("DOCS_DIR", "./data/docs")).resolve()
        context_dir_raw = os.getenv("CONTEXT_DIR", "./data/context")
        context_dir = Path(context_dir_raw).resolve() if context_dir_raw else None

        enable_web = os.getenv("ENABLE_WEB", "false").lower() == "true"

        pg_url = os.getenv(
            "PGVECTOR_URL",
            "postgresql+psycopg://langchain:langchain@localhost:6024/langchain",
        )
        schema = os.getenv("PG_SCHEMA", "public")

        chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        doc_top_k = int(os.getenv("DOC_TOP_K", "8"))
        ctx_top_k = int(os.getenv("CTX_TOP_K", "6"))

        vector_size = int(os.getenv("VECTOR_SIZE", "0"))  # set this to avoid probe for OpenAI
        reindex = os.getenv("REINDEX", "false").lower() == "true"

        chat_model = os.getenv("CHAT_MODEL", "openai:gpt-5-mini-2025-08-07")
        temperature = float(os.getenv("TEMP", "0"))

        # фиксируем namespace: если поменять — ids изменятся и будет дубляж
        ns_raw = os.getenv("UUID_NAMESPACE", "6ba7b811-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL default
        ns = uuid.UUID(ns_raw)

        # таблицы: если не заданы — используем шаблон с vector_size (который узнаем позже)
        docs_table = os.getenv("DOCS_TABLE", "")
        ctx_table = os.getenv("CTX_TABLE", "")

        return AppConfig(
            docs_dir = docs_dir,
            context_dir = context_dir,
            enable_web = enable_web,
            pg_url = pg_url,
            schema = schema,
            docs_table = docs_table,
            ctx_table = ctx_table,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            doc_top_k = doc_top_k,
            ctx_top_k = ctx_top_k,
            vector_size = vector_size,
            reindex = reindex,
            chat_model = chat_model,
            temperature = temperature,
            uuid_namespace = ns,
        )


try:
    from dotenv import load_dotenv  # uv add python-dotenv

    load_dotenv()
except Exception:
    pass
# Optional PPTX loader (unstructured)

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader

    HAS_PPTX = True
except Exception:
    HAS_PPTX = False
# Web search tool (optional)

try:
    from langchain_community.tools import DuckDuckGoSearchResults

    HAS_DDG = True
except Exception:
    HAS_DDG = False
# Simple HTML -> text


try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except Exception:
    HAS_BS4 = False
