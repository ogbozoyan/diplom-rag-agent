from __future__ import annotations

import io
import logging
from typing import Optional, Any

from PIL import Image
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from langgraph.graph.state import CompiledStateGraph

from app_config import AppConfig
from embedding import init_embeddings, resolve_vector_size
from langgraph_state_nodes import build_graph, RAGState
from logger_config import init_logging
from pg_vector_helpers import psycopg_dsn_from_sqlalchemy, ensure_pgvector_store, ingest_incremental
from timer import timed

init_logging()

_log = logging.getLogger(__name__)


# =========================
# Entry
# =========================

def save_graph_png( graph: CompiledStateGraph[Any, Any, Any, Any] ):
    png_bytes = graph.get_graph().draw_mermaid_png()
    graph_img = Image.open(io.BytesIO(png_bytes))
    graph_img.save("graph.png")


def run( question: str ):
    cfg = AppConfig.from_env()
    _log.info(
        "Config: docs_dir=%s context_dir=%s reindex=%s enable_web=%s",
        cfg.docs_dir, cfg.context_dir, cfg.reindex, cfg.enable_web,
    )

    if not cfg.docs_dir.exists():
        raise RuntimeError(f"DOCS_DIR does not exist: {cfg.docs_dir}")

    with timed("init_embeddings"):
        embeddings: OpenAIEmbeddings | HuggingFaceEmbeddings = init_embeddings()

    vector_size: int = resolve_vector_size(cfg, embeddings)
    docs_table: str = cfg.docs_table or f"rag_docs_{vector_size}"
    ctx_table: str = cfg.ctx_table or f"rag_ctx_{vector_size}"

    with timed("pg_engine"):
        pg_engine: PGEngine = PGEngine.from_connection_string(cfg.pg_url)

    pg_dsn: str = psycopg_dsn_from_sqlalchemy(cfg.pg_url)
    _log.info("Postgres DSN resolved (driver stripped)")

    # docs store
    docs_vs: PGVectorStore = ensure_pgvector_store(
        cfg, pg_engine, embeddings, docs_table, vector_size,
        reindex = cfg.reindex,
        pg_dsn = pg_dsn,
    )
    ingest_incremental(cfg, pg_dsn, docs_vs, cfg.docs_dir, docs_table, reindex = cfg.reindex)

    # context store (optional)
    ctx_vs: Optional[PGVectorStore] = None
    if cfg.context_dir and cfg.context_dir.exists():
        ctx_vs = ensure_pgvector_store(
            cfg, pg_engine, embeddings, ctx_table, vector_size, reindex = cfg.reindex,
        )
        ingest_incremental(cfg, pg_dsn, docs_vs, cfg.docs_dir, docs_table, reindex = cfg.reindex)
    else:
        _log.info("Context disabled (dir missing): %s", cfg.context_dir)

    # LLM
    with timed("init_chat_model"):
        model: BaseChatModel = init_chat_model(cfg.chat_model, temperature = cfg.temperature)

    graph = build_graph(model = model, docs_vs = docs_vs, ctx_vs = ctx_vs, cfg = cfg)

    save_graph_png(graph)

    state: RAGState = {
        "question": question,
        "seed_urls": [],
        "queries": [],
        "web_evidence": [],
        "doc_evidence": [],
        "ctx_evidence": [],
        "final_answer": "",
        "errors": [],
    }

    with timed("graph_invoke"):
        result: dict[str, Any] | Any = graph.invoke(state)

    print(result["final_answer"])


if __name__ == "__main__":
    run(
        """
            какие браузерные атаки существуют ? comet
            """,
    )
