from __future__ import annotations

from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore

from app_config import AppConfig
from embedding import init_embeddings, resolve_vector_size
from langgraph_state_nodes import build_graph, RAGState
from logger_config import logger
from pg_vector_helpers import psycopg_dsn_from_sqlalchemy, ensure_pgvector_store, ingest_incremental
from timer import timed


# =========================
# Entry
# =========================

def run( question: str ):
    cfg = AppConfig.from_env()
    logger.info(
        "Config: docs_dir=%s context_dir=%s reindex=%s enable_web=%s",
        cfg.docs_dir, cfg.context_dir, cfg.reindex, cfg.enable_web,
    )

    if not cfg.docs_dir.exists():
        raise RuntimeError(f"DOCS_DIR does not exist: {cfg.docs_dir}")

    with timed("init_embeddings"):
        embeddings = init_embeddings()

    vector_size = resolve_vector_size(cfg, embeddings)
    docs_table = cfg.docs_table or f"rag_docs_{vector_size}"
    ctx_table = cfg.ctx_table or f"rag_ctx_{vector_size}"

    with timed("pg_engine"):
        pg_engine = PGEngine.from_connection_string(cfg.pg_url)

    pg_dsn = psycopg_dsn_from_sqlalchemy(cfg.pg_url)
    logger.info("Postgres DSN resolved (driver stripped)")

    # docs store
    docs_vs = ensure_pgvector_store(
        cfg, pg_engine, embeddings, docs_table, vector_size,
        reindex = cfg.reindex,
        pg_dsn = pg_dsn,
    )
    ingest_incremental(cfg, pg_dsn, pg_engine, docs_vs, cfg.docs_dir, docs_table, reindex = cfg.reindex)

    # context store (optional)
    ctx_vs: Optional[PGVectorStore] = None
    if cfg.context_dir and cfg.context_dir.exists():
        ctx_vs = ensure_pgvector_store(
            cfg, pg_engine, embeddings, ctx_table, vector_size, reindex = cfg.reindex,
        )
        ingest_incremental(cfg, pg_dsn, pg_engine, docs_vs, cfg.docs_dir, docs_table, reindex = cfg.reindex)
    else:
        logger.info("Context disabled (dir missing): %s", cfg.context_dir)

    # LLM
    with timed("init_chat_model"):
        model = init_chat_model(cfg.chat_model, temperature = cfg.temperature)

    graph = build_graph(model = model, docs_vs = docs_vs, ctx_vs = ctx_vs, cfg = cfg)

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
        result = graph.invoke(state)

    print(result["final_answer"])


if __name__ == "__main__":
    run("Что такое LLM и как его применяют в кибербезе?")
