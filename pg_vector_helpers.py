# =========================
# PGVector helpers
# =========================
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Sequence

import psycopg
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore
from psycopg import sql as psql
from sqlalchemy.engine import make_url

from app_config import AppConfig
from embedding import load_documents_from_dir, split_documents
from logger_config import logger
from timer import timed


def pg_table_exists( pg_dsn: str, schema: str, table: str ) -> bool:
    full = f"{schema}.{table}"
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (full,))
            return cur.fetchone()[0] is not None


def pg_table_has_rows( pg_dsn: str, schema: str, table: str ) -> bool:
    q = psql.SQL('SELECT 1 FROM {}.{} LIMIT 1').format(
        psql.Identifier(schema),
        psql.Identifier(table),
    )
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            return cur.fetchone() is not None


def chunk_uuid( cfg: AppConfig, d, chunk_id: int ) -> str:
    """
    Stable UUID for a chunk.
    If you want "update-in-place" when content changes, don't include content hash.
    Here we include content hash -> content change = new id (old row remains).
    """
    src = str(d.metadata.get("source_file", "unknown"))
    page = str(d.metadata.get("page_human", ""))
    content_hash = hashlib.sha1((d.page_content or "").encode("utf-8")).hexdigest()[:16]
    key = f"{src}|{page}|{chunk_id}|{content_hash}"
    return str(uuid.uuid5(cfg.uuid_namespace, key))


def fetch_existing_ids( pg_dsn: str, schema: str, table: str, ids: Sequence[str], batch: int = 5000 ) -> set[str]:
    """
    ids: list[str] of UUID strings
    """
    if not ids:
        return set()

    q = psql.SQL('SELECT langchain_id::text FROM {}.{} WHERE langchain_id = ANY(%s)').format(
        psql.Identifier(schema),
        psql.Identifier(table),
    )

    out: set[str] = set()
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            for i in range(0, len(ids), batch):
                part = [uuid.UUID(x) for x in ids[i:i + batch]]
                cur.execute(q, (part,))
                out.update(r[0] for r in cur.fetchall())
    return out


def ensure_pgvector_store(
        cfg: AppConfig,
        pg_engine: PGEngine,
        embeddings,
        table_name: str,
        vector_size: int,
        *,
        reindex: bool,
        pg_dsn: str,
) -> PGVectorStore:
    exists = pg_table_exists(pg_dsn, cfg.schema, table_name)

    if reindex:
        logger.warning("REINDEX=true -> will drop & recreate table %s.%s", cfg.schema, table_name)

    if (not exists) or reindex:
        pg_engine.init_vectorstore_table(
            table_name = table_name,
            schema_name = cfg.schema,
            vector_size = vector_size,
            overwrite_existing = reindex,
            id_column = "langchain_id",
            store_metadata = True,
        )
        logger.info("Table ensured: %s.%s (vector_size=%d)", cfg.schema, table_name, vector_size)
    else:
        logger.info("Table exists: %s.%s (skip create)", cfg.schema, table_name)

    return PGVectorStore.create_sync(
        engine = pg_engine,
        embedding_service = embeddings,
        table_name = table_name,
        schema_name = cfg.schema,
    )


def psycopg_dsn_from_sqlalchemy( sa_url: str ) -> str:
    """
    sa_url example: postgresql+psycopg://user:pass@host:port/db
    psycopg needs:  postgresql://user:pass@host:port/db
    """
    u = make_url(sa_url)
    u2 = u.set(drivername = "postgresql")
    return u2.render_as_string(hide_password = False)


def ingest_incremental(
        cfg: AppConfig,
        pg_dsn: str,
        pg_engine: PGEngine,
        store: PGVectorStore,
        docs_dir: Path,
        table_name: str,
        *,
        reindex: bool,
):
    with timed(f"ingest:{table_name}"):
        raw = load_documents_from_dir(docs_dir)
        chunks = split_documents(raw, cfg.chunk_size, cfg.chunk_overlap)

        ids: list[str] = []
        for i, d in enumerate(chunks):
            d.metadata["chunk_id"] = i
            ids.append(chunk_uuid(cfg, d, i))

        if reindex:
            logger.info("REINDEX=true -> inserting all chunks=%d", len(chunks))
            store.add_documents(chunks, ids = ids)
            return

        # incremental: insert only missing ids
        if not pg_table_exists(pg_dsn, cfg.schema, table_name):
            logger.info("Table not found yet, inserting all chunks=%d", len(chunks))
            store.add_documents(chunks, ids = ids)
            return

        existing = fetch_existing_ids(pg_dsn, cfg.schema, table_name, ids)
        missing_docs = []
        missing_ids = []
        for d, id_ in zip(chunks, ids):
            if id_ not in existing:
                missing_docs.append(d)
                missing_ids.append(id_)

        logger.info("Chunks total=%d existing=%d missing=%d", len(chunks), len(existing), len(missing_docs))
        if missing_docs:
            store.add_documents(missing_docs, ids = missing_ids)
        else:
            logger.info("No new chunks to ingest")
