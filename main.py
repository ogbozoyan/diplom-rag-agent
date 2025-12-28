from __future__ import annotations

import hashlib
import json
import logging
import operator
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import psycopg
import requests
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from psycopg import sql as psql
from sqlalchemy.engine import make_url

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

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore

# --- Silence HF tokenizers fork warning (best practice) ---
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Optional .env support ---
try:
    from dotenv import load_dotenv  # uv add python-dotenv

    load_dotenv()
except Exception:
    pass


# =========================
# Logging
# =========================

def setup_logging( ) -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
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

    return logging.getLogger("agent_rag")


logger = setup_logging()


class timed:
    def __init__( self, label: str ):
        self.label = label
        self.t0 = 0.0

    def __enter__( self ):
        self.t0 = time.time()
        logger.info("%s: start", self.label)
        return self

    def __exit__( self, exc_type, exc, tb ):
        dt = time.time() - self.t0
        if exc:
            logger.exception("%s: failed in %.3fs", self.label, dt)
        else:
            logger.info("%s: done in %.3fs", self.label, dt)
        return False


# =========================
# Config
# =========================

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


# =========================
# Embeddings
# =========================

def init_embeddings( ):
    """
    Prefer OpenAI embeddings if OPENAI_API_KEY is set.
    Otherwise use local HuggingFace embeddings (sentence-transformers).
    """
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        logger.info("Embeddings: OpenAI model=%s", model)
        return OpenAIEmbeddings(model = model)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embeddings: HuggingFace model=%s", model_name)
        return HuggingFaceEmbeddings(model_name = model_name)


def resolve_vector_size( cfg: AppConfig, embeddings ) -> int:
    """
    Prefer VECTOR_SIZE from env (cfg.vector_size).
    If it's 0, probe with a single embed_query (OpenAI = extra API call).
    """
    if cfg.vector_size > 0:
        return cfg.vector_size

    with timed("vector_size_probe"):
        try:
            size = len(embeddings.embed_query("vector size probe"))
            return int(size)
        except Exception as e:
            raise RuntimeError(
                "VECTOR_SIZE is not set and probing failed. "
                "Set VECTOR_SIZE explicitly (e.g. 384 for all-MiniLM-L6-v2, 1536 for text-embedding-3-small).",
            ) from e


# =========================
# Documents loading
# =========================

def load_documents_from_dir( root: Path ):
    docs = []
    if not root.exists():
        logger.warning("Docs dir does not exist: %s", root)
        return docs

    for p in sorted(root.glob("**/*")):
        if p.is_dir():
            continue

        suffix = p.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(p))
                loaded = loader.load()
                for d in loaded:
                    d.metadata["source_file"] = str(p)
                    page = d.metadata.get("page", None)
                    if page is not None:
                        d.metadata["page_human"] = int(page) + 1
                docs.extend(loaded)

            elif suffix == ".pptx":
                if not HAS_PPTX:
                    logger.warning("Skip PPTX (unstructured not installed): %s", p)
                    continue
                loader = UnstructuredPowerPointLoader(str(p), mode = "elements")
                loaded = loader.load()
                for d in loaded:
                    d.metadata["source_file"] = str(p)
                docs.extend(loaded)

            else:
                continue
        except Exception:
            logger.exception("Failed to load file: %s", p)

    logger.info("Loaded documents: %d (from %s)", len(docs), root)
    return docs


def split_documents( docs, chunk_size: int, chunk_overlap: int ):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)
    logger.info("Split into chunks: %d (chunk_size=%d overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks


# =========================
# PGVector helpers
# =========================

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


# =========================
# Web retrieve (optional)
# =========================

def _clean_text( s: str ) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def fetch_url_text( url: str, timeout_sec: int = 15 ) -> str:
    r = requests.get(
        url,
        timeout = timeout_sec,
        headers = { "User-Agent": "Mozilla/5.0 (compatible; AgentRAG/1.0)" },
    )
    r.raise_for_status()
    html = r.text

    if HAS_BS4:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return _clean_text(soup.get_text(" "))

    return _clean_text(re.sub(r"<[^>]+>", " ", html))


@dataclass
class Evidence:
    source_type: Literal["web", "doc", "context"]
    source: str
    locator: str
    title: Optional[str]
    snippet: str
    score: float


def web_retrieve_evidence(
        question: str, seed_urls: list[str], max_search_results: int = 5, max_pages_fetch: int = 4,
) -> list[Evidence]:
    if not HAS_DDG and not seed_urls:
        logger.warning("Web disabled: DuckDuckGo tool not available and no seed_urls provided")
        return []

    urls: list[str] = []
    if seed_urls:
        urls = seed_urls[:max_pages_fetch]
    else:
        ddg = DuckDuckGoSearchResults(output_format = "list")
        results = ddg.invoke({ "query": question, "max_results": max_search_results })
        urls = [r.get("link") for r in results if r.get("link")][:max_pages_fetch]

    page_texts = []
    for u in urls:
        try:
            txt = fetch_url_text(u)
            if len(txt) >= 200:
                page_texts.append((u, txt))
        except Exception:
            logger.exception("Failed to fetch url: %s", u)

    out: list[Evidence] = []
    for url, txt in page_texts:
        out.append(
            Evidence(
                source_type = "web",
                source = url,
                locator = "fulltext",
                title = None,
                snippet = txt[:900],
                score = 0.0,
            ),
        )
    return out


# =========================
# LangGraph state & nodes
# =========================

class RAGState(TypedDict):
    question: str
    seed_urls: list[str]
    queries: list[str]

    web_evidence: Annotated[list[Evidence], operator.add]
    doc_evidence: Annotated[list[Evidence], operator.add]
    ctx_evidence: Annotated[list[Evidence], operator.add]

    final_answer: str
    errors: Annotated[list[str], operator.add]


def plan_node( state: RAGState, model ) -> dict:
    q = state["question"]
    prompt = (
        "Generate 2-5 short search query variants for the user question.\n"
        "Rules:\n"
        "- Keep them compact (keywords)\n"
        "- Include RU/EN variant if useful\n"
        "- Output JSON array of strings ONLY\n\n"
        f"Question: {q}"
    )
    resp = model.invoke([SystemMessage(content = prompt)])
    text_ = resp.content if hasattr(resp, "content") else str(resp)
    try:
        queries = json.loads(text_)
        if not isinstance(queries, list):
            queries = [q]
        queries = [str(x) for x in queries][:5]
    except Exception:
        queries = [q]

    logger.info("Plan queries=%s", queries)
    return { "queries": queries }


def docs_agent_node( state: RAGState, docs_vs: PGVectorStore, top_k: int ) -> dict:
    question = state["question"]
    with timed("docs_retrieval"):
        pairs = docs_vs.similarity_search_with_score(question, k = top_k)

    out: list[Evidence] = []
    for doc, score in pairs:
        src = doc.metadata.get("source_file", "unknown_file")
        page = doc.metadata.get("page_human")
        locator = f"page={page}" if page else "chunk"
        out.append(
            Evidence(
                source_type = "doc",
                source = str(src),
                locator = locator,
                title = None,
                snippet = _clean_text(doc.page_content)[:900],
                score = float(score),
            ),
        )

    logger.info("Docs evidence=%d", len(out))
    return { "doc_evidence": out }


def context_agent_node( state: RAGState, ctx_vs: Optional[PGVectorStore], top_k: int ) -> dict:
    if ctx_vs is None:
        return { "ctx_evidence": [] }

    question = state["question"]
    with timed("ctx_retrieval"):
        pairs = ctx_vs.similarity_search_with_score(question, k = top_k)

    out: list[Evidence] = []
    for doc, score in pairs:
        src = doc.metadata.get("source_file", "context")
        locator = doc.metadata.get("section", "chunk")
        out.append(
            Evidence(
                source_type = "context",
                source = str(src),
                locator = str(locator),
                title = None,
                snippet = _clean_text(doc.page_content)[:900],
                score = float(score),
            ),
        )

    logger.info("Context evidence=%d", len(out))
    return { "ctx_evidence": out }


def answer_agent_node( state: RAGState, model ) -> dict:
    q = state["question"]
    evidence = state.get("web_evidence", []) + state.get("doc_evidence", []) + state.get("ctx_evidence", [])
    evidence = sorted(evidence, key = lambda e: e.score)[:20]

    if not evidence:
        return { "final_answer": "В источниках ничего релевантного не найдено под этот вопрос." }

    lines = []
    for i, e in enumerate(evidence, start = 1):
        lines.append(
            f"[{i}] type={e.source_type} src={e.source} loc={e.locator}\n"
            f"snippet: {e.snippet}\n",
        )
    evidence_block = "\n".join(lines)

    system = (
        "You are a RAG assistant. Answer ONLY from the provided evidence.\n"
        "Hard rules:\n"
        "- Do NOT use outside knowledge.\n"
        "- Every meaningful claim must include a citation like [1] or [2].\n"
        "- If evidence is insufficient, say so and cite what you do have.\n"
        "- Answer in Russian.\n"
        "- After the answer, output a 'Источники:' section with numbered footnotes mapping [n] -> source.\n"
    )
    user = (
        f"Вопрос: {q}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Сформируй финальный ответ по алгоритму:\n"
        "1) краткий вывод\n"
        "2) детали/обоснование\n"
        "3) если есть расхождения источников — явно покажи\n"
    )

    with timed("llm_answer"):
        resp = model.invoke([SystemMessage(content = system), HumanMessage(content = user)])
    answer_text = resp.content if hasattr(resp, "content") else str(resp)

    if "Источники:" not in answer_text:
        foot = ["\nИсточники:"]
        for i, e in enumerate(evidence, start = 1):
            foot.append(f"[{i}] {e.source} ({e.locator})")
        answer_text = answer_text.rstrip() + "\n" + "\n".join(foot)

    return { "final_answer": answer_text }


def build_graph( model, docs_vs: PGVectorStore, ctx_vs: Optional[PGVectorStore], cfg: AppConfig ):
    g = StateGraph(RAGState)

    g.add_node("plan", lambda s: plan_node(s, model))
    g.add_node("docs_agent", lambda s: docs_agent_node(s, docs_vs, cfg.doc_top_k))
    g.add_node("context_agent", lambda s: context_agent_node(s, ctx_vs, cfg.ctx_top_k))
    g.add_node("answer_agent", lambda s: answer_agent_node(s, model))

    g.add_edge(START, "plan")
    g.add_edge("plan", "docs_agent")
    g.add_edge("docs_agent", "context_agent")
    g.add_edge("context_agent", "answer_agent")
    g.add_edge("answer_agent", END)

    return g.compile()


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
