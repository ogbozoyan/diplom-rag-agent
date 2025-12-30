# =========================
# Embeddings
# =========================

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_config import AppConfig, HAS_BS4, HAS_DDG
from app_config import HAS_PPTX
from logger_config import logger
from timer import timed


@dataclass
class Evidence:
    source_type: Literal["web", "doc", "context"]
    source: str
    locator: str
    title: Optional[str]
    snippet: str
    score: float


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


# =========================
# Documents loading
# =========================

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


# =========================
# Web retrieve (optional)
# =========================

def split_documents( docs, chunk_size: int, chunk_overlap: int ):
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    chunks = splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)
    logger.info("Split into chunks: %d (chunk_size=%d overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks


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
