from __future__ import annotations

import json
import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
# Loaders
from langchain_community.document_loaders import PyPDFLoader
# Vector store / embeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated

# Optional PPTX loader (unstructured)
try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader

    HAS_PPTX = True
except Exception:
    HAS_PPTX = False

# Web search tool
from langchain_community.tools import DuckDuckGoSearchResults

# Simple HTML -> text
try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except Exception:
    HAS_BS4 = False

import requests


# Embeddings
def init_embeddings( ):
    """
    Prefer OpenAI embeddings if OPENAI_API_KEY is set.
    Otherwise use local HuggingFace embeddings (sentence-transformers).
    """
    if os.getenv("OPENAI_API_KEY"):
        # pip install langchain-openai
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model = os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    else:
        # pip install langchain-huggingface sentence-transformers
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name = os.getenv(
                "HF_EMBED_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        )


from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_or_load_index( docs_dir: Path, persist_dir: Path, embeddings, reindex: bool = False ):
    persist_dir.mkdir(parents = True, exist_ok = True)

    raw_docs = load_documents_from_dir(docs_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 200)
    split_docs = splitter.split_documents(raw_docs)

    # <-- вот это чинит ValueError: got ['rus'] which is a list
    split_docs = filter_complex_metadata(split_docs)

    vs = Chroma.from_documents(
        split_docs,
        embedding = embeddings,
        persist_directory = str(persist_dir),
        collection_name = "docs",
    )
    return vs


# ----------------------------
# Data models
# ----------------------------

@dataclass
class Evidence:
    source_type: Literal["web", "doc", "context"]
    source: str  # url or file path or context-id
    locator: str  # page/slide/section/etc
    title: Optional[str]
    snippet: str
    score: float


def _clean_text( s: str ) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def format_citation( e: Evidence ) -> str:
    # Used in footnotes.
    if e.source_type == "web":
        return f"{e.source} ({e.locator})"
    return f"{e.source} ({e.locator})"


# ----------------------------
# Index building
# ----------------------------

def load_documents_from_dir( root: Path ):
    docs = []
    for p in sorted(root.glob("**/*")):
        if p.is_dir():
            continue
        suffix = p.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(p))
            loaded = loader.load()  # each page is a Document with metadata.page
            for d in loaded:
                d.metadata["source_file"] = str(p)
                # PyPDFLoader uses 0-based page in some versions, normalize to 1-based for humans:
                page = d.metadata.get("page", None)
                if page is not None:
                    d.metadata["page_human"] = int(page) + 1
            docs.extend(loaded)

        elif suffix == ".pptx":
            if not HAS_PPTX:
                # Skip if unstructured isn't installed
                continue
            loader = UnstructuredPowerPointLoader(str(p), mode = "elements")
            loaded = loader.load()
            # unstructured metadata varies; keep source file at least
            for d in loaded:
                d.metadata["source_file"] = str(p)
            docs.extend(loaded)

        else:
            # ignore other formats in this minimal example
            continue
    return docs


def build_or_load_faiss_index(
        docs_dir: Path,
        index_dir: Path,
        embeddings,
        reindex: bool = False,
):
    index_dir.mkdir(parents = True, exist_ok = True)
    faiss_path = index_dir / "faiss_index"

    if faiss_path.exists() and not reindex:
        return FAISS.load_local(
            str(faiss_path),
            embeddings = embeddings,
            allow_dangerous_deserialization = True,
        )

    raw_docs = load_documents_from_dir(docs_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200")),
    )
    split_docs = splitter.split_documents(raw_docs)

    vs = FAISS.from_documents(split_docs, embeddings)
    vs.save_local(str(faiss_path))
    return vs


# ----------------------------
# Web fetch & web mini-index per question
# ----------------------------

def fetch_url_text( url: str, timeout_sec: int = 15 ) -> str:
    r = requests.get(
        url, timeout = timeout_sec, headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)",
        },
    )
    r.raise_for_status()
    html = r.text

    if HAS_BS4:
        soup = BeautifulSoup(html, "lxml")
        # remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ")
        return _clean_text(text)

    # fallback (rough)
    text = re.sub(r"<[^>]+>", " ", html)
    return _clean_text(text)


def web_retrieve_evidence(
        question: str,
        seed_urls: list[str],
        embeddings,
        max_search_results: int = 5,
        max_pages_fetch: int = 4,
        top_k_snippets: int = 6,
) -> list[Evidence]:
    urls: list[str] = []

    if seed_urls:
        urls = seed_urls[:max_pages_fetch]
    else:
        ddg = DuckDuckGoSearchResults(output_format = "list")
        results = ddg.invoke({ "query": question, "max_results": max_search_results })
        # results: list[{"title","link","snippet"}]
        urls = [r.get("link") for r in results if r.get("link")]
        urls = urls[:max_pages_fetch]

    page_texts = []
    for u in urls:
        try:
            txt = fetch_url_text(u)
            if len(txt) < 200:
                continue
            page_texts.append((u, txt))
        except Exception:
            continue

    if not page_texts:
        return []

    # Build an ephemeral FAISS for these pages to pick best chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 150)
    from langchain_core.documents import Document

    docs = []
    for url, txt in page_texts:
        chunks = splitter.split_text(txt)
        for i, ch in enumerate(chunks[:30]):  # limit per page
            docs.append(
                Document(
                    page_content = ch,
                    metadata = { "url": url, "chunk": i },
                ),
            )

    vs = FAISS.from_documents(docs, embeddings)
    pairs = vs.similarity_search_with_score(question, k = top_k_snippets)

    out: list[Evidence] = []
    for doc, score in pairs:
        url = doc.metadata.get("url", "unknown")
        chunk = doc.metadata.get("chunk", "?")
        out.append(
            Evidence(
                source_type = "web",
                source = url,
                locator = f"chunk={chunk}",
                title = None,
                snippet = _clean_text(doc.page_content)[:900],
                score = float(score),
            ),
        )
    return out


# ----------------------------
# LangGraph state
# ----------------------------

class RAGState(TypedDict):
    question: str
    seed_urls: list[str]

    # query variants (optional)
    queries: list[str]

    # evidence buckets
    web_evidence: Annotated[list[Evidence], operator.add]
    doc_evidence: Annotated[list[Evidence], operator.add]
    ctx_evidence: Annotated[list[Evidence], operator.add]

    final_answer: str
    errors: Annotated[list[str], operator.add]


# ----------------------------
# Nodes (agents)
# ----------------------------

def plan_node( state: RAGState, model ) -> dict:
    """
    Generates 2-5 query variants for better recall (web/docs/context).
    """
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
    text = resp.content if hasattr(resp, "content") else str(resp)
    try:
        queries = json.loads(text)
        if not isinstance(queries, list):
            queries = [q]
        queries = [str(x) for x in queries][:5]
    except Exception:
        queries = [q]
    return { "queries": queries }


def web_agent_node( state: RAGState, embeddings ) -> dict:
    """
    Agent #1: web search by links (seed_urls) or by query -> fetch -> select evidence.
    """
    question = state["question"]
    seed_urls = state.get("seed_urls", [])
    evidence = web_retrieve_evidence(
        question = question,
        seed_urls = seed_urls,
        embeddings = embeddings,
        max_search_results = int(os.getenv("WEB_SEARCH_K", "6")),
        max_pages_fetch = int(os.getenv("WEB_FETCH_PAGES", "4")),
        top_k_snippets = int(os.getenv("WEB_TOP_SNIPPETS", "6")),
    )
    return { "web_evidence": evidence }


def docs_agent_node( state: RAGState, docs_vs: FAISS ) -> dict:
    """
    Agent #2: search in documents (local vector store).
    """
    question = state["question"]
    k = int(os.getenv("DOC_TOP_K", "8"))
    pairs = docs_vs.similarity_search_with_score(question, k = k)

    out: list[Evidence] = []
    for doc, score in pairs:
        src = doc.metadata.get("source_file", "unknown_file")
        page = doc.metadata.get("page_human")
        locator = f"page={page}" if page else "chunk"
        out.append(
            Evidence(
                source_type = "doc",
                source = src,
                locator = locator,
                title = None,
                snippet = _clean_text(doc.page_content)[:900],
                score = float(score),
            ),
        )
    return { "doc_evidence": out }


def context_agent_node( state: RAGState, ctx_vs: Optional[FAISS] ) -> dict:
    """
    Agent #3: search in additional context (notes/configs/etc).
    If no context index exists -> returns empty.
    """
    if ctx_vs is None:
        return { "ctx_evidence": [] }

    question = state["question"]
    k = int(os.getenv("CTX_TOP_K", "6"))
    pairs = ctx_vs.similarity_search_with_score(question, k = k)

    out: list[Evidence] = []
    for doc, score in pairs:
        src = doc.metadata.get("source_file", "context")
        locator = doc.metadata.get("section", "chunk")
        out.append(
            Evidence(
                source_type = "context",
                source = src,
                locator = str(locator),
                title = None,
                snippet = _clean_text(doc.page_content)[:900],
                score = float(score),
            ),
        )
    return { "ctx_evidence": out }


def answer_agent_node( state: RAGState, model ) -> dict:
    """
    Agent #4: final grounded answer with citations.
    """
    q = state["question"]
    evidence = (state.get("web_evidence", []) +
                state.get("doc_evidence", []) +
                state.get("ctx_evidence", []))

    # Lightweight re-rank: prefer best scores per source_type
    # (FAISS score semantics can differ; treat as relative)
    evidence = sorted(evidence, key = lambda e: e.score)[:20]

    if not evidence:
        return { "final_answer": "В загруженных источниках ничего релевантного не найдено под этот вопрос." }

    # Build evidence pack with stable ids
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

    resp = model.invoke([SystemMessage(content = system), HumanMessage(content = user)])
    answer_text = resp.content if hasattr(resp, "content") else str(resp)

    # Append footnotes (map ids -> sources) if model forgot
    if "Источники:" not in answer_text:
        foot = ["\nИсточники:"]
        for i, e in enumerate(evidence, start = 1):
            foot.append(f"[{i}] {format_citation(e)}")
        answer_text = answer_text.rstrip() + "\n" + "\n".join(foot)

    return { "final_answer": answer_text }


# ----------------------------
# Build graph
# ----------------------------

def build_graph( docs_vs: FAISS, ctx_vs: Optional[FAISS], embeddings, model ):
    g = StateGraph(RAGState)

    g.add_node("plan", lambda s: plan_node(s, model))
    # g.add_node("web_agent", lambda s: web_agent_node(s, embeddings))
    g.add_node("docs_agent", lambda s: docs_agent_node(s, docs_vs))
    g.add_node("context_agent", lambda s: context_agent_node(s, ctx_vs))
    g.add_node("answer_agent", lambda s: answer_agent_node(s, model))

    # Simple sequential multi-agent pipeline (можно распараллелить позже)
    g.add_edge(START, "plan")
    g.add_edge("plan", "docs_agent")
    # g.add_edge("web_agent", "docs_agent")
    g.add_edge("docs_agent", "context_agent")
    g.add_edge("context_agent", "answer_agent")
    g.add_edge("answer_agent", END)

    return g.compile()


from pathlib import Path


def run():
    # ===== CONFIG (можешь просто менять значения тут) =====
    DOCS_DIR = Path("/Users/onbozoyan/PycharmProjects/agent-rag-diplom/data/docs")
    CONTEXT_DIR = Path("/Users/onbozoyan/PycharmProjects/agent-rag-diplom/data/context")  # опционально
    REINDEX = False

    QUESTION = "Какие задачи решает LLM встроенный в Web ?"
    SEED_URLS = [
        # "https://example.com/page1",
        # "https://example.com/page2",
    ]
    # =====================================================

    # Если хочешь именно "ввод через программу" — раскомментируй:
    # QUESTION = input("Question: ").strip()
    # urls_raw = input("Seed URLs (comma-separated, optional): ").strip()
    # SEED_URLS = [u.strip() for u in urls_raw.split(",") if u.strip()]

    embeddings = init_embeddings()

    docs_vs = build_or_load_index(DOCS_DIR, Path("./.index/chroma_docs"), embeddings, REINDEX)
    ctx_vs = build_or_load_index(
        CONTEXT_DIR, Path("./.index/chroma_ctx"), embeddings, REINDEX,
    ) if CONTEXT_DIR.exists() else None

    model_name = os.getenv("CHAT_MODEL", "openai:gpt-5-mini-2025-08-07")
    model = init_chat_model(model_name, temperature = float(os.getenv("TEMP", "0")))

    graph = build_graph(docs_vs = docs_vs, ctx_vs = ctx_vs, embeddings = embeddings, model = model)

    initial_state: RAGState = {
        "question": QUESTION,
        "seed_urls": SEED_URLS,
        "queries": [],
        "web_evidence": [],
        "doc_evidence": [],
        "ctx_evidence": [],
        "final_answer": "",
        "errors": [],
    }

    result = graph.invoke(initial_state)
    print(result["final_answer"])


if __name__ == "__main__":
    run()
