# =========================
# LangGraph state & nodes
# =========================
import json
import operator
from typing import Optional

from langchain.messages import HumanMessage, SystemMessage
from langchain_postgres.v2.vectorstores import PGVectorStore
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

from app_config import AppConfig
from embedding import Evidence, _clean_text
from logger_config import logger
from timer import timed


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

    logger.info("Docs evidence=%d", out)
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

    logger.info("Context evidence=%d", out)
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
    graph = StateGraph(RAGState)

    graph.add_node("plan", lambda s: plan_node(s, model))
    graph.add_node("docs_agent", lambda s: docs_agent_node(s, docs_vs, cfg.doc_top_k))
    graph.add_node("context_agent", lambda s: context_agent_node(s, ctx_vs, cfg.ctx_top_k))
    graph.add_node("answer_agent", lambda s: answer_agent_node(s, model))

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "docs_agent")
    graph.add_edge("docs_agent", "context_agent")
    graph.add_edge("context_agent", "answer_agent")
    graph.add_edge("answer_agent", END)
    return graph.compile()
