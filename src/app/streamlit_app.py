from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools import search_papers_tool
from src.agent.conversation_state import (
    ConversationState,
    resolve_followup_query,
)
from src.app.chat_ui import (
    ensure_active_chat,
    process_chat_turn,
    render_chat_messages,
    render_sidebar_history,
    save_chat_settings,
    select_chat,
    start_new_chat,
)
from src.agent.langgraph_workflow import LangGraphWorkflow
from src.agent.workflow import AgentWorkflow
from src.config import SUPPORTED_CORPUS_MODES, load_corpus_config
from src.index.incremental_index import (
    add_uploaded_chunks,
    clear_upload_collection,
    upload_collection_name,
    upload_collection_stats,
)
from src.index.index_lock import IndexLockError
from src.index.index_registry import get_index_record
from src.ingest.ingest_uploaded_paper import (
    ingest_uploaded_papers,
    save_uploaded_pdf,
)
from src.rag.answer_generator import AnswerGenerator, load_llm_settings, unload_ollama_model
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.multi_collection_retriever import MultiCollectionRetriever
from src.retrieval.retrieval_debug import (
    inspect_retrieval_state,
    retrieve_debug_results,
)
from src.retrieval.reranker import Reranker


APP_CONFIG = PROJECT_ROOT / "configs" / "app_config.yaml"

MODE_RAG = "普通 RAG"
MODE_DEFECT = "缺陷诊断"
MODE_DEBUG = "检索调试"
MODES = [MODE_RAG, MODE_DEFECT, MODE_DEBUG]


def evidence_rows(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        source = item.get("source_location") or {}
        score = item.get("rerank_score")
        if score is None:
            score = item.get("relevance_score", item.get("score", 0.0))
        rows.append(
            {
                "paper_id": str(item.get("paper_id", "")),
                "title": str(item.get("title", "")),
                "section": str(item.get("section_name") or source.get("section") or ""),
                "score": round(float(score or 0.0), 4),
                "text_preview": str(item.get("text_preview") or item.get("matched_text") or "")[:200],
            }
        )
    return rows


def friendly_error(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, IndexLockError) or "索引正在更新" in str(exc):
        return "索引正在更新，请稍后"
    if "chroma" in message or "collection" in message or "persist" in message:
        return "向量索引尚未准备好，请先构建并检查 Chroma collection。"
    if "sentence-transformers" in message or "embedding" in message or "model" in message:
        return "本地向量模型尚未准备好，请检查模型配置后重试。"
    if "ollama" in message or "connection" in message:
        return "本地 Ollama 暂不可用，普通 RAG 可切换到 Mock 结果或稍后重试。"
    if "chunk" in message or "jsonl" in message:
        return "论文 chunk 数据尚未准备好，请先完成 chunk 构建。"
    return "运行过程中出现本地组件错误，请检查模型、向量库和项目配置。"


@st.cache_resource(show_spinner=False)
def get_bm25(chunks_path: str, modified_ns: int) -> BM25Retriever:
    del modified_ns
    return BM25Retriever(Path(chunks_path))


@st.cache_resource(show_spinner=False)
def get_dense(persist_dir: str, collection_name: str, collection_id: str) -> DenseRetriever:
    del collection_id
    return DenseRetriever(
        persist_dir=Path(persist_dir),
        collection_name=collection_name,
        subprocess_encoder=True,
    )


@st.cache_resource(show_spinner=False)
def get_reranker() -> Reranker:
    return Reranker(mode="rule")


def index_last_build_time(persist_dir: Path, effective_mode: str) -> str:
    if effective_mode == "full":
        report_path = PROJECT_ROOT / "data" / "logs" / "full_index_report.md"
        if report_path.is_file():
            match = re.search(
                r"^- completed_at:\s*(.+)$",
                report_path.read_text(encoding="utf-8"),
                flags=re.MULTILINE,
            )
            if match:
                return match.group(1).strip()
    database = persist_dir / "chroma.sqlite3"
    if not database.is_file():
        return "未构建"
    return datetime.fromtimestamp(database.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def readiness(corpus_mode: str | None = None) -> dict[str, Any]:
    corpus = load_corpus_config(mode=corpus_mode)
    model_ready = False
    try:
        with APP_CONFIG.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        local_model_path = config.get("embedding", {}).get("local_model_path")
        model_ready = bool(local_model_path and Path(local_model_path).is_dir())
    except (OSError, yaml.YAMLError):
        model_ready = False

    ollama_ready = False
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.6) as response:
            ollama_ready = response.status == 200
    except (OSError, urllib.error.URLError):
        ollama_ready = False
    try:
        vector_state = inspect_retrieval_state(
            corpus.chunks_path,
            corpus.vector_persist_dir,
            corpus.collection_name,
        )
    except Exception:
        vector_state = {
            "chunks_count": 0,
            "chroma_persist_dir": str(corpus.vector_persist_dir),
            "collection_name": corpus.collection_name,
            "collection_count": 0,
            "paper_count": 0,
            "collection_id": "",
            "vector_store_ready": False,
        }
    registry_record = get_index_record(
        corpus_mode=corpus.effective_mode,
        collection_name=corpus.collection_name,
    )
    if registry_record:
        vector_state["registry_index_version"] = str(registry_record.get("version") or "")
        vector_state["registry_built_at"] = str(registry_record.get("built_at") or "")
    return {
        "corpus_mode": corpus.corpus_mode,
        "effective_mode": corpus.effective_mode,
        "fallback_mode": corpus.fallback_mode,
        "fallback_reason": corpus.fallback_reason,
        "chunks_path": corpus.chunks_path_label,
        "vector_persist_dir": corpus.vector_persist_dir_label,
        "legacy_fallback_used": corpus.legacy_fallback_used,
        "chunks": vector_state["chunks_count"] > 0,
        "vector_store": vector_state["vector_store_ready"],
        "embedding_model": model_ready,
        "ollama": ollama_ready,
        "index_version": str(vector_state.get("collection_id") or "未构建")[:12],
        "last_build_time": index_last_build_time(corpus.vector_persist_dir, corpus.effective_mode),
        **vector_state,
    }


def shared_search(
    *,
    corpus_mode: str | None = None,
    upload_session_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    corpus = load_corpus_config(mode=corpus_mode)
    search_type = kwargs.get("search_type", "hybrid")
    if search_type in {"keyword", "hybrid"}:
        modified_ns = corpus.chunks_path.stat().st_mtime_ns if corpus.chunks_path.exists() else 0
        kwargs["_bm25"] = get_bm25(str(corpus.chunks_path), modified_ns)
    if search_type in {"semantic", "hybrid"}:
        llm_model, llm_base_url, fallback_models, _, _, _ = load_llm_settings()
        for model in [llm_model, *fallback_models]:
            unload_ollama_model(model, llm_base_url)
        state = readiness(corpus_mode)
        if not state["vector_store"]:
            raise RuntimeError(f"Chroma collection is not ready: {state['collection_name']}")
        base_dense = get_dense(
            state["chroma_persist_dir"],
            state["collection_name"],
            state["collection_id"],
        )
        if upload_session_id:
            kwargs["_dense"] = MultiCollectionRetriever.from_chroma(
                persist_dir=corpus.vector_persist_dir,
                base_collection_name=corpus.collection_name,
                upload_collection_name=upload_collection_name(upload_session_id),
                base_retriever=base_dense,
                reranker=get_reranker(),
                rerank_results=False,
            )
        else:
            kwargs["_dense"] = base_dense
    kwargs["_reranker"] = get_reranker()
    return search_papers_tool(**kwargs)


def confidence_value(result: dict[str, Any]) -> str:
    confidence = result.get("confidence", "low")
    if isinstance(confidence, (int, float)):
        return f"{float(confidence):.3f}"
    return str(confidence)


class ConversationAwareAnswerGenerator:
    """Bind bounded chat context without changing either workflow core."""

    def __init__(
        self,
        *,
        conversation_id: str | None,
        recent_turns: list[dict[str, Any]] | None,
        conversation_summary: str | None,
    ) -> None:
        self.generator = AnswerGenerator(mode="ollama")
        self.conversation_id = conversation_id
        self.recent_turns = list(recent_turns or [])
        self.conversation_summary = conversation_summary
        self.context_debug: dict[str, Any] = {}
        self.active_mode = "ollama"
        self.active_model = ""
        self.fallback_reason: str | None = None

    def generate(
        self,
        question: str,
        query_rewrite: dict[str, Any],
        evidence_results: list[dict[str, Any]],
    ) -> Any:
        result = self.generator.generate(
            question,
            query_rewrite,
            evidence_results,
            conversation_history=self.recent_turns,
            conversation_summary=self.conversation_summary,
            conversation_id=self.conversation_id,
            recent_turns=self.recent_turns,
        )
        self.context_debug = dict(result.context_debug or {})
        self.active_mode = self.generator.active_mode
        self.active_model = self.generator.active_model
        self.fallback_reason = (
            self.generator.fallback_reason
            or self.generator.model_fallback_reason
        )
        return result


def run_normal_rag(
    question: str,
    top_k: int,
    corpus_mode: str,
    workflow_backend: str = "classic",
    rewrite: dict[str, Any] | None = None,
    conversation_id: str | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
    conversation_summary: str | None = None,
    upload_session_id: str | None = None,
) -> dict[str, Any]:
    retrieval_top_k = max(top_k, 5)
    # The workflow re-runs query rewriting internally (it only receives the raw
    # question string). To let conversation-aware follow-up resolution actually
    # drive retrieval, inject the already-resolved rewrite (which may have
    # filled in omitted referents like "那对缩水呢？" from recent turns) instead
    # of letting the workflow silently recompute a plain, history-blind rewrite.
    rewriter = (lambda _query: rewrite) if rewrite else None
    answer_generator = ConversationAwareAnswerGenerator(
        conversation_id=conversation_id,
        recent_turns=recent_turns,
        conversation_summary=conversation_summary,
    )
    if workflow_backend == "langgraph":
        langgraph_kwargs: dict[str, Any] = {
            "retriever": lambda query, limit: shared_search(
                corpus_mode=corpus_mode,
                upload_session_id=upload_session_id,
                query=query,
                search_type="hybrid",
                filters={},
                top_k=limit,
                rerank=False,
                return_chunks=True,
                language="auto",
            ),
            "reranker": get_reranker(),
            "answer_generator": answer_generator,
            "retrieval_top_k": retrieval_top_k,
        }
        if rewriter is not None:
            langgraph_kwargs["rewriter"] = rewriter
        workflow = LangGraphWorkflow(**langgraph_kwargs)
    else:
        workflow_kwargs: dict[str, Any] = {
            "retrieval_top_k": retrieval_top_k,
            "search_tool": lambda **kwargs: shared_search(
                corpus_mode=corpus_mode,
                upload_session_id=upload_session_id,
                **kwargs,
            ),
            "reranker": get_reranker(),
            "answer_generator": answer_generator,
        }
        if rewriter is not None:
            workflow_kwargs["rewriter"] = rewriter
        workflow = AgentWorkflow(**workflow_kwargs)
    output = (
        workflow.run(question, conversation_id=conversation_id)
        if workflow_backend == "langgraph"
        else workflow.run(question)
    )
    output["context_debug"] = answer_generator.context_debug
    output["llm_mode"] = answer_generator.active_mode
    output["llm_model"] = answer_generator.active_model
    output["llm_fallback_reason"] = answer_generator.fallback_reason
    trigger = (output.get("review_ticket") or {}).get("audit_log", {}).get("trigger_reason")
    human_review_reason = output.get("human_review_reason")
    reasons = [human_review_reason or trigger] if human_review_reason or trigger else []
    trace = list(output.get("trace", []))
    trace_summary = {
        "workflow_backend": workflow_backend,
        "retrieved_count": int(output.get("retrieved_count", 0)),
        "reranked_count": int(output.get("reranked_count", 0)),
        "top_score": round(float(output.get("top_score", 0.0) or 0.0), 6),
        "confidence": output.get("confidence", "low"),
        "confidence_reason": output.get("confidence_reason", ""),
        "llm_mode": answer_generator.active_mode,
        "llm_model": answer_generator.active_model,
        "llm_fallback_reason": answer_generator.fallback_reason or "",
        "human_review_reason": human_review_reason or "normal_answer",
        "executed_nodes": output.get("node_history", []),
    }
    debug = {
        "nodes": output.get("node_history", []),
        "tools": [item.get("tool") for item in output.get("tool_calls", [])],
    }
    if workflow_backend == "langgraph":
        debug.update({"trace_summary": trace_summary, "trace": trace})
    return {
        **output,
        "review_reasons": reasons,
        "debug": debug,
        "workflow_backend": workflow_backend,
    }


def run_defect_diagnosis(
    question: str,
    top_k: int,
    rewrite: dict[str, Any],
    corpus_mode: str,
    workflow_backend: str = "classic",
    conversation_id: str | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
    conversation_summary: str | None = None,
    upload_session_id: str | None = None,
) -> dict[str, Any]:
    result = run_normal_rag(
        question=question,
        top_k=top_k,
        corpus_mode=corpus_mode,
        workflow_backend=workflow_backend,
        rewrite=rewrite,
        conversation_id=conversation_id,
        recent_turns=recent_turns,
        conversation_summary=conversation_summary,
        upload_session_id=upload_session_id,
    )
    limitations = list(result.get("limitations") or [])
    diagnosis_notice = "缺陷诊断仅提供有论文证据支持的候选排查方向，不构成直接生产调参指令。"
    if diagnosis_notice not in limitations:
        limitations.append(diagnosis_notice)
    result["limitations"] = limitations
    return result


def run_retrieval_debug(
    question: str,
    top_k: int,
    corpus_mode: str,
    rewrite: dict[str, Any] | None = None,
    upload_session_id: str | None = None,
) -> dict[str, Any]:
    corpus = load_corpus_config(mode=corpus_mode)
    retrieval_query = (rewrite or {}).get("normalized_query") or question
    output = retrieve_debug_results(
        retrieval_query,
        chunks_path=corpus.chunks_path,
        persist_dir=corpus.vector_persist_dir,
        collection_name=corpus.collection_name,
        top_k=top_k,
        reranker=get_reranker(),
    )
    keyword_results = output["bm25_results"]
    semantic_results = output["dense_results"]
    hybrid_results = output["reranked_results"] or output["hybrid_results"]
    if upload_session_id:
        combined = shared_search(
            corpus_mode=corpus_mode,
            upload_session_id=upload_session_id,
            query=retrieval_query,
            search_type="hybrid",
            filters={},
            top_k=top_k,
            rerank=True,
            return_chunks=True,
            language="auto",
        )
        hybrid_results = combined.get("results", hybrid_results)
    return {
        "answer": f"检索完成：BM25 {len(keyword_results)} 条，Dense {len(semantic_results)} 条，Hybrid {len(hybrid_results)} 条。",
        "evidence_list": hybrid_results,
        "confidence": max(
            (float(item.get("rerank_score", item.get("score", 0.0))) for item in hybrid_results),
            default=0.0,
        ),
        "need_human_review": False,
        "limitations": [],
        "review_reasons": [],
        "debug": {
            "BM25": {"results": keyword_results},
            "Dense": {"results": semantic_results},
            "Hybrid": {"results": hybrid_results},
        },
        "debug_stats": output["debug_stats"],
    }


def execute_mode(
    mode: str,
    question: str,
    top_k: int,
    corpus_mode: str,
    workflow_backend: str = "classic",
    conversation: ConversationState | None = None,
    conversation_id: str | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
    conversation_summary: str | None = None,
    upload_session_id: str | None = None,
) -> dict[str, Any]:
    if conversation is not None and conversation.turns:
        rewritten = resolve_followup_query(question, conversation).to_dict()
    else:
        rewritten = rewrite_query(question).to_dict()
    if mode == MODE_RAG:
        result = run_normal_rag(
            question,
            top_k,
            corpus_mode,
            workflow_backend,
            rewritten,
            conversation_id,
            recent_turns,
            conversation_summary,
            upload_session_id,
        )
    elif mode == MODE_DEFECT:
        result = run_defect_diagnosis(
            question,
            top_k,
            rewritten,
            corpus_mode,
            workflow_backend,
            conversation_id,
            recent_turns,
            conversation_summary,
            upload_session_id,
        )
    elif mode == MODE_DEBUG:
        result = run_retrieval_debug(
            question, top_k, corpus_mode, rewritten, upload_session_id
        )
    else:
        raise ValueError(f"Unsupported Streamlit mode: {mode}")
    result["query_rewrite"] = rewritten
    result["mode"] = mode
    result["corpus_mode"] = corpus_mode
    result["workflow_backend"] = workflow_backend
    return result


def write_startup_report(status: dict[str, Any]) -> Path:
    report_path = PROJECT_ROOT / "data" / "logs" / "app_startup_report.md"
    lines = [
        "# App Startup Report",
        "",
        f"- checked_at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- requested_corpus_mode: {status['corpus_mode']}",
        f"- effective_corpus_mode: {status['effective_mode']}",
        f"- fallback_mode: {status.get('fallback_mode') or 'none'}",
        f"- chunks_path: {status['chunks_path']}",
        f"- vector_persist_dir: {status['vector_persist_dir']}",
        f"- collection_name: {status['collection_name']}",
        f"- paper_count: {status['paper_count']}",
        f"- chunk_count: {status['chunks_count']}",
        f"- vector_count: {status['collection_count']}",
        f"- index_version: {status['index_version']}",
        f"- last_build_time: {status['last_build_time']}",
        f"- vector_store_ready: {str(status['vector_store']).lower()}",
        "",
    ]
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError:
        pass
    return report_path


def render_readiness(corpus_mode: str) -> dict[str, Any]:
    status = readiness(corpus_mode)
    if status.get("registry_index_version"):
        status["index_version"] = status["registry_index_version"]
    if status.get("registry_built_at"):
        status["last_build_time"] = status["registry_built_at"]
    st.sidebar.subheader("文献库")
    st.sidebar.caption(f"当前模式: `{status['effective_mode']}`")
    st.sidebar.caption(f"chunks_path: `{status['chunks_path']}`")
    st.sidebar.caption(f"collection_name: `{status['collection_name']}`")
    st.sidebar.caption(f"论文数量: `{status['paper_count']}`")
    st.sidebar.caption(f"chunk 数量: `{status['chunks_count']}`")
    st.sidebar.caption(f"向量数量: `{status['collection_count']}`")
    st.sidebar.caption(f"索引版本: `{status['index_version']}`")
    st.sidebar.caption(f"最后构建时间: `{status['last_build_time']}`")
    if status["fallback_mode"]:
        st.sidebar.warning(
            f"`{status['corpus_mode']}` 不可用，已自动降级到 `{status['fallback_mode']}`。"
        )
    if status["legacy_fallback_used"]:
        st.sidebar.caption("当前 dev 模式使用已验证的 legacy baseline。")
    if corpus_mode == "full" and not status["vector_store"]:
        st.sidebar.warning("请先运行 full ingest 和 full index 命令")
    st.sidebar.subheader("本地组件")
    for label, key in (
        ("Chunk 数据", "chunks"),
        ("Chroma 索引", "vector_store"),
        ("Embedding 模型", "embedding_model"),
        ("Ollama", "ollama"),
    ):
        if status[key]:
            st.sidebar.success(f"{label}：就绪")
        else:
            st.sidebar.warning(f"{label}：未就绪")
    if not status["ollama"]:
        st.sidebar.caption("Ollama 不可用时，普通 RAG 会使用 Mock 结果。")
    return status


def render_evidence_table(evidence: list[dict[str, Any]]) -> None:
    rows = evidence_rows(evidence)
    st.subheader("引用证据")
    if not rows:
        st.info("当前没有可展示的引用证据。")
        return
    frame = pd.DataFrame(rows)
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "paper_id": st.column_config.TextColumn("paper_id", width="medium"),
            "title": st.column_config.TextColumn("title", width="large"),
            "section": st.column_config.TextColumn("section", width="small"),
            "score": st.column_config.NumberColumn("score", format="%.4f", width="small"),
            "text_preview": st.column_config.TextColumn("text_preview", width="large"),
        },
    )


def render_debug(debug: dict[str, Any]) -> None:
    if all(key in debug for key in ("BM25", "Dense", "Hybrid")):
        tabs = st.tabs(["BM25", "Dense", "Hybrid"])
        for tab, key in zip(tabs, ("BM25", "Dense", "Hybrid")):
            with tab:
                rows = evidence_rows(debug[key].get("results", []))
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        safe_debug = {
            "nodes": debug.get("nodes", []),
            "tools": debug.get("tools", []),
            "methods": debug.get("methods", []),
        }
        st.json(safe_debug)
        trace_summary = debug.get("trace_summary") or {}
        if trace_summary:
            st.subheader("LangGraph trace summary")
            st.json(trace_summary)
        trace = list(debug.get("trace") or [])
        if trace:
            rows = [
                {
                    "node_name": item.get("node_name", ""),
                    "retrieved_count": item.get("retrieved_count", 0),
                    "reranked_count": item.get("reranked_count", 0),
                    "top_score": item.get("top_score", 0.0),
                    "confidence_before": item.get("confidence_before", ""),
                    "confidence_after": item.get("confidence_after", ""),
                    "confidence_reason": item.get("confidence_reason", ""),
                    "need_human_review": item.get("need_human_review", False),
                    "human_review_reason": (item.get("output_summary") or {}).get(
                        "human_review_reason", ""
                    ),
                    "error": item.get("error", ""),
                }
                for item in trace
            ]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_retrieval_stats(stats: dict[str, Any], corpus_mode: str | None = None) -> None:
    corpus = load_corpus_config(mode=corpus_mode)
    safe_stats = {
        "chunks_count": int(stats.get("chunks_count", 0)),
        "chroma_persist_dir": corpus.vector_persist_dir_label,
        "collection_name": str(stats.get("collection_name", "")),
        "collection_count": int(stats.get("collection_count", 0)),
        "dense_results_count": int(stats.get("dense_results_count", 0)),
        "bm25_results_count": int(stats.get("bm25_results_count", 0)),
        "hybrid_results_count": int(stats.get("hybrid_results_count", 0)),
    }
    st.subheader("检索调试信息")
    st.json(safe_stats)


def legacy_form_main() -> None:
    st.set_page_config(page_title="注塑论文 RAG Agent", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1380px; padding-top: 1.4rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.28); border-radius: 6px; padding: .7rem .9rem;}
        .stButton > button {border-radius: 6px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("注塑论文 RAG Agent")

    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = new_conversation_id()
    conversation = ensure_current_conversation_state(
        st.session_state.get("conversation_state"),
        conversation_id=st.session_state["conversation_id"],
    )
    st.session_state["conversation_state"] = conversation

    default_corpus_mode = load_corpus_config().corpus_mode
    with st.sidebar:
        st.header("运行设置")
        corpus_mode = st.selectbox(
            "Corpus mode",
            options=list(SUPPORTED_CORPUS_MODES),
            index=list(SUPPORTED_CORPUS_MODES).index(default_corpus_mode),
        )
        workflow_backend = st.segmented_control(
            "Workflow backend",
            options=["classic", "langgraph"],
            default="classic",
            width="stretch",
        ) or "classic"
        top_k = st.slider("证据数量", min_value=3, max_value=10, value=5, step=1)
        show_rewrite = st.toggle("显示 Query Rewrite", value=False)
        show_debug = st.toggle("显示运行详情", value=False)
        show_summary = st.toggle("显示对话摘要", value=False)
        st.divider()
        st.subheader("对话记忆")
        st.caption(f"当前会话轮数：{len(conversation.turns)} / {conversation.max_turns}")
        if conversation.summary is not None:
            st.caption("早期对话已压缩为摘要，近期轮次保留完整内容。")
        if st.button("清空当前对话记忆", width="stretch"):
            conversation.clear()
            st.session_state.pop("last_result", None)
            st.rerun()
        if show_summary:
            with st.expander("对话摘要", expanded=True):
                summary_dict = conversation.summary_dict()
                if summary_dict is None:
                    st.caption("对话轮数或长度尚未达到摘要触发条件。")
                else:
                    st.json(summary_dict)
        st.divider()
    previous_corpus_mode = st.session_state.get("active_corpus_mode")
    if previous_corpus_mode != corpus_mode:
        st.session_state.pop("last_result", None)
        st.session_state["active_corpus_mode"] = corpus_mode
    previous_workflow_backend = st.session_state.get("active_workflow_backend")
    if previous_workflow_backend != workflow_backend:
        st.session_state.pop("last_result", None)
        st.session_state["active_workflow_backend"] = workflow_backend
    startup_status = render_readiness(corpus_mode)
    write_startup_report(startup_status)

    mode = st.segmented_control("模式", MODES, default=MODE_RAG, width="stretch") or MODE_RAG
    with st.form("question_form", border=False):
        question = st.text_area(
            "用户问题",
            height=110,
            placeholder="例如：保压压力对缩水有什么影响？",
        )
        submitted = st.form_submit_button("运行", type="primary", width="stretch")

    if submitted:
        if not question.strip():
            st.warning("请输入问题后再运行。")
        elif not startup_status["vector_store"]:
            if corpus_mode == "full":
                st.warning("请先运行 full ingest 和 full index 命令")
            else:
                st.warning("当前 corpus mode 的向量索引尚未准备好。")
        else:
            try:
                with st.spinner("正在检索论文证据并生成结果..."):
                    result = execute_mode(
                        mode,
                        question.strip(),
                        top_k,
                        corpus_mode,
                        workflow_backend,
                        conversation=conversation,
                    )
                    st.session_state["last_result"] = result
                    review_reasons = result.get("review_reasons") or result.get("limitations") or []
                    conversation.add_turn(
                        question.strip(),
                        result.get("answer") or "",
                        evidence_list=result.get("evidence_list"),
                        rewrite=result.get("query_rewrite"),
                        need_human_review=bool(result.get("need_human_review")),
                        review_reason="；".join(str(reason) for reason in review_reasons if reason),
                    )
            except Exception as exc:
                st.session_state.pop("last_result", None)
                st.error(friendly_error(exc))

    result = st.session_state.get("last_result")
    if not result:
        return

    st.caption(
        f"当前结果模式：{result.get('mode', '')} · workflow_backend={result.get('workflow_backend', 'classic')}"
    )
    metric_confidence, metric_review, metric_evidence = st.columns(3)
    metric_confidence.metric("Confidence", confidence_value(result))
    metric_review.metric("Human Review", "需要" if result.get("need_human_review") else "不需要")
    metric_evidence.metric("Evidence", len(result.get("evidence_list", [])))

    if result.get("need_human_review"):
        reasons = result.get("review_reasons") or result.get("limitations") or ["当前证据或风险等级需要人工确认。"]
        st.warning("需要人工复核：" + "；".join(str(reason) for reason in reasons if reason))

    st.subheader("最终答案")
    st.markdown(result.get("answer") or "当前论文库证据不足。")
    if result.get("mode") == MODE_DEBUG:
        render_retrieval_stats(result.get("debug_stats", {}), result.get("corpus_mode"))
    render_evidence_table(result.get("evidence_list", []))

    if show_rewrite:
        with st.expander("Query Rewrite", expanded=True):
            st.json(result.get("query_rewrite", {}))
    if show_debug:
        with st.expander("运行详情", expanded=True):
            render_debug(result.get("debug", {}))


def main() -> None:
    st.set_page_config(page_title="注塑论文 RAG Agent", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 6rem;}
        div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.28); border-radius: 6px; padding: .55rem .75rem;}
        .stButton > button {border-radius: 6px;}
        [data-testid="stChatMessage"] {padding: .75rem 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    default_corpus_mode = load_corpus_config().corpus_mode
    chat = ensure_active_chat(
        st.session_state,
        default_mode=MODE_RAG,
        default_corpus_mode=default_corpus_mode,
    )
    upload_session_id = st.session_state.setdefault(
        "upload_session_id", uuid.uuid4().hex[:16]
    )

    with st.sidebar:
        st.header("聊天")
        if st.button("新建聊天", type="primary", width="stretch"):
            start_new_chat(
                st.session_state,
                mode=MODE_RAG,
                corpus_mode=default_corpus_mode,
            )
            st.rerun()
        selected_chat_id = render_sidebar_history(str(chat["conversation_id"]))
        if selected_chat_id and selected_chat_id != chat["conversation_id"]:
            select_chat(st.session_state, selected_chat_id)
            st.rerun()

        st.divider()
        st.header("运行设置")
        configured_corpus = str(chat.get("corpus_mode") or default_corpus_mode)
        if configured_corpus not in SUPPORTED_CORPUS_MODES:
            configured_corpus = default_corpus_mode
        corpus_mode = st.selectbox(
            "Corpus mode",
            options=list(SUPPORTED_CORPUS_MODES),
            index=list(SUPPORTED_CORPUS_MODES).index(configured_corpus),
            key=f"corpus_mode_{chat['conversation_id']}",
        )
        workflow_backend = st.segmented_control(
            "Workflow backend",
            options=["classic", "langgraph"],
            default="classic",
            key=f"workflow_backend_{chat['conversation_id']}",
            width="stretch",
        ) or "classic"
        top_k = st.slider(
            "证据数量",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key=f"top_k_{chat['conversation_id']}",
        )
        show_rewrite = st.toggle(
            "显示 Query Rewrite",
            value=False,
            key=f"show_rewrite_{chat['conversation_id']}",
        )
        show_debug = st.toggle(
            "显示运行详情",
            value=False,
            key=f"show_debug_{chat['conversation_id']}",
        )
        show_summary = st.toggle(
            "显示对话摘要",
            value=False,
            key=f"show_summary_{chat['conversation_id']}",
        )
        if show_summary:
            if chat.get("summary"):
                st.json(chat["summary"])
            else:
                st.caption("当前聊天尚未达到摘要触发条件。")
        st.divider()

        st.subheader("上传新论文")
        st.caption(f"会话上传索引：`{upload_collection_name(upload_session_id)}`")
        uploaded_files = st.file_uploader(
            "选择 PDF",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_upload_{upload_session_id}",
        )
        active_corpus = load_corpus_config(mode=corpus_mode)
        if st.button(
            "处理并更新知识库",
            width="stretch",
            disabled=not uploaded_files,
            key=f"process_uploads_{upload_session_id}",
        ):
            try:
                with st.status("正在处理上传论文...", expanded=True) as upload_status_ui:
                    saved_paths = []
                    for uploaded_file in uploaded_files or []:
                        saved_paths.append(
                            save_uploaded_pdf(
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                session_id=upload_session_id,
                            )
                        )
                    upload_status_ui.write(f"已上传：{len(saved_paths)} 篇")
                    upload_status_ui.write("解析中：parse → clean → chunk")
                    ingest_stats = ingest_uploaded_papers(
                        upload_session_id,
                        pdf_paths=saved_paths,
                    )
                    upload_status_ui.write(
                        f"已生成 chunk：{ingest_stats.chunk_count} 条"
                    )
                    index_stats = add_uploaded_chunks(
                        chunks_path=Path(ingest_stats.chunks_path),
                        persist_dir=active_corpus.vector_persist_dir,
                        session_id=upload_session_id,
                        base_collection_name=active_corpus.collection_name,
                        batch_size=4,
                    )
                    st.session_state["upload_status"] = {
                        "state": "索引完成",
                        "uploaded_count": ingest_stats.uploaded_count,
                        "parsed_count": ingest_stats.parsed_count,
                        "failed_count": ingest_stats.failed_count,
                        "chunk_count": ingest_stats.chunk_count,
                        "vector_count": index_stats.get("collection_count", 0),
                        "collection_name": index_stats.get("upload_collection_name", ""),
                        "failures": ingest_stats.failures,
                    }
                    upload_status_ui.update(label="上传知识库索引完成", state="complete")
            except Exception as exc:
                st.session_state["upload_status"] = {
                    "state": "失败",
                    "failure_reason": friendly_error(exc)
                    if isinstance(exc, IndexLockError)
                    else f"{type(exc).__name__}: {str(exc)[:300]}",
                }

        upload_index_state = upload_collection_stats(
            persist_dir=active_corpus.vector_persist_dir,
            session_id=upload_session_id,
        )
        current_upload_status = st.session_state.get("upload_status") or {}
        if current_upload_status.get("state") == "索引完成":
            st.success(
                f"索引完成：{current_upload_status.get('chunk_count', 0)} chunks，"
                f"{upload_index_state['vector_count']} vectors"
            )
            if current_upload_status.get("failed_count"):
                st.warning(
                    f"失败文件：{current_upload_status['failed_count']} 篇；"
                    "详情见上传状态文件。"
                )
        elif current_upload_status.get("state") == "失败":
            st.error(f"处理失败：{current_upload_status.get('failure_reason', 'unknown error')}")
        else:
            st.caption(f"当前上传向量数：{upload_index_state['vector_count']}")

        if st.button(
            "清空本次上传文献索引",
            width="stretch",
            disabled=not upload_index_state["exists"],
            key=f"clear_uploads_{upload_session_id}",
        ):
            try:
                clear_upload_collection(
                    persist_dir=active_corpus.vector_persist_dir,
                    session_id=upload_session_id,
                )
                st.session_state["upload_status"] = {
                    "state": "已清空",
                    "chunk_count": 0,
                    "vector_count": 0,
                }
                st.rerun()
            except IndexLockError:
                st.warning("索引正在更新，请稍后")
        st.divider()

    startup_status = render_readiness(corpus_mode)
    write_startup_report(startup_status)

    st.title("注塑论文 RAG Agent")
    st.caption(f"{chat.get('title', '新聊天')} · conversation_id={chat['conversation_id'][:12]}")
    configured_mode = str(chat.get("mode") or MODE_RAG)
    if configured_mode not in MODES:
        configured_mode = MODE_RAG
    mode = st.segmented_control(
        "模式",
        MODES,
        default=configured_mode,
        key=f"answer_mode_{chat['conversation_id']}",
        width="stretch",
    ) or MODE_RAG
    chat = save_chat_settings(chat, mode=mode, corpus_mode=corpus_mode)

    render_chat_messages(
        chat,
        show_rewrite=show_rewrite,
        show_debug=show_debug,
    )

    prompt = st.chat_input("继续提问注塑缺陷、工艺参数、材料或质量预测...")
    if prompt:
        if not startup_status["vector_store"]:
            if corpus_mode == "full":
                st.warning("请先运行 full ingest 和 full index 命令。")
            else:
                st.warning("当前 corpus mode 的向量索引尚未准备好。")
            return
        try:
            with st.status("正在检索论文证据并生成回答...", expanded=False):
                result, chat = process_chat_turn(
                    execute_mode,
                    chat=chat,
                    question=prompt.strip(),
                    top_k=top_k,
                    corpus_mode=corpus_mode,
                    mode=mode,
                    workflow_backend=workflow_backend,
                    upload_session_id=upload_session_id,
                )
                st.session_state["last_result"] = result
            st.rerun()
        except Exception as exc:
            st.error(friendly_error(exc))


if __name__ == "__main__":
    main()
