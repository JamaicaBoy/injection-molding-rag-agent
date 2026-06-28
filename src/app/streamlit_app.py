from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools import defect_diagnosis_tool, method_compare_tool, search_papers_tool
from src.agent.workflow import AgentWorkflow
from src.rag.answer_generator import AnswerGenerator
from src.retrieval.bm25_retriever import BM25Retriever, DEFAULT_CHUNKS
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.query_rewrite import rewrite_query
from src.retrieval.retrieval_debug import (
    inspect_retrieval_state,
    load_vector_store_settings,
    retrieve_debug_results,
)
from src.retrieval.reranker import Reranker


APP_CONFIG = PROJECT_ROOT / "configs" / "app_config.yaml"

MODE_RAG = "普通 RAG"
MODE_DEFECT = "缺陷诊断"
MODE_METHOD = "方法对比"
MODE_DEBUG = "检索调试"
MODES = [MODE_RAG, MODE_DEFECT, MODE_METHOD, MODE_DEBUG]

KNOWN_METHODS = {
    "GA": (r"\bga\b", "遗传算法"),
    "PSO": (r"\bpso\b", "粒子群"),
    "machine learning": ("机器学习", r"\bmachine learning\b"),
    "deep learning": ("深度学习", r"\bdeep learning\b"),
    "random forest": ("随机森林", r"\brandom forest\b"),
    "neural network": ("神经网络", r"\bneural network\b"),
    "DOE": (r"\bdoe\b", "试验设计", "实验设计"),
    "response surface": ("响应面", r"\bresponse surface\b"),
    "Moldflow": ("moldflow",),
}


def extract_method_names(query: str) -> list[str]:
    names: list[str] = []
    for name, patterns in KNOWN_METHODS.items():
        if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns):
            names.append(name)
    return names or ["machine learning"]


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
def get_bm25() -> BM25Retriever:
    return BM25Retriever()


@st.cache_resource(show_spinner=False)
def get_dense(persist_dir: str, collection_name: str, collection_id: str) -> DenseRetriever:
    del collection_id
    return DenseRetriever(persist_dir=Path(persist_dir), collection_name=collection_name)


@st.cache_resource(show_spinner=False)
def get_reranker() -> Reranker:
    return Reranker(mode="rule")


def readiness() -> dict[str, Any]:
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
    persist_dir, collection_name = load_vector_store_settings()
    try:
        vector_state = inspect_retrieval_state(DEFAULT_CHUNKS, persist_dir, collection_name)
    except Exception:
        vector_state = {
            "chunks_count": 0,
            "chroma_persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "collection_count": 0,
            "collection_id": "",
            "vector_store_ready": False,
        }
    return {
        "chunks": vector_state["chunks_count"] > 0,
        "vector_store": vector_state["vector_store_ready"],
        "embedding_model": model_ready,
        "ollama": ollama_ready,
        **vector_state,
    }


def shared_search(**kwargs: Any) -> dict[str, Any]:
    search_type = kwargs.get("search_type", "hybrid")
    if search_type in {"keyword", "hybrid"}:
        kwargs["_bm25"] = get_bm25()
    if search_type in {"semantic", "hybrid"}:
        state = readiness()
        if not state["vector_store"]:
            raise RuntimeError(f"Chroma collection is not ready: {state['collection_name']}")
        kwargs["_dense"] = get_dense(
            state["chroma_persist_dir"],
            state["collection_name"],
            state["collection_id"],
        )
    kwargs["_reranker"] = get_reranker()
    return search_papers_tool(**kwargs)


def confidence_value(result: dict[str, Any]) -> str:
    confidence = result.get("confidence", "low")
    if isinstance(confidence, (int, float)):
        return f"{float(confidence):.3f}"
    return str(confidence)


def run_normal_rag(question: str, top_k: int) -> dict[str, Any]:
    workflow = AgentWorkflow(
        retrieval_top_k=max(top_k, 5),
        search_tool=shared_search,
        reranker=get_reranker(),
        answer_generator=AnswerGenerator(mode="ollama"),
    )
    output = workflow.run(question)
    trigger = (output.get("review_ticket") or {}).get("audit_log", {}).get("trigger_reason")
    reasons = [trigger] if trigger else []
    return {
        **output,
        "review_reasons": reasons,
        "debug": {
            "nodes": output.get("node_history", []),
            "tools": [item.get("tool") for item in output.get("tool_calls", [])],
        },
    }


def run_defect_diagnosis(question: str, top_k: int, rewrite: dict[str, Any]) -> dict[str, Any]:
    search = shared_search(query=question, search_type="hybrid", filters={}, top_k=top_k, rerank=True)
    defect_type = (rewrite.get("defect_type") or [None])[0]
    material = (rewrite.get("material") or [None])[0]
    diagnosis = defect_diagnosis_tool(
        defect_description=question,
        defect_type=defect_type,
        product_context={"material": material},
        retrieval_top_k=top_k,
        risk_level=rewrite.get("risk_level", "medium"),
        _search_fn=lambda **_: search,
    )
    causes = diagnosis.get("possible_causes", [])
    if causes:
        answer_lines = ["根据当前论文证据，可优先核查以下候选原因："]
        for cause in causes[:5]:
            refs = "".join(f"[{item}]" for item in cause.get("supporting_evidence_ids", []))
            answer_lines.append(f"- {cause.get('cause_description', '')[:260]} {refs}")
        answer_lines.append(diagnosis["not_final_decision_notice"])
        answer = "\n".join(answer_lines)
    else:
        answer = "当前论文库证据不足，暂时无法形成可靠的缺陷候选原因。"
    confidence = max((item.get("confidence_score", 0.0) for item in causes), default=0.0)
    return {
        "answer": answer,
        "evidence_list": search.get("results", []),
        "confidence": confidence,
        "need_human_review": diagnosis.get("need_human_review", False),
        "limitations": ["缺陷诊断仅提供候选排查方向，不构成生产指令。"],
        "review_reasons": ["证据覆盖不足、现场上下文缺失或问题风险较高。"] if diagnosis.get("need_human_review") else [],
        "debug": {"diagnosis": diagnosis},
    }


def run_method_compare(question: str, top_k: int) -> dict[str, Any]:
    methods = extract_method_names(question)
    search = shared_search(query=question, search_type="hybrid", filters={}, top_k=top_k, rerank=True)
    comparison = method_compare_tool(
        methods=[{"method_name": name} for name in methods],
        application_context={"task_type": "parameter_optimization", "available_data": []},
        retrieval_top_k=top_k,
        _search_fn=lambda **_: search,
    )
    lines = ["当前论文证据支持以下定性对比："]
    for row in comparison.get("comparison_table", []):
        refs = "".join(f"[{item}]" for item in row.get("supporting_evidence_ids", []))
        lines.append(f"- **{row['method_name']}**：{row.get('core_idea', '')[:260]} {refs}")
    lines.append(comparison["recommendation_for_context"]["not_decision_notice"])
    confidences = [row.get("confidence_score", 0.0) for row in comparison.get("comparison_table", [])]
    return {
        "answer": "\n".join(lines),
        "evidence_list": search.get("results", []),
        "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "need_human_review": comparison.get("need_human_review", False),
        "limitations": comparison.get("evidence_conflicts", []),
        "review_reasons": ["方法证据不足或真实技术选型需要专家确认。"] if comparison.get("need_human_review") else [],
        "debug": {"methods": methods, "comparison": comparison},
    }


def run_retrieval_debug(question: str, top_k: int) -> dict[str, Any]:
    persist_dir, collection_name = load_vector_store_settings()
    output = retrieve_debug_results(
        question,
        chunks_path=DEFAULT_CHUNKS,
        persist_dir=persist_dir,
        collection_name=collection_name,
        top_k=top_k,
        reranker=get_reranker(),
    )
    keyword_results = output["bm25_results"]
    semantic_results = output["dense_results"]
    hybrid_results = output["reranked_results"] or output["hybrid_results"]
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


def execute_mode(mode: str, question: str, top_k: int) -> dict[str, Any]:
    rewritten = rewrite_query(question).to_dict()
    if mode == MODE_RAG:
        result = run_normal_rag(question, top_k)
    elif mode == MODE_DEFECT:
        result = run_defect_diagnosis(question, top_k, rewritten)
    elif mode == MODE_METHOD:
        result = run_method_compare(question, top_k)
    else:
        result = run_retrieval_debug(question, top_k)
    result["query_rewrite"] = rewritten
    result["mode"] = mode
    return result


def render_readiness() -> None:
    status = readiness()
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


def render_retrieval_stats(stats: dict[str, Any]) -> None:
    safe_stats = {
        "chunks_count": int(stats.get("chunks_count", 0)),
        "chroma_persist_dir": "vector_store/chroma",
        "collection_name": str(stats.get("collection_name", "")),
        "collection_count": int(stats.get("collection_count", 0)),
        "dense_results_count": int(stats.get("dense_results_count", 0)),
        "bm25_results_count": int(stats.get("bm25_results_count", 0)),
        "hybrid_results_count": int(stats.get("hybrid_results_count", 0)),
    }
    st.subheader("检索调试信息")
    st.json(safe_stats)


def main() -> None:
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

    with st.sidebar:
        st.header("运行设置")
        top_k = st.slider("证据数量", min_value=3, max_value=10, value=5, step=1)
        show_rewrite = st.toggle("显示 Query Rewrite", value=False)
        show_debug = st.toggle("显示运行详情", value=False)
        st.divider()
    render_readiness()

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
        else:
            try:
                with st.spinner("正在检索论文证据并生成结果..."):
                    st.session_state["last_result"] = execute_mode(mode, question.strip(), top_k)
            except Exception as exc:
                st.session_state.pop("last_result", None)
                st.error(friendly_error(exc))

    result = st.session_state.get("last_result")
    if not result:
        return

    st.caption(f"当前结果模式：{result.get('mode', '')}")
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
        render_retrieval_stats(result.get("debug_stats", {}))
    render_evidence_table(result.get("evidence_list", []))

    if show_rewrite:
        with st.expander("Query Rewrite", expanded=True):
            st.json(result.get("query_rewrite", {}))
    if show_debug:
        with st.expander("运行详情", expanded=True):
            render_debug(result.get("debug", {}))


if __name__ == "__main__":
    main()
