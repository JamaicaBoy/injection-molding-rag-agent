from src.app.streamlit_app import (
    MODES,
    evidence_rows,
    friendly_error,
    render_retrieval_stats,
    run_defect_diagnosis,
)


def test_streamlit_helpers_keep_evidence_safe_and_structured() -> None:
    rows = evidence_rows(
        [
            {
                "paper_id": "paper_1",
                "title": "Warpage study",
                "source_location": {"section": "Results"},
                "relevance_score": 0.81234,
                "matched_text": "Evidence preview",
                "file_path": "E:/private/paper.pdf",
            }
        ]
    )

    assert rows == [
        {
            "paper_id": "paper_1",
            "title": "Warpage study",
            "section": "Results",
            "score": 0.8123,
            "text_preview": "Evidence preview",
        }
    ]
    assert "E:/private" not in str(rows)


def test_streamlit_modes_exclude_method_compare_and_hide_local_errors() -> None:
    assert "方法对比" not in MODES
    message = friendly_error(FileNotFoundError("Chroma missing at E:/private/vector_store"))
    assert "E:/private" not in message
    assert "Chroma" in message


def test_retrieval_stats_hide_absolute_persist_path(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr("src.app.streamlit_app.st.subheader", lambda value: None)
    monkeypatch.setattr("src.app.streamlit_app.st.json", lambda value: captured.update(value))

    render_retrieval_stats(
        {
            "chunks_count": 1864,
            "chroma_persist_dir": "E:/private/vector_store/chroma",
            "collection_name": "injection_molding_chunks",
            "collection_count": 1864,
            "dense_results_count": 5,
            "bm25_results_count": 5,
            "hybrid_results_count": 5,
        },
        corpus_mode="dev",
    )

    assert captured["chroma_persist_dir"] == "vector_store/chroma"
    assert captured["collection_name"] == "injection_molding_chunks"
    assert captured["collection_count"] == 1864


def test_defect_mode_reuses_rag_generation_and_adds_safety_notice(monkeypatch) -> None:
    captured = {}

    def fake_run_normal_rag(**kwargs):
        captured.update(kwargs)
        return {
            "answer": "提高注射压力可能增加飞边风险。[E1]",
            "evidence_list": [{"evidence_id": "E1"}],
            "confidence": "medium",
            "need_human_review": False,
            "limitations": [],
        }

    monkeypatch.setattr("src.app.streamlit_app.run_normal_rag", fake_run_normal_rag)
    output = run_defect_diagnosis(
        question="飞边可能是什么原因？",
        top_k=5,
        rewrite={"normalized_query": "飞边 flash 原因"},
        corpus_mode="full",
        workflow_backend="langgraph",
        conversation_id="conversation_1",
    )

    assert captured["workflow_backend"] == "langgraph"
    assert captured["conversation_id"] == "conversation_1"
    assert output["answer"].startswith("提高注射压力")
    assert any("不构成直接生产调参指令" in item for item in output["limitations"])
