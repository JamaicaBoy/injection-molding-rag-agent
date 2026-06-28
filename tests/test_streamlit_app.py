from src.app.streamlit_app import evidence_rows, extract_method_names, friendly_error, render_retrieval_stats


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


def test_streamlit_helpers_detect_methods_and_hide_local_errors() -> None:
    assert extract_method_names("对比 GA、PSO 和随机森林") == ["GA", "PSO", "random forest"]
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
        }
    )

    assert captured["chroma_persist_dir"] == "vector_store/chroma"
    assert captured["collection_name"] == "injection_molding_chunks"
    assert captured["collection_count"] == 1864
