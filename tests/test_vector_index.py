import json
from pathlib import Path

from src.index.build_vector_index import REQUIRED_METADATA_FIELDS, build_index, chroma_metadata
from src.index.inspect_index import inspect_index


class FakeEmbeddingModel:
    def __init__(self, model_name: str = "fake-local-embedding") -> None:
        self.model_name = model_name

    def encode_texts(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return [[float(len(text)), float(index), 1.0] for index, text in enumerate(texts)]


def write_chunks(path: Path) -> None:
    rows = [
        {
            "chunk_id": f"chunk_{index}",
            "paper_id": "paper_1",
            "file_name": "Smith - 2025 - Test.pdf",
            "title": "Test",
            "year": "2025",
            "section_name": "Abstract",
            "chunk_type": "text",
            "text": f"Chunk text {index}",
            "char_count": 12,
            "token_estimate": 4,
            "page_start": 1,
            "page_end": 1,
            "metadata": {},
        }
        for index in range(4)
    ]
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_chroma_metadata_has_required_fields() -> None:
    metadata = chroma_metadata(
        {
            "paper_id": "paper_1",
            "title": "Title",
            "year": "2025",
            "section_name": "Abstract",
            "chunk_type": "text",
            "page_start": 1,
            "page_end": 2,
            "file_name": "paper.pdf",
        }
    )

    assert set(REQUIRED_METADATA_FIELDS).issubset(metadata)


def test_build_and_inspect_index_with_fake_embeddings(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    persist_dir = tmp_path / "chroma"
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    write_chunks(chunks_path)

    stats = build_index(
        chunks_path=chunks_path,
        persist_dir=persist_dir,
        collection_name="test_chunks",
        reset=True,
        limit=3,
        batch_size=2,
        embedding_model=FakeEmbeddingModel(str(model_path)),
    )
    inspected = inspect_index(persist_dir=persist_dir, collection_name="test_chunks")

    assert stats["collection_count"] == 3
    assert stats["embedding_dim"] == 3
    assert inspected["collection_count"] == 3
    assert inspected["embedding_dim"] == 3
    assert inspected["embedding_backend"] == "sentence-transformers"
    assert inspected["embedding_model"] == str(model_path)
    assert inspected["embedding_local_path"] == str(model_path.resolve())
    assert inspected["sample_metadatas"]
