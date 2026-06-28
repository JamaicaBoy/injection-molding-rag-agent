from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SECTIONS = PROJECT_ROOT / "data" / "processed" / "cleaned_sections.jsonl"
DEFAULT_PAPER_CARDS = PROJECT_ROOT / "data" / "processed" / "paper_cards.jsonl"
DEFAULT_DEFECT_CARDS = PROJECT_ROOT / "data" / "processed" / "defect_cards.jsonl"
DEFAULT_METHOD_CARDS = PROJECT_ROOT / "data" / "processed" / "method_cards.jsonl"
DEFAULT_PARAMETER_CARDS = PROJECT_ROOT / "data" / "processed" / "parameter_cards.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "chunks" / "chunk_report.md"

TARGET_CHARS = 760
MIN_CHARS = 280
MAX_CHARS = 1100
OVERLAP_CHARS = 120

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+|\n+")
FIGURE_TABLE_PATTERN = re.compile(r"^\s*(figure|fig\.?|table|图|表)\s*[\d一二三四五六七八九十ivxlc]*", re.IGNORECASE)
ENGLISH_WORD_PATTERN = re.compile(r"[A-Za-z]+")

CHUNK_FIELDS = [
    "chunk_id",
    "paper_id",
    "file_name",
    "title",
    "year",
    "section_name",
    "chunk_type",
    "text",
    "char_count",
    "token_estimate",
    "page_start",
    "page_end",
    "metadata",
]


def read_jsonl(path: Path, optional: bool = False) -> list[dict[str, Any]]:
    if optional and not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def title_from_file_name(file_name: str) -> str:
    title = Path(file_name).stem
    title = re.sub(r"^\s*.+?\s+-\s+(19\d{2}|20\d{2})\s+-\s+", "", title)
    title = re.sub(r"[_]+", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def year_from_file_name(file_name: str) -> str:
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", file_name)
    return match.group(1) if match else ""


def build_paper_card_lookup(paper_cards: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(card.get("paper_id")): card for card in paper_cards if card.get("paper_id")}


def estimate_tokens(text: str) -> int:
    english_words = ENGLISH_WORD_PATTERN.findall(text)
    if english_words and len(" ".join(english_words)) / max(len(text), 1) > 0.45:
        return max(1, int(len(english_words) * 1.15))
    return max(1, math.ceil(len(text) / 1.7))


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_meaningful_text(text: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text))


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n|\n", text) if paragraph.strip()]
    merged: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) < 80 and not is_figure_table_line(paragraph):
            buffer = f"{buffer} {paragraph}".strip()
            continue
        if buffer:
            merged.append(buffer)
            buffer = ""
        merged.append(paragraph)
    if buffer:
        merged.append(buffer)
    return merged


def split_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    if len(sentences) <= 1 and len(text) > MAX_CHARS:
        sentences = [part.strip() for part in re.split(r"[;；]", text) if part.strip()]
    return sentences or [text.strip()]


def is_figure_table_line(line: str) -> bool:
    return bool(FIGURE_TABLE_PATTERN.match(line.strip()))


def extract_figure_table_contexts(section_text: str) -> tuple[list[str], str]:
    lines = section_text.splitlines()
    context_indexes: set[int] = set()
    for index, line in enumerate(lines):
        if is_figure_table_line(line):
            for context_index in range(max(0, index - 1), min(len(lines), index + 3)):
                context_indexes.add(context_index)

    contexts: list[str] = []
    if context_indexes:
        current: list[str] = []
        last_index: int | None = None
        for index in sorted(context_indexes):
            if last_index is not None and index > last_index + 1 and current:
                contexts.append(normalize_text("\n".join(current)))
                current = []
            current.append(lines[index])
            last_index = index
        if current:
            contexts.append(normalize_text("\n".join(current)))

    remaining_lines = [line for index, line in enumerate(lines) if index not in context_indexes]
    return [context for context in contexts if context], normalize_text("\n".join(remaining_lines))


def window_sentences(sentences: list[str], target_chars: int = TARGET_CHARS, max_chars: int = MAX_CHARS) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if sentence_len > max_chars:
            if current:
                chunks.append(normalize_text(" ".join(current)))
                current = []
                current_len = 0
            chunks.extend(split_long_sentence_like_text(sentence, max_chars=max_chars))
            continue

        if current and current_len + sentence_len + 1 > max_chars:
            chunks.append(normalize_text(" ".join(current)))
            overlap = build_overlap(current)
            current = [overlap] if overlap else []
            current_len = len(overlap)

        current.append(sentence)
        current_len += sentence_len + 1
        if current_len >= target_chars:
            chunks.append(normalize_text(" ".join(current)))
            overlap = build_overlap(current)
            current = [overlap] if overlap else []
            current_len = len(overlap)

    if current:
        text = normalize_text(" ".join(current))
        if text and (not chunks or text != chunks[-1]):
            chunks.append(text)
    return merge_short_chunks(chunks)


def split_long_sentence_like_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    parts = [part.strip() for part in re.split(r"([,，;；。.!?\n])", text) if part.strip()]
    combined: list[str] = []
    buffer = ""
    for part in parts:
        if len(buffer) + len(part) + 1 <= max_chars:
            buffer = f"{buffer}{part}" if part in ",，;；" else f"{buffer} {part}".strip()
        else:
            if buffer:
                combined.append(normalize_text(buffer))
            buffer = part
    if buffer:
        combined.append(normalize_text(buffer))
    return combined or [text[:max_chars]]


def hard_wrap_without_sentence_break(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    pieces = split_long_sentence_like_text(text, max_chars=max_chars)
    wrapped: list[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            wrapped.append(piece)
            continue
        start = 0
        while start < len(piece):
            wrapped.append(piece[start : start + max_chars].strip())
            start += max_chars
    return [piece for piece in wrapped if piece and has_meaningful_text(piece)]


def build_overlap(sentences: list[str]) -> str:
    overlap_sentences: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > OVERLAP_CHARS and overlap_sentences:
            break
        overlap_sentences.append(sentence)
        total += len(sentence)
    return normalize_text(" ".join(reversed(overlap_sentences)))


def merge_short_chunks(chunks: list[str]) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < MIN_CHARS and len(merged[-1]) + len(chunk) + 1 <= MAX_CHARS:
            merged[-1] = normalize_text(merged[-1] + " " + chunk)
        else:
            merged.append(chunk)
    return merged


def split_section_text(section_name: str, text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    if section_name == "Abstract":
        return window_sentences(split_sentences(text), target_chars=850, max_chars=1050)
    if section_name in {"Conclusion", "Discussion"} and len(text) <= 1400:
        return [text]
    paragraphs = split_paragraphs(text)
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHARS:
            if buffer:
                chunks.extend(window_sentences(split_sentences(" ".join(buffer))))
                buffer = []
                buffer_len = 0
            chunks.extend(window_sentences(split_sentences(paragraph)))
            continue
        if buffer and buffer_len + len(paragraph) + 1 > MAX_CHARS:
            chunks.extend(window_sentences(split_sentences(" ".join(buffer))))
            buffer = []
            buffer_len = 0
        buffer.append(paragraph)
        buffer_len += len(paragraph) + 1
    if buffer:
        chunks.extend(window_sentences(split_sentences(" ".join(buffer))))
    return chunks


def stable_chunk_id(parts: list[str]) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"


def make_chunk(
    *,
    paper_id: str,
    file_name: str,
    title: str,
    year: str,
    section_name: str,
    chunk_type: str,
    text: str,
    page_start: Any,
    page_end: Any,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "title": title,
        "year": year,
        "section": section_name,
        "paper_id": paper_id,
        "chunk_type": chunk_type,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    text = normalize_text(text)
    return {
        "chunk_id": stable_chunk_id([paper_id, section_name, chunk_type, str(page_start), text[:120]]),
        "paper_id": paper_id,
        "file_name": file_name,
        "title": title,
        "year": year,
        "section_name": section_name,
        "chunk_type": chunk_type,
        "text": text,
        "char_count": len(text),
        "token_estimate": estimate_tokens(text),
        "page_start": page_start,
        "page_end": page_end,
        "metadata": metadata,
    }


def split_oversized_chunk(chunk: dict[str, Any], max_chars: int = MAX_CHARS) -> list[dict[str, Any]]:
    if int(chunk["char_count"]) <= max_chars:
        return [chunk]
    parts = hard_wrap_without_sentence_break(str(chunk["text"]), max_chars=max_chars)
    split_chunks: list[dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        new_chunk = dict(chunk)
        metadata = dict(chunk["metadata"])
        metadata["split_from_chunk_id"] = chunk["chunk_id"]
        metadata["split_index"] = index
        metadata["split_total"] = len(parts)
        new_chunk["text"] = part
        new_chunk["char_count"] = len(part)
        new_chunk["token_estimate"] = estimate_tokens(part)
        new_chunk["metadata"] = metadata
        new_chunk["chunk_id"] = stable_chunk_id(
            [chunk["chunk_id"], str(index), chunk["paper_id"], chunk["chunk_type"], part[:120]]
        )
        split_chunks.append(new_chunk)
    return split_chunks


def split_oversized_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        for split_chunk in split_oversized_chunk(chunk):
            if has_meaningful_text(str(split_chunk["text"])):
                result.append(split_chunk)
    return result


def build_text_chunks(sections: list[dict[str, Any]], paper_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section in sections:
        if section.get("is_reference_section"):
            continue
        paper_id = str(section["paper_id"])
        file_name = str(section.get("file_name", ""))
        card = paper_lookup.get(paper_id, {})
        title = str(card.get("title") or title_from_file_name(file_name))
        year = str(card.get("year") or year_from_file_name(file_name))
        section_name = str(section.get("section_name", "Unknown"))
        section_text = str(section.get("clean_text", ""))
        figure_contexts, remaining_text = extract_figure_table_contexts(section_text)

        for index, context in enumerate(figure_contexts, start=1):
            chunks.append(
                make_chunk(
                    paper_id=paper_id,
                    file_name=file_name,
                    title=title,
                    year=year,
                    section_name=section_name,
                    chunk_type="table_or_figure_context",
                    text=context,
                    page_start=section.get("page_start", ""),
                    page_end=section.get("page_end", ""),
                    extra_metadata={"figure_table_context_index": index},
                )
            )

        for index, text_chunk in enumerate(split_section_text(section_name, remaining_text), start=1):
            chunks.append(
                make_chunk(
                    paper_id=paper_id,
                    file_name=file_name,
                    title=title,
                    year=year,
                    section_name=section_name,
                    chunk_type="text",
                    text=text_chunk,
                    page_start=section.get("page_start", ""),
                    page_end=section.get("page_end", ""),
                    extra_metadata={"section_chunk_index": index},
                )
            )
    return chunks


def card_to_text(card_type: str, card: dict[str, Any]) -> str:
    lines = [f"card_type: {card_type}"]
    for key, value in card.items():
        if isinstance(value, list):
            value_text = "; ".join(str(item) for item in value)
        elif isinstance(value, dict):
            value_text = json.dumps(value, ensure_ascii=False)
        else:
            value_text = str(value)
        if value_text:
            lines.append(f"{key}: {value_text}")
    return "\n".join(lines)


def build_knowledge_card_chunks(
    paper_cards: list[dict[str, Any]],
    defect_cards: list[dict[str, Any]],
    method_cards: list[dict[str, Any]],
    parameter_cards: list[dict[str, Any]],
    paper_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    card_groups = [
        ("paper_card", paper_cards, "paper_id"),
        ("defect_card", defect_cards, "evidence_paper_id"),
        ("method_card", method_cards, "evidence_paper_id"),
        ("parameter_card", parameter_cards, "evidence_paper_id"),
    ]
    for card_type, cards, id_field in card_groups:
        for index, card in enumerate(cards, start=1):
            paper_id = str(card.get(id_field, ""))
            if not paper_id:
                continue
            paper_card = paper_lookup.get(paper_id, {})
            title = str(paper_card.get("title") or card.get("title") or "")
            year = str(paper_card.get("year") or card.get("year") or "")
            file_name = str(paper_card.get("file_name") or "")
            chunks.append(
                make_chunk(
                    paper_id=paper_id,
                    file_name=file_name,
                    title=title,
                    year=year,
                    section_name="Knowledge Card",
                    chunk_type="knowledge_card",
                    text=card_to_text(card_type, card),
                    page_start="",
                    page_end="",
                    extra_metadata={"card_type": card_type, "card_index": index},
                )
            )
    return chunks


def build_report(chunks: list[dict[str, Any]]) -> str:
    total_chunks = len(chunks)
    paper_counts = Counter(chunk["paper_id"] for chunk in chunks)
    avg_per_paper = mean(paper_counts.values()) if paper_counts else 0
    length_buckets = Counter()
    section_counts = Counter(chunk["section_name"] for chunk in chunks)
    short_chunks = [chunk for chunk in chunks if chunk["char_count"] < 120]
    long_chunks = [chunk for chunk in chunks if chunk["char_count"] > MAX_CHARS]

    for chunk in chunks:
        length = int(chunk["char_count"])
        if length < 120:
            bucket = "<120"
        elif length < 300:
            bucket = "120-299"
        elif length < 500:
            bucket = "300-499"
        elif length <= 900:
            bucket = "500-900"
        elif length <= 1100:
            bucket = "901-1100"
        else:
            bucket = ">1100"
        length_buckets[bucket] += 1

    lines = [
        "# Chunk Report",
        "",
        "## Summary",
        "",
        f"- Total chunks: {total_chunks}",
        f"- Papers represented: {len(paper_counts)}",
        f"- Average chunks per paper: {avg_per_paper:.2f}",
        "",
        "## Chunk Length Distribution",
        "",
    ]
    for bucket in ["<120", "120-299", "300-499", "500-900", "901-1100", ">1100"]:
        lines.append(f"- {bucket}: {length_buckets.get(bucket, 0)}")

    lines.extend(["", "## Chunk Count by Section", ""])
    for section_name, count in section_counts.most_common():
        lines.append(f"- {section_name}: {count}")

    lines.extend(["", "## Short Chunk Samples", ""])
    if short_chunks:
        for chunk in short_chunks[:5]:
            lines.append(f"- `{chunk['chunk_id']}` | {chunk['paper_id']} | {chunk['section_name']} | chars={chunk['char_count']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Long Chunk Samples", ""])
    if long_chunks:
        for chunk in long_chunks[:5]:
            lines.append(f"- `{chunk['chunk_id']}` | {chunk['paper_id']} | {chunk['section_name']} | chars={chunk['char_count']}")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def write_report(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(chunks), encoding="utf-8")


def print_stats(chunks: list[dict[str, Any]]) -> None:
    paper_counts = Counter(chunk["paper_id"] for chunk in chunks)
    chunk_types = Counter(chunk["chunk_type"] for chunk in chunks)
    length_values = [int(chunk["char_count"]) for chunk in chunks]
    print(f"总 chunk 数: {len(chunks)}")
    print(f"论文数: {len(paper_counts)}")
    print(f"每篇平均 chunk 数: {mean(paper_counts.values()) if paper_counts else 0:.2f}")
    print("chunk_type 分布:")
    for chunk_type, count in chunk_types.most_common():
        print(f"  {chunk_type}: {count}")
    if length_values:
        print(f"chunk 长度: min={min(length_values)}, avg={mean(length_values):.2f}, max={max(length_values)}")


def build_chunks(
    sections_path: Path = DEFAULT_SECTIONS,
    paper_cards_path: Path = DEFAULT_PAPER_CARDS,
    defect_cards_path: Path = DEFAULT_DEFECT_CARDS,
    method_cards_path: Path = DEFAULT_METHOD_CARDS,
    parameter_cards_path: Path = DEFAULT_PARAMETER_CARDS,
    output_path: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> list[dict[str, Any]]:
    sections = read_jsonl(sections_path)
    paper_cards = read_jsonl(paper_cards_path)
    defect_cards = read_jsonl(defect_cards_path, optional=True)
    method_cards = read_jsonl(method_cards_path, optional=True)
    parameter_cards = read_jsonl(parameter_cards_path, optional=True)
    paper_lookup = build_paper_card_lookup(paper_cards)

    chunks = build_text_chunks(sections, paper_lookup)
    chunks.extend(build_knowledge_card_chunks(paper_cards, defect_cards, method_cards, parameter_cards, paper_lookup))
    chunks = split_oversized_chunks(chunks)
    write_jsonl(output_path, chunks)
    write_report(report_path, chunks)
    print_stats(chunks)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build section-aware RAG chunks from cleaned paper sections and cards.")
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--paper_cards", type=Path, default=DEFAULT_PAPER_CARDS)
    parser.add_argument("--defect_cards", type=Path, default=DEFAULT_DEFECT_CARDS)
    parser.add_argument("--method_cards", type=Path, default=DEFAULT_METHOD_CARDS)
    parser.add_argument("--parameter_cards", type=Path, default=DEFAULT_PARAMETER_CARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_chunks(
        sections_path=args.sections,
        paper_cards_path=args.paper_cards,
        defect_cards_path=args.defect_cards,
        method_cards_path=args.method_cards,
        parameter_cards_path=args.parameter_cards,
        output_path=args.output,
        report_path=args.report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
