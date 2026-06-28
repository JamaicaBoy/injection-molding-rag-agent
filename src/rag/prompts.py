from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """你是注塑成型领域的论文 RAG 助手。

必须遵守以下规则：
1. 只能基于给定 evidence 回答，不得使用未提供的外部事实。
2. 不得编造论文名、参数范围、具体数值或实验结论。
3. 如果证据不足，必须明确写出“当前论文库证据不足”。
4. 工艺参数建议只能作为候选方向，不能作为直接生产指令；实际生产设置需要工程师结合材料、设备、模具和验证试验确认。
5. 每条关键结论后必须附 evidence 编号，格式为 [E1]、[E2]。
6. 不要引用不存在的 evidence 编号，不要输出参考文献中未出现的论文名。
"""


def format_evidence(evidence_list: list[dict[str, Any]]) -> str:
    if not evidence_list:
        return "(no evidence)"
    blocks: list[str] = []
    for evidence in evidence_list:
        blocks.append(
            "\n".join(
                (
                    f"[{evidence['evidence_id']}]",
                    f"title: {evidence.get('title', '')}",
                    f"paper_id: {evidence.get('paper_id', '')}",
                    f"section: {evidence.get('section_name', '')}",
                    f"chunk_type: {evidence.get('chunk_type', '')}",
                    f"content: {str(evidence.get('text_preview', ''))[:600]}",
                )
            )
        )
    return "\n\n".join(blocks)


def build_answer_prompt(
    question: str,
    query_rewrite: dict[str, Any],
    evidence_list: list[dict[str, Any]],
) -> str:
    return f"""用户问题：
{question}

结构化查询：
{json.dumps(query_rewrite, ensure_ascii=False, indent=2)}

Evidence：
{format_evidence(evidence_list)}

请直接给出中文答案。先回答核心问题，再说明证据限制。每条关键结论后使用 [E编号] 引用。
如果风险等级为 high，必须明确说明需要人工审核，且不得给出可直接执行的生产参数。
"""

