# NEXT_TASK.md

## Current Next Task (2026-06-28)

审计当前知识库来源和论文数量。

重点确认 `data/chunks/chunks.jsonl` 和 Chroma collection `injection_molding_chunks` 的唯一 `paper_id` 数量、来源清单，以及当前知识库是否只使用了 30 篇开发集论文。本任务只做来源审计，不读取或打印 `data/raw_papers/` 的论文全文。

## Current Next Task (2026-06-28)

The Streamlit Chroma readiness issue is resolved. Continue with retrieval-quality analysis or run the existing evaluation pipeline; no index rebuild is required for this fix.

## Current Next Task (2026-06-27)

Run `python scripts/run_build_index.py --reset` to synchronize Chroma with the newly verified dev chunks, then run `python scripts/run_eval.py --llm_mode mock`. After reviewing that baseline, use a small explicit Ollama evaluation such as `python scripts/run_eval.py --llm_mode ollama --limit 10`.

当前 50 题检索与 Mock 生成评测基线已经落盘。下一步分析 retrieval_eval.csv 的漏召回问题，调优查询扩展与 Hybrid 权重，并选择小样本运行本地 Ollama 生成质量评测。
