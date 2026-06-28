# PROJECT_STATUS.md

## 2026-06-28 Runnable-Version Checkpoint

- The current Streamlit system is runnable at `http://127.0.0.1:8501/`.
- Core app, Agent, and retrieval modules pass the import smoke check.
- Active chunks: `data/chunks/chunks.jsonl`, 1,864 records.
- Active Chroma: `vector_store/chroma`, collection `injection_molding_chunks`, 1,864 records, 1,024-dimensional embeddings.
- Local model configuration is present: Ollama `qwen2.5:7b` and sentence-transformers `E:/AI_Models/BAAI/bge-m3`.
- The current configuration snapshot is stored at `outputs/checkpoints/current_config_snapshot.md`.
- No business logic, paper data, chunks, or index contents were changed for this checkpoint.
- Next: audit whether the current knowledge base was built only from the 30-paper development set.

## 2026-06-28 Streamlit Chroma Readiness Fix

- Unified Streamlit and retrieval debug settings at `vector_store/chroma` / `injection_molding_chunks`.
- Confirmed `vector_store/chroma` remains a junction to the existing ASCII runtime directory.
- Streamlit readiness now opens the named Chroma collection and requires `collection.count() > 0`.
- DenseRetriever cache keys now include the Chroma collection ID, so a `--reset` rebuild invalidates stale collection handles.
- Retrieval debug mode reuses `src/retrieval/retrieval_debug.py` and displays seven requested diagnostics without exposing the absolute local runtime path.
- Verified CLI retrieval and the browser UI with 1,864 chunks, a 1,864-item collection, and 5 BM25 / 5 Dense / 5 Hybrid results.
- Test status: `python -m pytest -q` passed with 60 tests.

## 2026-06-27 One-click Pipeline

- Added `scripts/run_ingest_dev.py`: parses `data/dev_papers`, cleans sections, extracts knowledge cards, and builds chunks.
- Added `scripts/run_ingest_selected.py`: runs the same stages for `data/selected_papers` and writes independent `selected_*` outputs.
- Added `scripts/run_build_index.py`: builds Chroma from a chosen chunks JSONL file and defaults to the configured local `E:\AI_Models\BAAI\bge-m3` model.
- Added `scripts/run_eval.py`: runs retrieval and generation evaluation; Mock generation is the safe default and Ollama remains explicitly selectable.
- All pipeline scripts write timestamped logs under `data/logs/`, capture verbose child-module output, and report the exact failed step.
- Full `data/raw_papers` processing is blocked unless `--input data/raw_papers --confirm_full_run yes` is supplied explicitly.
- Dev pipeline verification passed: 30 papers, 561 pages, 147 sections, 30 paper cards, 98 defect cards, 105 method cards, 162 parameter cards, and 1,864 chunks.
- Dev chunk types: 860 text, 576 table/figure context, and 428 knowledge-card chunks.
- Section recognition rate was 100%; parse failures were 0.
- Test status: `python -m pytest -q` passed with 58 tests.
- The current Chroma index and evaluation baselines were not automatically rebuilt during dev ingest validation.

## 当前状态

项目骨架已创建，项目虚拟环境 `.venv` 已创建，依赖清单已补齐并安装完成。

默认本地模型方案已配置：

- LLM provider: `ollama`
- LLM model: `qwen2.5:7b`
- Embedding provider: `sentence-transformers`
- Embedding model: `BAAI/bge-m3`
- Embedding 本地路径：`E:\AI_Models\BAAI\bge-m3`

已创建 `scripts/check_env.py` 用于检查 Python 版本、关键依赖和项目路径，并已通过检查。

已实现文献库扫描功能 `src/ingest/scan_papers.py`。该脚本只读取文件系统元数据和文件名，不解析 PDF 全文。

最新扫描结果：

- 输出文件：`data/metadata/paper_inventory.csv`
- 总 PDF 数：896
- 年份分布：2022: 1；2023: 273；2024: 352；2025: 264；unknown: 6
- 主要关键词分布：injection molding: 165；optimization: 145；process parameter: 80；simulation: 39；CAE: 36；PP: 18；machine learning: 18；defect: 10；quality prediction: 5；warpage: 4；sensor: 2；ABS: 2；shrinkage: 1

已实现论文筛选功能 `src/ingest/select_papers.py`。该脚本只基于 `paper_inventory.csv` 中的文件名、title guess、keyword tags 和文件路径做筛选，不解析 PDF 全文。

最新筛选结果：

- 输出文件：`data/metadata/selected_papers.csv`
- `data/selected_papers/`：84 篇
- `data/dev_papers/`：当前实际检测 30 篇
- selected 类别分布：注塑缺陷 13；工艺参数 13；质量预测 10；工艺优化 19；算法方法 14；材料和场景 15
- dev 类别分布：注塑缺陷 4；工艺参数 4；质量预测 4；工艺优化 4；算法方法 4；材料和场景 4

已实现 PDF 论文解析功能 `src/ingest/parse_papers.py`。该脚本只处理 `data/dev_papers/`，使用 PyMuPDF 按页抽取文本并写入 JSONL，不在终端打印页面全文。

最新解析结果：

- 输出文件：`data/interim/parsed_docs.jsonl`
- 错误文件：`data/interim/parse_errors.csv`
- 总论文数：30
- 成功数：30
- 失败数：0
- 总页数：561
- 平均每页字数：1340.74
- JSONL 记录数：561

已实现论文文本清洗和章节识别功能 `src/ingest/clean_sections.py`。该脚本合并同一论文页面文本，清理页眉页脚、孤立页码、重复版权声明，修复英文断词，保留 Figure/Fig./Table/图/表标题，并将 References 及之后内容标记为默认不进入 RAG chunk。

最新清洗结果：

- 输出文件：`data/processed/cleaned_sections.jsonl`
- 报告文件：`data/processed/clean_report.md`
- 总论文数：30
- 章节识别成功论文数：30
- 章节识别成功率：100.00%
- 总 section 数：147
- References 标记 section 数：33
- 被删除文本比例：4.85%
- References 默认排除比例：13.08%
- 异常论文：0

已实现论文知识卡抽取功能 `src/ingest/extract_cards.py`。该脚本默认只使用规则和 section 信息，不调用 OpenAI 或其他付费 API；`--use_local_llm` 参数已预留但默认关闭。

最新知识卡抽取结果：

- `data/processed/paper_cards.jsonl`：30 条
- `data/processed/defect_cards.jsonl`：98 条
- `data/processed/method_cards.jsonl`：105 条
- `data/processed/parameter_cards.jsonl`：162 条
- `data/manual_review/review_queue.csv`：5 条低置信度复核记录
- defect evidence_text 最大长度：694 字符，按规则只保存短证据句，不保存整段全文

已实现 section-aware 论文 chunk 切分功能 `src/index/build_chunks.py`。该脚本默认跳过 `is_reference_section=true` 的 section，将 Abstract、Conclusion/Discussion、Method/Experiment/Results 按章节策略切分，图表上下文单独输出为 `table_or_figure_context`，知识卡输出为 `knowledge_card`。

最新 chunk 构建结果：

- 输出文件：`data/chunks/chunks.jsonl`
- 报告文件：`data/chunks/chunk_report.md`
- 总 chunk 数：1864
- 覆盖论文数：30
- 每篇平均 chunk 数：62.13
- chunk_type 分布：text 860；table_or_figure_context 576；knowledge_card 428
- chunk 长度范围：14-1100 字符
- 长度分布：<120: 419；120-299: 206；300-499: 259；500-900: 693；901-1100: 287；>1100: 0

已实现 Chroma 向量库构建和检查功能：

- 构建脚本：`src/index/build_vector_index.py`
- 检查脚本：`src/index/inspect_index.py`
- 输入文件：`data/chunks/chunks.jsonl`
- 项目输出入口：`vector_store/chroma/`
- Windows/Chroma runtime 路径：`C:\Users\JZ\.cache\injection_molding_rag_agent\chroma`
- `vector_store/chroma/` 仍为指向上述 runtime 路径的 Windows junction
- collection：`injection_molding_chunks`
- collection 数量：1
- chunk 数：1864
- embedding backend：`sentence-transformers`
- embedding model：`BAAI/bge-m3`
- embedding 本地路径：`E:\AI_Models\BAAI\bge-m3`
- embedding 维度：1024
- 全量索引构建参数：`batch_size=4`，未发生内存错误，无需降级 batch size
- 全量 `pytest`：14 passed

说明：embedding 已从 `BAAI/bge-small-zh-v1.5` 切换为本地 `BAAI/bge-m3`。已依次完成 20 条、200 条和全量 1864 条 chunk 的构建与独立 inspect 验证；正式索引未使用 hashing fallback，也未调用 OpenAI API。

已实现本地检索模块：

- BM25：`src/retrieval/bm25_retriever.py`，从 `data/chunks/chunks.jsonl` 建立关键词索引，并为中文注塑问题提供基础领域词扩展
- Dense：`src/retrieval/dense_retriever.py`，使用现有 Chroma collection 和本地 `BAAI/bge-m3` 查询向量
- Hybrid：`src/retrieval/hybrid_retriever.py`，默认 `dense_weight=0.6`、`bm25_weight=0.4`，按 `chunk_id` 去重
- 调试脚本：`src/retrieval/retrieval_debug.py`，支持单问题或内置四个注塑测试问题，分别展示 BM25、Dense、Hybrid top5
- 统一结果字段：`chunk_id`、`paper_id`、`title`、`section_name`、`chunk_type`、`score`、`source`、`text_preview`、`metadata`
- 终端预览限制为 200 字，并兼容 PDF 不可见字符和 Windows 控制台编码
- 四个内置问题真实检索 smoke test：通过
- Chroma 查询结果已与全量 embedding 直接余弦排序交叉验证，一致
- 当前全量 `pytest`：18 passed

已知观察：当前 Dense top 结果容易集中在同一篇论文，这是 bge-m3 对现有 query/chunk 的原始向量排序结果，不是 Chroma 或 junction 读取异常。后续应通过检索评测集和权重实验量化调优。

已实现检索重排模块 `src/retrieval/reranker.py`：

- `rule` 模式：结合 query 关键词覆盖、原始 Hybrid 分数、section 权重和 chunk_type 权重重排
- `Abstract`、`Conclusion`、`Results`、`Discussion` 获得较高 section 权重，`References` 默认降权
- 工艺参数建议、缺陷原因类问题会额外加权 `parameter_card` 和 `defect_card` 知识卡
- `model` 模式：支持本地 sentence-transformers `CrossEncoder` / BGE reranker 目录，启用 `local_files_only=True`，不联网下载、不调用付费 API
- 本地模型缺失或加载失败时自动回退到 `rule`，并保留清晰的 fallback reason
- 重排结果保留 `original_score` 和 `rerank_score`
- `retrieval_debug.py` 已增加 `--use_rerank`、`--rerank_mode`、`--reranker_model`，可对比 Hybrid 重排前后结果
- rule 真实检索 smoke test：通过
- model 缺失目录回退 smoke test：通过
- 当前机器尚未配置独立的本地 CrossEncoder/BGE reranker 模型
- 当前全量 `pytest`：21 passed

已实现口语化查询改写模块 `src/retrieval/query_rewrite.py`：

- 纯规则实现，不调用付费 API
- 输出字段：`original_query`、`normalized_query`、`intent`、`defect_type`、`material`、`parameters`、`quality_metric`、`must_have_terms`、`expanded_terms`、`risk_level`
- 支持翘曲、缩水/缩痕、熔接痕、短射、飞边等中英文缺陷映射
- 支持 PMMA、PC、ABS、PP、POM 材料识别和常见注塑工艺参数识别
- 支持“缩水咋办”“透明件发雾”“保压是不是越大越好”等口语表达
- 透明件发雾且未指定材料时，提供 PMMA/PC 候选，并识别 `transmittance/haze`
- 直接索要具体生产参数或设定值时标记 `risk_level=high` 和 `intent=parameter_recommendation`
- 提供命令行 JSON 测试入口：`python -m src.retrieval.query_rewrite "问题"`
- 三个指定口语问题 CLI smoke test：通过
- 当前全量 `pytest`：28 passed

已实现 RAG 答案生成与引用保护模块：

- Prompt：`src/rag/prompts.py`，强制仅基于 evidence 回答、禁止编造论文名/参数/实验结论、证据不足时明确提示，并要求关键结论附 `[E编号]`
- 答案生成：`src/rag/answer_generator.py`，输入原问题、query rewrite 和 rerank 证据，输出 `answer`、`evidence_list`、`confidence`、`limitations`、`need_human_review`
- 默认 LLM：本地 Ollama `qwen2.5:7b`，base URL 保持 `http://localhost:11434`
- Mock：支持显式 `--llm_mode mock`；Ollama 不可用或模型调用失败时自动回退，并在 limitations 中记录原因
- Citation guard：`src/rag/citation_guard.py`，检查缺失引用、无效 evidence 编号、证据外具体数值和显式论文名
- Citation guard 或直接生产参数请求触发高风险时，设置 `need_human_review=true` 并兼容追加到 `data/manual_review/review_queue.csv`
- 命令行入口会串联 query rewrite、BM25、Dense、Hybrid、rule rerank 和答案生成
- Mock 完整 pipeline smoke test：通过
- 本地 Ollama 真实答案 smoke test：通过，输出 `[E1][E2]`，`llm_mode=ollama`
- 当前全量 `pytest`：33 passed

已根据 `docs/agent_tool_design.md` 实现 Agent 工具层 `src/agent/tools.py`：

- `search_papers_tool`：复用 BM25、Dense、Hybrid 和 rule rerank，输出可审计 evidence、source location、citation 和 confidence
- `defect_diagnosis_tool`：输出论文证据支持的候选原因、相关参数和排查项，不给最终生产诊断
- `parameter_effect_tool`：整理参数影响趋势、机制、参数交互和适用条件，高风险或低覆盖时要求人工复核
- `method_compare_tool`：生成带 evidence ID 的定性方法对比，并避免在证据并列或企业部署场景下武断选型
- `evidence_extract_tool`：将检索片段转换为结构化 evidence table，支持实体归一化、去重和低置信度标记
- `human_review_tool`：创建 pending 人工复核工单，并追加到 `data/manual_review/agent_review_tickets.jsonl`
- `knowledge_gap_tool`：记录并去重知识缺口，追加到 `data/manual_review/knowledge_gaps.jsonl`
- 7 个核心工具均有独立最小测试，测试使用 fake search，不加载 bge-m3 或 Ollama
- 真实关键词检索 smoke test：返回 3 条证据，overall confidence 约 0.687
- 当前全量 `pytest`：40 passed

已实现简化 Agentic workflow：

- 状态：`src/agent/state.py`，包含 query、normalized query、intent、tool calls、retrieved evidence、answer、confidence、human review、errors、step count，并保留节点与检索历史用于审计
- 工作流：`src/agent/workflow.py`，显式实现 `classify_and_rewrite_query`、`retrieve_evidence`、`rerank_evidence`、`decide_answer_or_review`、`generate_answer`、`citation_check`、`final_response` 七个节点
- 默认循环限制：`max_steps=8`、`max_tool_calls=5`
- 同一 normalized query 连续两次检索且结果数量/最佳分数无改善时停止重试
- 工具报错超过 2 次时创建人工复核工单
- 空证据或低于默认 0.30 阈值的证据最多重检一次，随后进入 knowledge gap；高风险和冲突证据直接进入 human review
- Guardrails：`src/agent/guardrails.py`，禁止证据外具体工艺参数范围、禁止把论文结论写成生产指令、禁止在证据冲突时输出单一确定答案
- citation/guardrail 不通过时，最终响应会替换违规草稿，不向用户透传
- `tests/test_guardrails.py`：4 项 guardrail 测试
- `tests/test_pipeline_smoke.py`：5 项状态机测试，覆盖正常 7 节点、低证据 8 步停止、三次工具报错、tool call 上限和冲突转人工
- 真实 Hybrid + rule rerank + mock answer workflow smoke test：7 节点完成，10 条证据，无错误，无需人工复核
- 当前全量 `pytest`：49 passed

已实现简单可控的 Agent memory：`src/agent/memory.py`。

- 使用本地 JSONL，默认路径为已忽略的 `data/logs/agent_memory.jsonl`
- Memory 为显式 opt-in，不会在 workflow 中静默自动记录
- 每条记录包含 timestamp、脱敏 query、intent、top evidence IDs、answer confidence、human review 标记和脱敏 user feedback
- 不保存答案或论文全文；最多保存 5 个 evidence ID，每条 evidence 摘要最多 160 字
- 写入前对邮箱、手机号、身份证号、姓名标签、工号/客户编号/订单号等常见隐私模式做基础脱敏
- `find_recent_similar` 按 intent、标准化缺陷/参数实体和词面相似度读取最近 N 条同类问题
- `similar_question_hint` 可生成“你之前问过类似问题”提示
- `export_statistics` 输出高频缺陷、高频参数、低置信度问题数和人工复核数
- `tests/test_memory.py` 覆盖脱敏、摘要截断、相似问题和统计
- 当前全量 `pytest`：50 passed

已修复 Agent memory CLI：

- `src/agent/memory.py` 现包含 `main()` 和 `if __name__ == "__main__": main()`
- CLI 参数：`--memory_path`、`--recent_n`、`--demo`、`--stats`
- CLI 默认路径：`data/logs/agent_trace.jsonl`，目录不存在时自动创建
- `--demo` 写入 3 条模拟记录，读取最近 N 条并打印总记录数、高频缺陷、高频参数和低置信度数量
- `--stats` 可在独立进程中重新读取 JSONL 并打印统计
- `read_recent` 支持读取最近 N 条记录，最新记录优先
- Memory 不再持久化派生缺陷/参数字段，统计时从脱敏 query 重新解析，进一步缩小存储面
- `full_text`、`paper_full_text`、`raw_text`、`document(s)` 等字段会从嵌套 feedback 中移除，也不会从 evidence 写入
- `python -m src.agent.memory --demo`：通过，写入 3 条、recent 3 条、low confidence 1 条
- `python -m src.agent.memory --stats --memory_path data/logs/agent_trace.jsonl`：通过，total records 3 条
- `tests/test_memory.py`：2 passed
- 当前全量 `pytest`：51 passed

已实现 Streamlit Demo：

- 页面入口：`src/app/streamlit_app.py`
- 启动脚本：`scripts/run_app.py`
- 项目启动配置：`.streamlit/config.toml`，关闭不必要的第三方模块文件监视与 usage stats
- 支持模式：普通 RAG、缺陷诊断、方法对比、检索调试
- 页面展示：最终答案、paper_id/title/section/score/text_preview 五列证据表、confidence、human review 状态和人工复核原因
- Query Rewrite 与运行详情为可选折叠内容
- 页面仅展示安全证据字段和友好错误，不显示 PDF 路径、模型路径、Chroma runtime 路径或 Python traceback
- Chunk、Chroma、embedding model、Ollama 提供轻量就绪状态；组件缺失时显示友好提示
- 真实检索调试交互：BM25/Dense/Hybrid 各返回 5 条，页面展示 Hybrid 5 条证据
- 桌面与 390px 窄屏浏览器检查：通过，无控件重叠
- 浏览器控制台错误：0；页面 Traceback：0；绝对 Windows 路径暴露：0
- `tests/test_streamlit_app.py`：2 passed
- 当前全量 `pytest`：53 passed
- Streamlit 当前运行地址：`http://127.0.0.1:8501`

已实现检索与生成评测脚本：

- 检索评测：`src/eval/eval_retrieval.py`
- 生成评测：`src/eval/eval_generation.py`
- 输入：`data/eval/eval_questions.csv`，共 50 题
- 检索输出：`data/eval/retrieval_eval.csv`，50 行、27 列、0 错误
- 生成输出：`data/eval/generation_eval.csv`，50 行、30 列、0 错误
- 检索指标包括 Hit@K、expected keyword recall、top/mean score、论文覆盖数、overall confidence 和 latency
- 生成指标包括答案关键词召回、引用数量与合法性、证据外数值/论文名、answer guardrail、confidence、human review 和 latency
- 评测逐题容错，单题失败会记录 `status=error` 和错误摘要，不中断整批
- BM25、bge-m3、Chroma 和 rule reranker 在一次评测中只初始化一次
- 完整检索基线：50/50 成功，Hit@5=0.7200，平均关键词召回率=0.2403
- 完整生成基线使用 `--llm_mode mock`：50/50 成功，平均答案关键词召回率=0.2003，citation guard 通过率=1.0000
- Mock 生成结果仅用于验证 pipeline、引用与指标落盘，不代表本地 Ollama 的生成质量
- `tests/test_eval.py`：1 passed
- 当前全量 `pytest`：54 passed

当前系统仅发现 Python 3.12.4；脚本会提示推荐使用 Python 3.10 或 3.11，但不会阻塞当前开发。

## 下一步

基于 retrieval_eval.csv 调优检索词扩展与融合权重，并选择小样本运行本地 Ollama 生成评测。
