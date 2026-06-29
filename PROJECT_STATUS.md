# PROJECT_STATUS.md

## 2026-06-29 发布前检查

- 已准备生成 GitHub Release 数据包。
- GitHub remote、`main` 分支、敏感数据忽略规则和 Streamlit 启动 smoke test 已检查。
- 当前默认 corpus mode 为 `full`，使用 `data/chunks/full_chunks.jsonl`、`vector_store/chroma_full` 和 collection `injection_papers_full`。
- 当前工作区仍有已跟踪修改与未跟踪源码/配置/文档，正式提交前需要人工确认暂存范围；没有暂存、提交或推送大型数据文件。

## 2026-06-29 GitHub Release Package: full_release_no_pdf_v1

- The public full-corpus package is fixed to `full_release_no_pdf_v1`; there are no alternative packaging modes.
- Added the fixed GitHub Release configuration in `configs/artifact_config.yaml` with release tag `full-release-v1`.
- Rebuilt `scripts/package_full_release.py` to package only full chunks, public metadata, the requested knowledge-card files, and `vector_store/chroma_full` plus the generated manifest files.
- The packager rejects every PDF and every path containing `raw_papers`; it never reads or packages the 896 source PDFs.
- Archives larger than 1.5 GiB are split as `full_release_no_pdf_v1.zip.001`, `.002`, `.003`, and so on.
- Rebuilt `scripts/verify_full_release.py` to verify SHA256, JSONL validity, unique paper IDs, Chroma readability, vector count, and embedding dimension.
- Rebuilt `scripts/download_full_release.py` to use GitHub Release assets only, require consecutive numbered parts, extract safely into a temporary directory, verify there, and install only after verification passes.
- `public_full_artifact` now reads from `artifacts/full_release_no_pdf_v1/`.
- Generated `data/metadata/paper_metadata.csv` from the existing 896-row full inventory without reading PDF contents.
- Real-data preflight: 28 payload files, about 627.3 MiB before ZIP compression, 43,343 chunks, 559 unique chunk paper IDs, 43,343 Chroma vectors, 1,024 dimensions, and zero PDFs.
- Packaging now uses an offline Chroma snapshot in an ASCII temporary directory, preventing live SQLite changes and Unicode-path HNSW failures from invalidating checksums.
- The packager verifies every ZIP member against `MANIFEST.json` before reporting success; the verifier opens a disposable Chroma copy so verification never mutates the artifact.
- Tests: `python -m pytest tests/test_full_release.py -q` passed (5 tests).
- Generated `dist/full_release_no_pdf_v1/full_release_no_pdf_v1.zip` (467,144,742 bytes) and independently extracted and verified it: 28 checksums, 43,343 chunks/vectors, 559 paper IDs, and 1,024 dimensions.
- Nothing has been uploaded. `github_owner` must be set to the real GitHub owner before using the downloader.

## 2026-06-29（追加功能）打包/校验脚本支持命令行路径覆盖 + 本地自检

- 用户在真实机器上用 `python scripts/package_full_release.py --output ...` 和 `python scripts/verify_full_release.py --artifact_dir ...` 这样的命令行参数尝试自定义输出/校验路径，但两个脚本当时都没有 `argparse`，这些参数被静默忽略（Python 不读取未使用的 `sys.argv`，也不会报错）。打包本身是成功的（28 个文件、43343 chunk、559 唯一 paper_id、单个 467MB zip 未超过 1.5GB 阈值未分卷），但紧接着的 `verify_full_release.py` 必然失败——因为打包脚本只产出 `dist/full_release_no_pdf_v1/full_release_no_pdf_v1.zip`，从未自动解压到 `artifact_dir`，而 `verify_full_release.py` 默认校验的正是 `artifact_dir`（即 `artifacts/full_release_no_pdf_v1/`，本地从未解压过，必然报"未找到"）。这不是 bug，是两个脚本本来的职责分工：打包脚本只产出待上传的发布物，真正的"下载并解压后校验"流程由 `download_full_release.py` 完成；但用户在本地上传前想先自检，原设计缺一个便捷入口。
- 现在两个脚本都补上了真正的 `argparse`：
  - `package_full_release.py` 新增 `--output <dir>`（覆盖 `local_dist_dir`）和 `--extract-local`（若未分卷，打包后就地解压到 `artifact_dir`，方便立即本地校验，内部实现为 `extract_for_local_verify()`；若已分卷则打印提示改用 `download_full_release.py`，不强行处理分卷合并）。
  - `verify_full_release.py` 新增 `--artifact_dir <dir>`（覆盖默认校验目录，自动按 `artifact_dir/data/chunks/full_chunks.jsonl` 和 `artifact_dir/vector_store/chroma_full` 的相对结构重新拼接 chunks/Chroma 路径，而不是继续套用配置文件里写死的默认路径）。
- 沙箱内用真实命令行参数重新验证（用合成 fixture，文件名已对齐为 `full_paper_inventory.csv`/`full_defect_cards.jsonl`/`full_parameter_cards.jsonl`）：`--output dist/custom_loc` 确认 zip 落到自定义目录；`--extract-local` 打包后自动解压、随后默认 `verify_full_release.py`（无参数）直接 PASS（三项校验全部通过）；`verify_full_release.py --artifact_dir <自定义解压目录>` 同样 PASS。三个场景均按预期工作。
- 用户当前实际可执行的下一步（针对他刚才报告的 467MB 单个 zip、未分卷的真实打包结果）：直接重新运行 `python scripts/package_full_release.py --extract-local`，会重新打包并自动解压到 `artifacts/full_release_no_pdf_v1/`，再运行 `python scripts/verify_full_release.py`（不需要任何参数）即可看到三项校验结果。

## 2026-06-29（追加修复）GitHub Release 打包脚本源文件名修复

- 用户在真实机器上运行 `python scripts/package_full_release.py` 时报错"源文件/目录缺失"，缺失项之一是 `data/metadata/paper_metadata.csv`。排查后发现这是 `package_full_release.py` 自身的命名 bug，不是语料未构建：
  - `SOURCE_FILES` 之前写的是占位文件名 `data/metadata/paper_metadata.csv`、`data/processed/defect_cards.jsonl`、`data/processed/parameter_cards.jsonl`，但 `scripts/run_ingest_full.py`（`FullIngestPaths`）实际产出的文件名是 `data/metadata/full_paper_inventory.csv`、`data/processed/full_defect_cards.jsonl`、`data/processed/full_parameter_cards.jsonl`（均带 `full_` 前缀，且没有 `paper_metadata.csv` 这个名字）。
  - 已修复 `scripts/package_full_release.py` 的 `SOURCE_FILES` 元组与模块 docstring，改为指向这三个真实文件名；`data/chunks/full_chunks.jsonl`、`data/processed/full_paper_cards.jsonl`、`vector_store/chroma_full/` 三项命名本来就正确，未改动。
- 同时确认（未改动，仅记录）：用户机器上 `data/chunks/full_chunks.jsonl` 与四个 `full_*_cards.jsonl` 目前是 0 字节占位（时间戳统一为构建中断的某一时刻），但 `data/interim/full_parsed_docs.jsonl`、`data/processed/full_cleaned_sections.jsonl`、`data/metadata/full_paper_inventory.csv`（896 篇）等中间产物完整存在，说明 PDF 解析/清洗阶段已完成，只是卡片抽取/分块阶段的输出被清空或未跑完；`vector_store/chroma_full` 是指向项目外 `E:\AI_Vector_Stores\...` 的 Windows 目录联接（junction），在当前挂载视图下不可读，需要在真实 Windows 机器上确认该联接目标是否还存在。
- 建议的真实机器修复顺序：`python scripts/run_ingest_full.py --resume`（应可跳过已完成的解析/清洗阶段，只重跑卡片抽取与分块）→ 确认 4 个 `full_*` 文件不再是 0 字节 → 确认 `vector_store/chroma_full` 联接可访问（或重新运行 `python scripts/run_build_index.py --corpus_mode full --reset` 重建）→ 重新运行 `python scripts/package_full_release.py`。

## 2026-06-29 GitHub Release 数据包方案（full_release_no_pdf_v1，单一固定方案）

- 方案固定为 `full_release_no_pdf_v1`，不做多方案分支（区别于 `configs/corpus_config.yaml` 现有的 `dev`/`selected`/`full`/`public_sample`/`public_full_artifact`/`upload_only` 多模式矩阵——本次新增的三个脚本只服务这一个固定方案，没有 `--mode` 之类的分支参数）。
- 新增 `configs/artifact_config.yaml`：固定字段 `artifact_name`/`artifact_source`/`github_owner`（占位，需替换为真实用户名）/`github_repo`/`release_tag`/`artifact_dir`/`chunks_path`/`vector_persist_dir`/`collection_name`，外加打包脚本需要的辅助字段 `max_volume_size_mb`（默认 1500）/`release_manifest_dir`/`local_dist_dir`。`chunks_path`/`vector_persist_dir` 是解压后的**目标**路径（`artifacts/full_release_no_pdf_v1/data/chunks/full_chunks.jsonl` 等），不是打包时的源路径。
- 新增 `scripts/package_full_release.py`：
  - 只打包固定的 8 项内容：`data/chunks/full_chunks.jsonl`、`data/metadata/paper_metadata.csv`、`data/processed/full_paper_cards.jsonl`、`data/processed/defect_cards.jsonl`、`data/processed/parameter_cards.jsonl`、`vector_store/chroma_full/`，以及脚本自己生成的 `release_manifest/MANIFEST.json`、`release_manifest/SHA256SUMS.txt`。
  - 任何源路径（文件名或目录名）包含 `raw_papers` 子串会被 `_assert_not_forbidden()` 拒绝，作为防止 PDF 原文意外混入打包的硬性兜底（需求 #3）。
  - 若必需的源文件/目录缺失（例如全量语料尚未构建完成），`collect_source_files()` 收集全部缺失项后一次性抛出 `FileNotFoundError`，打印清晰的中文错误列表并以退出码 1 终止，不会打印任何论文/chunk 全文（需求 #7）。
  - `split_if_needed()` 按 `max_volume_size_mb`（先乘以 1024*1024 再 `int()` 截断，避免对 <1 的小数值截断成 0 导致源 zip 被静默删除却不产生任何分卷——这是沙箱测试中发现并修复的真实 bug）切分超限的 zip 为 `full_release_no_pdf_v1.zip.001/.002/.003/...`；`max_volume_size_mb<=0` 时显式抛 `ValueError` 并保留原始 zip 不被破坏。
  - 只输出统计信息：打包文件数、源数据总字节数、chunk 数量、唯一 `paper_id` 数量、各分卷文件名与字节数。
- 新增 `scripts/verify_full_release.py`：按 `release_manifest/SHA256SUMS.txt` 逐文件校验 SHA256、校验 `full_chunks.jsonl` 每行是否为合法 JSON 并统计 chunk 数与唯一 `paper_id` 数、用 `chromadb.PersistentClient` 打开 Chroma collection 并报告条目数；三项校验独立报告通过/失败，全部通过才返回退出码 0，全程只打印统计与错误摘要。
- 新增 `scripts/download_full_release.py`：从 `configs/artifact_config.yaml` 读取 `github_owner`/`github_repo`/`release_tag`，优先用 GitHub API 列出 Release 资产并按 `.zip.NNN` 数字后缀排序下载；API 不可用时降级为按约定命名顺序直接尝试 `https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}`，连续失败即停止并提示手动下载（不会无限重试或崩溃）；下载完成后合并分卷为单个 zip、解压到 `artifact_dir`，并以模块方式调用 `verify_full_release.main()` 自动校验，返回码与校验结果一致。
- 更新 `.gitignore`：新增 `!artifacts/README.md`（在已有的 `artifacts/` 整体忽略规则基础上显式保留说明文件）以及 `artifacts/full_release_no_pdf_v1/`、`full_release_no_pdf_v1.zip`、`full_release_no_pdf_v1.zip.*`、`dist/full_release_no_pdf_v1/`、`dist/download_staging/`、`release_manifest/` 等生成产物的忽略规则。
- 新增 `artifacts/README.md`：说明该目录用途、如何运行 `download_full_release.py`/`verify_full_release.py`，以及为何这一个文件被排除在 `artifacts/` 的忽略规则之外。
- 验证（沙箱内构造最小合成数据，未触碰 `data/raw_papers/` 真实 896 篇 PDF）：
  - 用合成的 `full_chunks.jsonl`（3 行）、`paper_metadata.csv`、`full_paper_cards.jsonl`、`defect_cards.jsonl`、`parameter_cards.jsonl` 和真实构建的最小 Chroma collection（2 条向量）跑通 打包 → 解压 → 校验 全流程，三项校验（SHA256/chunks/Chroma）均 PASS。
  - 解析打包出的 zip 内容列表，确认混入的 `data/raw_papers/should_not_be_packaged.pdf` 没有出现在任何分卷中（PDF 排除生效）。
  - 故意删除 `data/metadata/paper_metadata.csv` 后运行打包脚本，确认以退出码 1 终止并打印缺失文件清单，不崩溃、不打印全文。
  - 用 ~14MB 的合成 chunks（高熵文本，避免压缩失真）配合 `max_volume_size_mb=1` 触发真实分卷，产出 7 个 `.001`-`.007` 卷；用 `download_full_release.py` 的 `merge_volumes`/`extract_zip`/`run_verification` 逐一验证合并、解压、校验全部成功。
  - 在测试中发现并修复了 `split_if_needed()` 对极小 `max_volume_size_mb`（如 0.005）的整数截断 bug（修复前会截断为 0 字节读取块，导致源 zip 被删除且不产生任何分卷文件）；修复后该配置值改为先乘以字节换算再截断，并对 `<=0` 显式报错而非静默丢数据。
  - 验证 `download_full_release.py` 在指向占位 `github_owner`（尚未替换为真实用户名）时会优雅降级：GitHub API 调用失败后自动尝试直接下载 URL，最终因资产不存在而以退出码 1 终止，并打印手动下载指引，无未捕获异常。
- 已知限制（有意简化）：`github_owner` 在 `configs/artifact_config.yaml` 中仍是占位符 `YOUR_GITHUB_USERNAME`，需要在真正发布 Release 之前手动替换；`package_full_release.py` 目前要求 `data/processed/full_paper_cards.jsonl` 已经由全量语料构建流程生成，而当前项目这一文件仍是 0 字节占位（全量 Chroma 索引与全量知识卡构建是 `NEXT_TASK.md` 中另一项尚未完成的前置工作），打包脚本会在源文件为空目录/缺失时直接报错退出，不会用 30 篇开发集数据掺假替代。

## 2026-06-29 Long-Conversation Rolling Summarization

- Added `src/agent/conversation_summarizer.py`: a citation-gated, rule-based-first rolling summarizer.
  - `should_trigger_summary(turns, max_turns=8, token_threshold=1400)`：当 in-memory turns 数量超过 8 轮，或全部 turn 文本的 `estimate_tokens()`（复用 `context_manager.py` 现有实现，不重复造轮子）超过 1400 时返回 `True`。
  - `ConversationSummary` dataclass：`user_goal`/`confirmed_materials`/`confirmed_defects`/`confirmed_parameters`/`cited_papers`/`open_questions`/`human_review_items`，与需求字段一一对应；有 `to_dict()`/`from_dict()`。
  - `rule_summarize_turns(turns, previous_summary=None)`：确定性规则摘要，是 `confirmed_*`/`cited_papers`/`human_review_items` 的唯一写入路径——**只有同一轮 turn 同时带 `cited_paper_ids` 时，该轮的 material/defect_type/parameters 才会被提升进 `confirmed_*`**；没有引用支持的实体绝不写入 confirmed 字段，对应的问题改记入 `open_questions`（满足需求 #4，且这是结构性保证，不依赖对 LLM 输出做 schema 校验）。
  - `summarize_turns(turns, previous_summary, llm_client=None)`：先无条件跑 `rule_summarize_turns`；若提供了实现 `SummarizerLLMClient.generate(system_prompt, user_prompt)` 协议（结构化类型，兼容 `answer_generator.py` 里的 `OllamaClient`）的本地 LLM，**只允许它改写 `user_goal` 一句话**，任何异常或本地 LLM 不可用都直接保留规则摘要结果不变（满足需求 #5 的 fallback，无需特判"LLM 不可用"这一条件分支）。
  - `maybe_compress_turns(turns, previous_summary, recent_keep=4, ...)`：触发后把除最近 `recent_keep` 轮之外的所有轮次摘要进 `previous_summary`（增量合并，不是从零重算），返回 `(kept_turns, new_summary, triggered)`。
- 更新 `src/agent/conversation_state.py`：
  - `DEFAULT_MAX_TURNS` 由 6 提升到 20（仅作为兜底硬上限——新的摘要触发条件 `>8` 轮需要在这个上限造成数据丢失之前先生效）。
  - `ConversationTurn` 新增 `need_human_review: bool = False`、`review_reason: str = ""` 两个有默认值的字段，`to_dict()`/`from_dict()` 同步更新且对旧版（无这两个字段）的 JSONL 行向后兼容。
  - `ConversationState.__init__` 新增 `recent_turns_after_summary`/`summary_trigger_turns`/`summary_token_threshold`/`summarizer_llm_client` 参数，新增 `self.summary: ConversationSummary | None`。
  - `add_turn()` 新增 `need_human_review`/`review_reason` 参数；在原有"追加 → 按 max_turns 裁剪 → 持久化"流程后调用新的 `_maybe_compress()`，触发时把 `self.turns`/`self.summary` 替换为压缩结果，从而让超长对话退化为"summary + recent_turns"而不是被硬裁剪静默丢弃（满足需求 #6）。
  - `clear()` 同时重置 `self.summary = None`；新增 `summary_dict()` 辅助方法。
- 更新 `src/app/streamlit_app.py`：侧边栏新增"显示对话摘要" toggle（`show_summary`），勾选后用 `st.expander` + `st.json(conversation.summary_dict())` 展示当前摘要，未触发摘要时显示提示文案；`add_turn()` 调用处新增传入 `need_human_review=bool(result.get("need_human_review"))` 和从 `result.get("review_reasons")`/`limitations` 拼接的 `review_reason`（满足需求 #7）。
- 新增 `tests/test_conversation_summarizer.py`（9 项），覆盖：按轮数/按 token 触发摘要的正反两面、**核心需求 #4 验证**——有引用的轮次实体进入 `confirmed_*` 而无引用的轮次实体绝不进入（改记 open_questions）、`need_human_review` 轮次被收进 `human_review_items`、与已有摘要的增量合并、**核心需求 #8 验证**——12 轮长对话后 `state.summary is not None`、in-memory `turns` 数量始终不超过 `summary_trigger_turns`、`recent_turns` 是对话尾部连续片段且不含早期轮次的原文、早期轮次的引用确实流入了摘要的 `cited_papers`、`clear()` 后摘要被重置。
- 验证：沙箱副本中 `pytest tests/test_conversation_summarizer.py`（9/9 通过）与 `pytest tests/test_conversation_state.py`（10/10 通过，未受字段扩展/`DEFAULT_MAX_TURNS` 改动影响）；排除环境缺少 `fitz` 的 `test_parse_papers.py`/`test_run_ingest_full.py` 与既有的 `scan_papers.py` f-string 语法问题后，全量套件 111/112 通过，唯一失败 `test_pipeline_scripts.py::test_index_pipeline_prefers_configured_local_model` 是 Windows 路径分隔符在 Linux 沙箱下的既有环境差异，与本次改动无关。
- 已知限制（有意简化，未实现）：`ConversationState.load()` 从磁盘重建会话时不会重新触发摘要压缩——重新加载后只会按 `max_turns` 截断原始 turns，`summary` 为空；这是范围内的取舍，需求未要求跨进程重启后的摘要持久化。

## 2026-06-29 Context Manager (Priority-Ordered, Token-Budgeted Prompt Assembly)

- Added `src/agent/context_manager.py`: `build_llm_context(current_query, query_info, conversation_history, conversation_summary, reranked_evidence, token_budget=3200, top_n_evidence=6, recent_turns_window=3)` returns a `ManagedContext(llm_context, context_debug)`.
- `llm_context` exposes exactly six keys: `system_instruction`（复用 `src/rag/prompts.py` 的 `SYSTEM_PROMPT`）、`conversation_summary`、`recent_turns`、`current_query`、`evidence_table`、`risk_rules`。
- 优先级策略：current_query/system_instruction/risk_rules 始终保留；evidence 一次性截断为 `top_n_evidence`（取值范围 5-8，越界自动 clamp）后永不再删；超出 `token_budget` 时先丢弃最近对话轮（从保留窗口最旧的一轮开始），再压缩 `conversation_summary`（按 70% 比例迭代收缩，下限 40 字符），最后才整段丢弃摘要——evidence 始终不受影响。
- 每条 evidence 只保留 `evidence_id`/`paper_id`/`title`/`section_name`/`chunk_type`/压缩后的 `text`（`text_preview`/`evidence_text`/`matched_text` 取第一个非空值，截断到 320 字符并加 `…`），不写入过长全文。
- `context_debug` 记录：`token_budget`、`estimated_tokens_used`、`over_budget`、`evidence_kept_ids`/`evidence_dropped_ids`、`recent_turns_kept_indices`/`recent_turns_dropped_indices`、`older_turns_excluded_indices`、`conversation_summary_compressed`、`conversation_summary_dropped`，可直接审计每一步删了什么、留了什么。
- `render_prompt(llm_context, query_info=None)` 把结构化 `llm_context` 拼成最终用户 prompt 文本（摘要 → 最近对话 → 当前问题 → 结构化 query → evidence 表 → 风险规则 → 中文/`[E编号]` 引用要求）。
- 修改 `src/rag/answer_generator.py`：`AnswerGenerator.generate()` 新增可选参数 `conversation_history`、`conversation_summary`、`token_budget`、`top_n_evidence`（均有默认值，原有 3 个位置参数调用方式不受影响）；prompt 构造统一改为 `build_llm_context()` + `render_prompt()`，不再使用 `src.rag.prompts.build_answer_prompt()` 直接拼接；`GeneratedAnswer` 新增 `context_debug` 字段，便于上层（如 Streamlit 调试面板）展示裁剪过程。
- 新增 `tests/test_context_manager.py`（13 项），覆盖：必需字段集合、空 query/非正 token_budget 报错、evidence 默认截断到 6 条、top_n_evidence 越界 clamp 到 [5,8]、evidence 文本压缩不超 320 字符且不等于原文、最近对话窗口排除更早历史、高风险规则按需追加、极小 token_budget 下证据保持不变而历史被裁剪（核心回归测试）、宽松预算下无任何裁剪、`ManagedContext.to_dict()` 结构、`render_prompt()` 内容覆盖、`estimate_tokens()` 单调性。
- 验证：在隔离的 sandbox 副本中运行 `pytest tests/test_context_manager.py tests/test_rag_answer.py tests/test_conversation_state.py tests/test_streamlit_app.py -q`，31/31 通过；扩大到全量套件（跳过环境缺少 `fitz`/PyMuPDF 的 `test_parse_papers.py`、`test_run_ingest_full.py`，以及沙箱 Python 3.10 下 `scan_papers.py` 一处 f-string 反斜杠语法在 3.12 才支持、与本次改动无关的既有问题）后 102/103 通过；唯一失败 `test_pipeline_scripts.py::test_index_pipeline_prefers_configured_local_model` 是 Windows 路径 `E:\AI_Models\...` 在 Linux 沙箱下的路径分隔符差异，与本次 context_manager/answer_generator 改动无关。
- 未改动：`src/rag/prompts.py` 的 `SYSTEM_PROMPT`/`build_answer_prompt`（后者仍被 `src/langchain_adapters/prompts.py` 使用）；`src/agent/workflow.py` 中 `AgentWorkflow.generate_answer()` 的 3-位置参数调用方式无需修改即可继续工作。

## 2026-06-29 Streamlit Short-Term Conversation Memory

- Added `src/agent/conversation_state.py`: `ConversationState` keeps the most recent `max_turns` turns (default 6) per session and persists every turn as one JSONL line under `data/runtime/conversations/<conversation_id>.jsonl`; `data/runtime/` is already covered by `.gitignore` (no change needed).
- Each stored turn keeps only `user_question`, a short `system_answer_brief`, extracted `key_entities` (defect_type/material/parameters/quality_metric), and up to 5 `cited_paper_ids` — never raw retrieval evidence or full paper text. Question/answer text is sanitized through the existing `sanitize_text()` from `src/agent/memory.py`.
- Added `resolve_followup_query(question, history)`: when a question contains a follow-up marker (那/这个/这种/它/该/此/上述/上面/刚才/之前/继续/呢) and omits an entity category, it fills that category from the most recent turn's entities while leaving explicitly stated entities untouched; otherwise falls back to the plain `rewrite_query()`.
- `src/app/streamlit_app.py`: each session now creates a `conversation_id` and a `ConversationState` in `st.session_state`; `execute_mode()` uses `resolve_followup_query()` once a conversation has turns; a new sidebar "对话记忆" section shows the current turn count and a "清空当前对话记忆" button that clears in-memory turns and deletes the on-disk JSONL.
- Added `tests/test_conversation_state.py` (10 tests): JSONL persistence + in-memory trimming, evidence-text exclusion, clear/delete, reload-from-disk, follow-up marker detection, and the core two-turn case ("保压压力对翘曲有什么影响？" then "那对缩水呢？" correctly inherits `parameters=["packing_pressure"]`), plus explicit-entity-override and no-marker/no-history fallback cases.
- Verified `python3 -m py_compile src/app/streamlit_app.py src/agent/conversation_state.py` and `pytest tests/test_conversation_state.py` (10/10 passed) in a clean sandbox copy. `tests/test_streamlit_app.py` ran 21/22 passing against the same copy; the one failure (`test_retrieval_stats_hide_absolute_persist_path`) is pre-existing and unrelated — it depends on real `vector_store/chroma` + `data/chunks/chunks.jsonl` artifacts that exist on the real machine but were not reproduced in the throwaway sandbox copy used for verification.
- **Bugfix (same day):** the conversation-resolved rewrite from `resolve_followup_query()` was only being attached to `result["query_rewrite"]` for display — `AgentWorkflow`/`LangGraphWorkflow` (普通 RAG) independently recompute a plain, history-blind `rewrite_query(question)` internally, and `run_defect_diagnosis`/`run_method_compare`/`run_retrieval_debug` were all searching with the raw `question` text. So a follow-up like "那对缩水呢？" never actually changed what got retrieved, only what showed up in the "Query Rewrite" debug panel.
- Fixed by: (1) `run_normal_rag` now accepts the precomputed `rewrite` dict and injects it via a fixed `rewriter` callable into both `AgentWorkflow(rewriter=...)` and `LangGraphWorkflow(rewriter=...)` (both already supported this constructor parameter), so `state.normalized_query` — the actual retrieval query — carries the resolved entities; (2) `run_defect_diagnosis`, `run_method_compare`, and `run_retrieval_debug` now search with `rewrite.get("normalized_query") or question` instead of the raw question text; (3) `execute_mode` passes the resolved `rewritten` dict through to all four mode handlers.
- Verified with a scripted smoke test: two-turn conversation ("保压压力对翘曲有什么影响？" then "那对缩水呢？") confirms the literal retrieval query sent to the search layer is `"那对缩水呢 sink_mark/shrinkage packing_pressure"` — i.e. the inherited `packing_pressure` parameter now reaches retrieval, not just the debug display. Full test suite re-run after the fix: same 21/22 result (`test_conversation_state.py` 10/10, `test_streamlit_app.py` 21/22 with the one pre-existing/unrelated environment-only failure noted above).

## 2026-06-29 LangGraph Workflow Backend

- Added a parallel LangGraph workflow while preserving `src/agent/workflow.py` as the classic backend.
- The graph includes query rewrite, retrieval, reranking, answer generation, citation guard, risk check, memory update, and human review nodes.
- Conditional routing sends insufficient evidence to human review, citation failures to one optional revision or human review, and safe answers to memory update and completion.
- The graph uses a configurable recursion limit (`max_steps`, default 12) and a bounded revision count.
- Streamlit now exposes `workflow_backend=classic/langgraph`; classic remains the default.
- LangGraph tests use only mock retrieval, mock LLM, mock memory, and mock review tools; no real full-corpus index is opened.
- Test status: `.venv/Scripts/python.exe -m pytest -q` passed with 82 tests.

## 2026-06-29 LangChain Adapter Layer

- Added a thin `src/langchain_adapters` layer without replacing the existing retrieval, RAG, or Streamlit paths.
- Hybrid retrieval now has an optional LangChain `BaseRetriever` adapter with `invoke()` and `get_relevant_documents()` interfaces.
- Retrieval results convert to LangChain `Document` objects with paper, title, section, score, and chunk metadata.
- The prompt adapter reuses the existing `SYSTEM_PROMPT` and `build_answer_prompt()` implementation, preserving answer instructions and citation format.
- The local Ollama adapter implements the LangChain Runnable interface and retains an offline mock fallback.
- Added LangChain, Core, Community, text-splitter, and Chroma integration dependencies to `requirements.txt` and installed them in `.venv`.
- Real full-corpus smoke passed through Hybrid retrieval -> Document -> prompt -> mock LLM.
- Test status: `.venv/Scripts/python.exe -m pytest -q` passed with 78 tests.

## 2026-06-29 Full Corpus Retrieval Regression

- Added 20 smoke questions covering packing pressure, warpage, shrinkage, weld lines, transparent-part haze, quality prediction, machine learning, optimization, and knowledge graphs.
- Compared Hybrid + rule-rerank top-10 retrieval against the independent dev and full Chroma collections.
- All 40 mode/query runs completed successfully.
- Across the 20 questions, dev retrieved 24 aggregate unique paper IDs and full retrieved 111.
- Mean unique papers per query increased from 5.50 (dev) to 7.90 (full); full was higher on 17 questions, tied on 2, and lower on 1.
- The result confirms that full mode retrieves from a substantially broader paper set, while not claiming that every full result is more relevant.
- Outputs: `data/eval/dev_vs_full_retrieval_compare.csv` and `data/eval/full_corpus_validation_report.md`.

## 2026-06-28 Full Chroma Index Completed

- Built an independent full-mode Chroma index from the current `data/chunks/full_chunks.jsonl`.
- Full collection: `injection_papers_full` at `vector_store/chroma_full`.
- `vector_store/chroma_full` is a junction to `E:/AI_Vector_Stores/injection_molding_rag_agent/chroma_full`; the existing dev junction was preserved.
- Embedding backend/model: `sentence-transformers` with local `E:/AI_Models/BAAI/bge-m3`.
- Embedding dimension: 1024; batch size: 8; build time: 19,062.09 seconds.
- Current full chunks: 43,343; current full vectors: 43,343; Chroma/chunks consistency: true.
- Current full chunks and Chroma contain 559 unique `paper_id` values. The raw directory contains 896 PDFs, so the current full artifact is an audited raw-paper subset rather than complete 896-paper coverage.
- Full build report: `data/logs/full_index_report.md`.
- Full audit outputs: `data/logs/corpus_audit_full_report.md` and `data/logs/corpus_audit_full_stats.csv`.
- The dev index remains unchanged at 1,864 vectors in collection `injection_molding_chunks`.
- `src/index/audit_corpus.py` now pages Chroma metadata reads, avoiding SQLite variable limits on full collections.
- Test status: `python -m pytest -q` passed with 72 tests.

## 2026-06-28 Full Corpus Ingest Completed

- Added `scripts/run_ingest_full.py` with `scan -> parse -> clean -> extract_paper_cards -> build_chunks` orchestration.
- Supports `--limit`, `--resume`, `--force`, and `--workers` (default 2, maximum 8).
- PDF parsing uses one atomic JSONL part per paper under `data/interim/full_parsed_parts`; completed papers are skipped on resume.
- Existing monolithic parsed output was validated by page count and migrated into resumable parts instead of being discarded.
- The 10-paper smoke run passed: 10 papers, 0 failures, 88 sections, and 749 chunks.
- Full run passed: 896 papers, 0 parse failures, 14,898 pages, 7,505 sections, and 71,307 chunks.
- Full knowledge cards: 896 paper cards, 1,484 defect cards, 1,197 method cards, and 2,545 parameter cards.
- `data/chunks/full_chunks.jsonl` is about 81 MB and contains 896 unique `paper_id` values.
- Full ingest report: `data/logs/full_ingest_report.md`.
- Full audit now reports chunks ready and Chroma missing, as expected before full-index construction.
- `configs/corpus_config.yaml` full mode correctly points to `data/chunks/full_chunks.jsonl`, `vector_store/chroma_full`, and `injection_papers_full`.
- The default Streamlit mode was not switched; the verified dev fallback remains active until the full vector index is built.
- Test status: `python -m pytest -q` passed with 72 tests.

## 2026-06-28 Corpus Audit `--mode` Fix

- `src/index/audit_corpus.py` now supports `--mode` for all six corpus modes.
- Explicit `--chunks`, `--persist_dir`, and `--collection` values override the selected mode field by field.
- No-argument audit still inspects the currently runnable dev fallback baseline.
- Explicit mode audits use the mode's configured target paths, without silently switching to the legacy fallback.
- Missing chunks, persist directories, or collections no longer crash the audit; Markdown and CSV outputs mark them as `未构建/缺失`.
- Default outputs are mode-specific: `data/logs/corpus_audit_<mode>_report.md` and `data/logs/corpus_audit_<mode>_stats.csv`.
- Verified `audit_corpus --help`, no-argument audit, `--mode dev`, and `--mode full`.
- Current default dev fallback remains ready: 30 papers, 1,864 chunks, and 1,864 vectors.
- Preferred `dev` and `full` mode targets are currently reported as not built.
- Streamlit sidebar now displays corpus mode, chunks path, collection, paper count, chunk count, and vector count.
- Test status: `python -m pytest -q` passed with 70 tests.

## 2026-06-28 Corpus Mode Configuration

- Added `configs/corpus_config.yaml` with six modes: `dev`, `selected`, `full`, `public_sample`, `public_full_artifact`, and `upload_only`.
- Added `src/config.py` with validated `load_corpus_config()` and optional `CORPUS_MODE` environment override.
- App, BM25, Dense, Hybrid defaults, retrieval debug, corpus audit, chunk building, vector-index building, and index inspection now use the unified corpus configuration.
- Each mode has independent chunks, vector-store, collection, source-paper, upload-policy, and public-demo settings.
- The preferred dev targets are `data/chunks/dev_chunks.jsonl`, `vector_store/chroma_dev`, and `injection_molding_dev`.
- Until those preferred dev artifacts are built, dev mode automatically uses the verified legacy baseline: `data/chunks/chunks.jsonl`, `vector_store/chroma`, and `injection_molding_chunks`.
- Streamlit sidebar now displays corpus mode, chunks path, collection name, paper count, and chunk count.
- Browser verification passed with dev fallback: 30 papers, 1,864 chunks, and all local components ready.
- Command-line BM25, Dense, and Hybrid retrieval remain operational.
- Test status: `python -m pytest -q` passed with 67 tests.

## 2026-06-28 Active Corpus Audit

- Added `src/index/audit_corpus.py` and audited the effective Streamlit retrieval data sources.
- Streamlit/BM25 currently uses `data/chunks/chunks.jsonl`; neither `chunks_path` nor `corpus_mode` is explicitly configured in YAML.
- Active chunks: 1,864 records and 30 unique `paper_id` values.
- Active Chroma: `vector_store/chroma`, collection `injection_molding_chunks`, 1,864 vectors and 30 unique metadata `paper_id` values.
- Chunks and Chroma counts, paper-ID sets, and chunk-ID multisets are fully consistent.
- Source attribution is conclusive: the 30 chunk file names exactly match the 30 PDFs in `data/dev_papers`.
- The active App does not currently use the 100-paper selected set or the full raw directory.
- Current source directories contain 30 dev PDFs, 100 selected PDFs, and 896 raw PDFs.
- Reports: `data/logs/corpus_audit_report.md` and `data/logs/corpus_audit_stats.csv`.
- No PDF full text was read for source attribution.

## 2026-06-28 Full Artifact Packaging Design

- Added `scripts/package_full_artifact.py` for packaging `data/raw_papers`, `data/metadata`, `data/chunks`, and `vector_store` with ZIP64, per-file SHA-256, a content manifest, and optional 1,900 MB release parts.
- Added `scripts/verify_full_artifact.py` for verifying a complete ZIP, a local split-parts descriptor, or an extracted installation.
- Added `scripts/download_full_artifact.py` for direct or multipart downloads, checksum verification, safe extraction, and post-extraction verification.
- Current package preflight expectation is 896 PDFs; packaging requires explicit `--confirm_publication_rights yes`.
- Confirmed the packager can traverse the existing Chroma junction: 42 vector-store files, about 48.85 MB.
- Updated README with packaging, verification, download, hosting, and redistribution-rights guidance.
- No full artifact archive was generated and no paper or vector-store data was added to Git.
- Test status: `python -m pytest -q` passed with 63 tests.

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
