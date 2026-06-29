# 注塑企业论文知识库 Agent 公开完整发布策略

文件路径：`docs/public_full_release_strategy.md`

## 1. 背景与当前状态

本项目名为“注塑企业论文知识库 Agent”，目标是构建一个面向注塑成型论文知识库的企业级 RAG + Agent 应用。当前本地版本已经能够正常运行，使用 Streamlit 启动后可以通过 `localhost:8501` 进行问答、缺陷诊断、工艺参数影响分析、论文方法对比和结构化知识卡查询。

当前本地完整文献库约包含 896 篇注塑成型相关论文。为了让公开 Demo 的效果尽量接近本地完整库，同时又保持 GitHub 仓库轻量、清晰、可维护，本项目采用如下确定方案：

- GitHub 仓库只放代码、配置模板、脚本和说明文档；
- GitHub Release 发布完整数据制品包；
- 本次公开主线使用的数据包命名为 `full_release_no_pdf_v1`；
- `full_release_no_pdf_v1` 不包含 PDF 原文；
- `full_release_no_pdf_v1` 包含已经处理好的 chunks、metadata、paper cards、defect cards、parameter cards 和 Chroma vector store。

该方案的核心思想是：**代码和数据制品分离发布，仓库保持轻量，Demo 效果尽量贴近本地完整知识库。**

---

## 2. 为什么普通 GitHub 仓库只放代码，不放 PDF、chunks 和 vector_store

普通 GitHub 仓库适合托管源代码、配置文件、文档和小规模样例数据，不适合直接存放大规模论文原文、切分后的 chunks 和向量库文件。本项目将普通 GitHub 仓库定位为“代码仓库”，主要原因如下。

### 2.1 保持仓库轻量，降低 clone 成本

本地完整文献库约 896 篇论文，如果将 PDF 原文、处理后的 chunks、metadata 和 Chroma vector store 全部放入 GitHub 仓库，会导致仓库体积迅速膨胀。

这会带来几个问题：

1. 用户执行 `git clone` 时下载速度很慢；
2. 仓库历史记录会越来越大，即使删除大文件，Git 历史中仍可能保留；
3. 每次更新数据制品都会造成大量二进制文件变更；
4. 开发者只想看代码时，也被迫下载完整数据；
5. GitHub 页面浏览、分支切换、CI 检查都会变慢。

因此，普通仓库应该只保留项目运行所需的代码、配置模板、下载脚本、README、docs 和少量最小样例，而不直接承载完整数据制品。

### 2.2 避免将数据制品和源码版本强绑定

chunks、metadata、paper cards、defect cards、parameter cards 和 vector store 本质上是由论文库经过解析、清洗、切分、embedding 和索引构建后生成的数据制品。它们不是源代码，而是构建产物。

如果把这些构建产物直接放入 GitHub 仓库，会导致代码版本和数据版本强行混在一起。未来只要重新构建知识库，即使代码没有变化，也会产生大量文件变更，不利于维护和审查。

更合理的方式是：

- GitHub 仓库管理代码版本；
- GitHub Release 管理数据包版本；
- README 中明确代码版本和数据包版本的对应关系；
- 用户通过下载脚本获取指定版本的数据包。

这样可以让项目结构更清晰，也更接近企业项目中“代码、模型、索引、数据制品分离管理”的工程实践。

### 2.3 vector_store 不适合作为普通 Git 文件长期维护

Chroma vector store 通常包含向量索引、SQLite 文件、parquet 文件或其他持久化文件。它们往往是二进制或半结构化文件，不适合频繁用 Git 追踪。

直接把 vector store 放入仓库会出现这些问题：

1. diff 不可读，无法像代码一样审查具体变化；
2. 每次重新 embedding 或重建索引都会生成大文件变更；
3. 容易产生平台兼容、路径配置和锁文件问题；
4. 不适合多人协作开发时频繁提交。

因此，vector store 更适合作为 Release 数据包的一部分，而不是普通 Git 仓库内容。

---

## 3. 为什么由 GitHub Release 分发 `full_release_no_pdf_v1`

GitHub Release 适合发布相对稳定、体积较大、按版本管理的构建产物。对于本项目来说，`full_release_no_pdf_v1` 就是一个公开 Demo 使用的完整数据制品包，因此适合放在 GitHub Release 中分发。

### 3.1 Release 天然适合版本化数据包

`full_release_no_pdf_v1` 是一个明确的数据版本。它对应一批固定的文献解析结果、chunks、metadata、知识卡和向量库。将其放入 Release 后，可以清楚地表达：

- 这是第一个公开完整数据包；
- 该数据包不包含 PDF 原文；
- 该数据包用于复现接近本地完整库的 Demo 效果；
- 后续可以继续发布 `full_release_no_pdf_v2`、`full_release_no_pdf_v3` 等版本。

这种方式比直接把数据制品提交到主分支更清晰，也方便用户按需下载。

### 3.2 用户可以先 clone 代码，再按需下载数据

采用 Release 分发后，用户的使用流程变成：

1. 先 `git clone` 轻量代码仓库；
2. 安装依赖；
3. 运行下载脚本；
4. 解压或自动放置 `full_release_no_pdf_v1`；
5. 启动 Streamlit Demo。

这样既不会影响普通开发者查看代码，也能让需要完整 Demo 的用户获取接近本地完整库的效果。

### 3.3 Release 便于说明数据包内容和边界

在 Release 页面中，可以专门写清楚数据包包含什么、不包含什么、适合什么场景、不适合什么场景。例如：

- 包含：chunks、metadata、paper cards、defect cards、parameter cards、Chroma vector store；
- 不包含：PDF 原文；
- 用途：公开 Demo、RAG 问答、缺陷诊断、参数影响分析、方法对比；
- 限制：不能替代完整论文阅读，不能作为生产参数的唯一依据；
- 版权说明：仅发布处理后的知识库数据制品，不直接分发 PDF 原文。

这比把数据散落在仓库目录里更利于用户理解。

---

## 4. 为什么不把 PDF 原文放入本次公开主线

本次公开主线明确不发布 PDF 原文，只发布 `full_release_no_pdf_v1`。原因主要包括体积、下载体验、版权说明复杂度和实际 RAG 效果四个方面。

### 4.1 PDF 原文体积大，容易导致仓库和下载包过重

完整论文库约 896 篇 PDF，原文文件体积通常远大于处理后的 metadata 或知识卡。如果直接公开 PDF 原文，会显著增加数据包体积，导致下载、解压、备份和迁移成本变高。

对于一个面向求职展示和公开 Demo 的项目，过大的数据包会降低用户体验。面试官或其他开发者往往只希望快速看到项目能否运行、RAG 效果如何、Agent 工作流是否清晰，而不是先下载大量原始 PDF。

### 4.2 PDF 原文版权说明复杂

论文 PDF 可能来自不同出版商、会议、期刊、作者主页或开放平台。即使部分论文可以公开访问，也不代表项目仓库可以直接二次分发 PDF 原文。

如果将 PDF 原文放入公开主线，需要逐篇确认授权方式、开放许可、下载来源和再分发边界，维护成本很高，也容易引入版权风险。

因此，本项目公开主线不直接分发 PDF 原文，而是发布用于 Demo 的处理后数据制品，并在 README 和 Release 说明中明确：用户如需阅读论文原文，应通过论文 DOI、出版社页面、作者主页或学校/机构订阅渠道自行获取。

### 4.3 RAG Demo 的主要效果依赖 chunks 和 vector_store

对于本项目的问答、缺陷诊断和参数分析功能来说，运行时最关键的是：

- 已清洗和切分的文本 chunks；
- 每篇论文的 metadata；
- 可检索的 Chroma vector store；
- 结构化 paper cards；
- defect cards；
- parameter cards。

RAG 检索和生成回答时，并不一定需要实时读取 PDF 原文。只要 chunks 和 vector store 已经由完整论文库构建完成，公开 Demo 就可以较好复现本地完整库的问答效果。

因此，`full_release_no_pdf_v1` 虽然不包含 PDF 原文，但仍然保留了 RAG 运行所需的主要知识表示和检索索引。

### 4.4 PDF 原文可以作为本地私有增强，不作为公开主线

本项目可以在本地保留完整 PDF 文献库，用于继续构建、调试、增量更新和人工核验。但公开版本不把 PDF 原文作为默认依赖。

公开版本的定位是：

- 让外部用户能够快速运行完整 Demo；
- 展示基于大规模论文库构建 RAG Agent 的工程能力；
- 避免公开分发 PDF 原文带来的体积和版权问题；
- 保留后续本地私有扩展能力。

---

## 5. `full_release_no_pdf_v1` 固定目录结构

`full_release_no_pdf_v1` 解压后建议放置在项目根目录下的 `data/full_release_no_pdf_v1/` 中。固定目录结构如下：

```text
data/
└── full_release_no_pdf_v1/
    ├── README.md
    ├── manifest.json
    ├── checksums.sha256
    │
    ├── full_chunks/
    │   ├── chunks.jsonl
    │   ├── chunk_index.csv
    │   └── chunk_stats.json
    │
    ├── metadata/
    │   ├── papers_metadata.csv
    │   ├── papers_metadata.jsonl
    │   ├── source_statistics.json
    │   └── field_schema.json
    │
    ├── paper_cards/
    │   ├── paper_cards.jsonl
    │   └── paper_cards_index.csv
    │
    ├── defect_cards/
    │   ├── defect_cards.jsonl
    │   └── defect_taxonomy.json
    │
    ├── parameter_cards/
    │   ├── parameter_cards.jsonl
    │   └── parameter_taxonomy.json
    │
    └── vector_store/
        └── chroma/
            ├── chroma.sqlite3
            ├── index/
            └── collections.json
```

各目录含义如下。

### 5.1 `README.md`

数据包说明文件，写清楚：

- 数据包名称；
- 数据包版本；
- 是否包含 PDF；
- 适用的代码版本；
- 使用方式；
- 数据边界；
- 版权说明；
- 常见问题。

### 5.2 `manifest.json`

数据包清单文件，记录数据包的基本信息，例如：

```json
{
  "release_name": "full_release_no_pdf_v1",
  "contains_pdf": false,
  "paper_count": 896,
  "embedding_model": "bge-m3",
  "vector_store": "chroma",
  "created_at": "YYYY-MM-DD",
  "compatible_app_version": "v1.x",
  "description": "Full public demo data package without original PDF files."
}
```

### 5.3 `checksums.sha256`

用于校验数据包完整性。下载脚本可以在解压后读取该文件，对关键文件进行哈希校验，避免文件损坏或下载不完整。

### 5.4 `full_chunks/`

存放由完整论文库解析、清洗和切分得到的文本块。RAG 检索时，chunks 是最核心的证据来源之一。

建议包含：

- `chunks.jsonl`：每行一个 chunk；
- `chunk_index.csv`：chunk 与论文、页码、章节、类型之间的索引关系；
- `chunk_stats.json`：chunk 数量、平均长度、来源分布等统计信息。

### 5.5 `metadata/`

存放论文元数据，例如标题、作者、年份、期刊、关键词、DOI、来源、材料、工艺、任务类型等。

建议包含：

- `papers_metadata.csv`：方便人工查看；
- `papers_metadata.jsonl`：方便程序读取；
- `source_statistics.json`：来源统计；
- `field_schema.json`：字段解释。

### 5.6 `paper_cards/`

存放每篇论文的结构化知识卡，用于快速展示论文贡献、方法、数据、结论和适用场景。

### 5.7 `defect_cards/`

存放围绕注塑缺陷整理的结构化知识卡，例如翘曲、缩水、熔接痕、短射、飞边、气泡、烧焦等缺陷的可能原因、相关参数、论文证据和风险提示。

### 5.8 `parameter_cards/`

存放围绕工艺参数整理的结构化知识卡，例如熔体温度、模具温度、注射速度、保压压力、冷却时间等参数对质量、缺陷和成型稳定性的影响。

### 5.9 `vector_store/chroma/`

存放 Chroma 向量库文件。应用启动时应直接加载该目录下的持久化向量库，而不是要求用户重新 embedding。

---

## 6. README 中如何写下载 full release 数据包的步骤

项目 README 中建议增加“下载完整公开数据包”章节。示例内容如下。

```markdown
## 下载完整公开数据包

本仓库只包含代码、配置模板、脚本和文档，不直接包含完整论文库、chunks 或向量库。

为了让公开 Demo 效果尽量接近本地完整库，请下载 GitHub Release 中的完整数据制品包：

- Release 名称：`full_release_no_pdf_v1`
- 是否包含 PDF 原文：否
- 包含内容：full_chunks、metadata、paper_cards、defect_cards、parameter_cards、Chroma vector_store

### 方式一：使用脚本自动下载

在项目根目录运行：

```bash
python scripts/download_full_release.py
```

脚本会自动完成：

1. 下载 `full_release_no_pdf_v1`；
2. 校验文件完整性；
3. 解压到 `data/full_release_no_pdf_v1/`；
4. 检查必要文件是否存在。

### 方式二：手动下载

1. 打开本项目 GitHub Release 页面；
2. 找到 `full_release_no_pdf_v1`；
3. 下载压缩包；
4. 解压到项目根目录下的：

```text
data/full_release_no_pdf_v1/
```

解压完成后，目录中应包含：

```text
full_chunks/
metadata/
paper_cards/
defect_cards/
parameter_cards/
vector_store/
manifest.json
checksums.sha256
```

### 启动应用

安装依赖后运行：

```bash
streamlit run app.py
```

或：

```bash
streamlit run src/app.py
```

具体启动命令以本仓库实际入口文件为准。
```

README 中还应明确提醒：

```markdown
如果缺少 `data/full_release_no_pdf_v1/`，应用不会自动切换到其他小型样例库。请先运行：

```bash
python scripts/download_full_release.py
```

这样可以避免用户误以为已经在使用完整公开知识库。
```

---

## 7. 启动逻辑设计：缺少 full release 时不自动切换到其他库

为了保证公开 Demo 的一致性，本项目启动时应优先检查 `full_release_no_pdf_v1` 是否存在。

### 7.1 启动检查逻辑

应用启动时执行以下检查：

1. 检查 `data/full_release_no_pdf_v1/manifest.json` 是否存在；
2. 检查 `data/full_release_no_pdf_v1/full_chunks/chunks.jsonl` 是否存在；
3. 检查 `data/full_release_no_pdf_v1/metadata/papers_metadata.csv` 是否存在；
4. 检查 `data/full_release_no_pdf_v1/vector_store/chroma/` 是否存在；
5. 检查 Chroma collection 是否可以正常加载。

如果检查通过，则加载 base full release collection。

如果检查不通过，则页面直接提示用户下载完整数据包，不自动切换到其他库。

### 7.2 为什么不自动切换到其他库

不自动切换的原因是避免 Demo 效果被误解。

如果缺少 `full_release_no_pdf_v1` 时自动切换到一个很小的样例库，用户可能会看到较差的检索结果、较少的论文证据和较弱的缺陷诊断效果，从而误以为这是完整系统能力。

因此，本项目采用更明确的策略：

- 缺少完整数据包时，页面直接提示；
- 不静默降级；
- 不自动切换到 demo_small；
- 不自动重建空库；
- 不让用户在未知状态下进入低质量问答。

### 7.3 页面提示文案建议

当缺少 `full_release_no_pdf_v1` 时，Streamlit 页面可以显示如下提示：

```text
未检测到完整公开数据包 full_release_no_pdf_v1。

本项目的公开 Demo 依赖 GitHub Release 中的数据制品包：
data/full_release_no_pdf_v1/

请先在项目根目录运行：

python scripts/download_full_release.py

该数据包不包含 PDF 原文，但包含已经构建好的 chunks、metadata、paper cards、defect cards、parameter cards 和 Chroma vector store，可用于复现接近本地完整库的 RAG Demo 效果。

为避免误用小型样例库，本应用不会自动切换到其他知识库。
```

### 7.4 推荐伪代码

```python
from pathlib import Path
import streamlit as st

FULL_RELEASE_DIR = Path("data/full_release_no_pdf_v1")

REQUIRED_PATHS = [
    FULL_RELEASE_DIR / "manifest.json",
    FULL_RELEASE_DIR / "full_chunks" / "chunks.jsonl",
    FULL_RELEASE_DIR / "metadata" / "papers_metadata.csv",
    FULL_RELEASE_DIR / "vector_store" / "chroma",
]

def check_full_release_ready() -> bool:
    return all(path.exists() for path in REQUIRED_PATHS)

if not check_full_release_ready():
    st.error("未检测到完整公开数据包 full_release_no_pdf_v1。")
    st.code("python scripts/download_full_release.py", language="bash")
    st.info("为避免误用小型样例库，本应用不会自动切换到其他知识库。")
    st.stop()

# 检查通过后再加载完整公开知识库
```

---

## 8. 上传新论文功能：作为增量补充，不污染 base full release collection

公开 Demo 需要支持用户上传新论文 PDF，但上传内容不应直接写入 base full release collection。推荐采用“base collection + upload collection”的双集合设计。

### 8.1 base full release collection

base full release collection 来自 `full_release_no_pdf_v1`，是公开 Demo 的基础知识库。它具有以下特点：

- 来源固定；
- 版本固定；
- 可复现；
- 不应被用户上传内容直接修改；
- 对应 Release 中发布的数据制品；
- 用于展示接近本地完整库的基础能力。

### 8.2 upload collection

upload collection 用于保存用户运行时上传的新论文 PDF 解析结果。它具有以下特点：

- 来源是用户本地上传；
- 可以增量写入；
- 可以清空或重建；
- 不影响 base full release collection；
- 适合作为本地个性化扩展；
- 不进入 GitHub Release 的固定数据包。

### 8.3 检索时如何合并结果

用户提问时，系统可以同时检索两个 collection：

1. base full release collection；
2. upload collection。

然后对两部分结果进行统一 rerank 或排序融合。推荐策略如下：

```text
用户问题
  ↓
检索 base full release collection
  ↓
检索 upload collection
  ↓
合并候选证据
  ↓
rerank
  ↓
生成带来源标识的回答
```

回答中应明确标识证据来源，例如：

- `[base]`：来自 `full_release_no_pdf_v1`；
- `[upload]`：来自用户上传论文。

这样既能保证公开基础库稳定，又能让用户上传的新论文参与问答。

### 8.4 为什么不能污染 base collection

如果用户上传的 PDF 直接写入 base full release collection，会产生以下问题：

1. 破坏 Release 数据包的可复现性；
2. 难以判断某条证据来自公开数据包还是用户上传；
3. 用户多次上传后，base collection 状态不可控；
4. 出错后难以恢复；
5. 不利于调试和演示；
6. 不利于未来升级 `full_release_no_pdf_v2`。

因此，上传功能必须作为增量补充，而不是直接修改基础知识库。

### 8.5 推荐目录结构

```text
data/
├── full_release_no_pdf_v1/
│   └── vector_store/
│       └── chroma/
│
└── uploads/
    ├── raw_pdfs/
    ├── parsed_chunks/
    ├── metadata/
    └── vector_store/
        └── chroma_upload/
```

---

## 9. 配置建议

建议在 `configs/app_config.yaml` 中显式配置 full release 路径和 collection 名称。

示例：

```yaml
data:
  base_release_name: "full_release_no_pdf_v1"
  base_release_dir: "data/full_release_no_pdf_v1"
  require_full_release: true
  allow_auto_fallback: false

vector_store:
  provider: "chroma"
  base_collection_name: "injection_papers_full_release_no_pdf_v1"
  upload_collection_name: "injection_papers_upload"
  base_persist_dir: "data/full_release_no_pdf_v1/vector_store/chroma"
  upload_persist_dir: "data/uploads/vector_store/chroma_upload"

upload:
  enabled: true
  raw_pdf_dir: "data/uploads/raw_pdfs"
  parsed_chunks_dir: "data/uploads/parsed_chunks"
  metadata_dir: "data/uploads/metadata"
  write_to_base_collection: false
```

其中最关键的配置是：

```yaml
require_full_release: true
allow_auto_fallback: false
write_to_base_collection: false
```

这三个配置共同保证：

- 没有完整数据包就不启动完整 Demo；
- 不静默切换到小库；
- 用户上传内容不会污染基础知识库。

---

## 10. `download_full_release.py` 脚本职责

`download_full_release.py` 是公开 Demo 的关键脚本，建议放在：

```text
scripts/download_full_release.py
```

该脚本应承担以下职责：

1. 定义 Release 数据包下载地址；
2. 下载 `full_release_no_pdf_v1` 压缩包；
3. 支持断点或重复运行时跳过已完成文件；
4. 解压到 `data/full_release_no_pdf_v1/`；
5. 校验 `checksums.sha256`；
6. 检查关键目录是否存在；
7. 输出下一步启动命令。

脚本运行成功后的终端提示建议如下：

```text
full_release_no_pdf_v1 下载并解压完成。

数据目录：
data/full_release_no_pdf_v1/

已检测到：
- full_chunks
- metadata
- paper_cards
- defect_cards
- parameter_cards
- vector_store/chroma

现在可以启动应用：

streamlit run app.py
```

如果下载失败，应提示用户手动去 GitHub Release 页面下载，而不是直接创建空目录或切换小型库。

---

## 11. `.gitignore` 建议

由于数据包不进入普通 Git 仓库，应在 `.gitignore` 中忽略以下目录：

```gitignore
# Full release data packages
data/full_release_no_pdf_v1/
data/full_release_no_pdf_v*/
data/releases/

# Local uploaded PDFs and user-built indexes
data/uploads/
data/raw_pdfs/
data/local_papers/

# Vector stores
**/vector_store/
**/chroma/
*.sqlite3
*.parquet

# Large generated knowledge artifacts
data/**/chunks.jsonl
data/**/paper_cards.jsonl
data/**/defect_cards.jsonl
data/**/parameter_cards.jsonl
```

但可以保留小型样例数据，例如：

```text
data/sample/
```

如果项目需要 sample 数据用于单元测试或最小运行示例，应确保 README 明确说明 sample 数据不是完整 Demo 数据。

---

## 12. Release 页面说明建议

GitHub Release 页面建议使用如下说明模板。

```markdown
# full_release_no_pdf_v1

这是“注塑企业论文知识库 Agent”的第一个公开完整数据制品包。

## 包含内容

- full_chunks
- metadata
- paper_cards
- defect_cards
- parameter_cards
- Chroma vector_store

## 不包含内容

- 不包含 PDF 原文
- 不包含商业或内部私有文件
- 不包含用户上传后的本地增量数据

## 使用方式

请将压缩包解压到：

```text
data/full_release_no_pdf_v1/
```

或在项目根目录运行：

```bash
python scripts/download_full_release.py
```

## 说明

本数据包用于复现接近本地完整论文库的公开 RAG Demo 效果。由于版权和体积原因，本 Release 不分发 PDF 原文。用户如需查看论文原文，请通过 DOI、出版社页面、作者主页或机构订阅渠道自行获取。

## 边界

本项目输出的缺陷原因、参数影响和工艺建议仅作为论文证据辅助分析，不可直接作为批量生产参数、质量放行、设备安全或法规合规的唯一依据。
```

---

## 13. 面试时怎么讲

面试时可以用下面这段话概括本设计。

> 我的项目本地有约 896 篇注塑论文，完整库已经可以支撑 Streamlit RAG Demo。公开 GitHub 时，我没有把 PDF、chunks 和向量库直接塞进仓库，而是把代码和数据制品分离：仓库只放代码、配置、脚本和文档，完整公开数据包通过 GitHub Release 发布，命名为 `full_release_no_pdf_v1`。这个数据包不包含 PDF 原文，但包含 full chunks、metadata、paper cards、defect cards、parameter cards 和 Chroma vector store，所以公开 Demo 的检索效果可以尽量接近本地完整库，同时避免仓库过大、clone 过慢和 PDF 版权说明复杂的问题。

进一步展开时，可以这样说：

1. **工程可维护性**  
   代码仓库保持轻量，便于 clone、review、维护和版本管理。数据制品作为 Release 单独发布，避免大文件污染 Git 历史。

2. **Demo 效果一致性**  
   公开 Demo 不依赖小型 toy 数据，而是要求用户下载 `full_release_no_pdf_v1`。如果缺少该数据包，系统会提示运行 `download_full_release.py`，不会自动切换到其他小库，避免用户误判系统效果。

3. **版权和体积控制**  
   PDF 原文体积大，版权来源复杂，所以本次公开主线不分发 PDF。RAG 运行主要依赖已经构建好的 chunks 和 vector store，因此不发布 PDF 也能展示核心检索和问答能力。

4. **增量上传隔离**  
   用户上传的新论文会进入 upload collection，不会污染 base full release collection。检索时可以同时查 base 和 upload，再统一 rerank。这样既保证基础 Demo 可复现，又支持本地个性化扩展。

5. **企业级思路**  
   这个设计体现了企业项目中常见的代码、数据、索引、模型制品分离管理思想。代码走 Git，数据制品走 Release 或对象存储，用户增量数据单独管理，基础知识库保持稳定可复现。

面试中可以强调一句：

> 这个公开发布方案不是简单地把本地文件夹上传到 GitHub，而是按照工程化思路把“源码仓库、公开数据制品、用户增量数据、版权边界、启动校验逻辑”拆开管理，从而兼顾 Demo 效果、仓库可维护性和公开发布风险控制。

---

## 14. 总结

本项目最终采用如下公开发布策略：

1. GitHub 仓库只放代码、配置、脚本和文档；
2. GitHub Release 发布 `full_release_no_pdf_v1`；
3. `full_release_no_pdf_v1` 包含完整公开 Demo 所需的数据制品；
4. 本次公开主线不包含 PDF 原文；
5. RAG 效果主要由 chunks、metadata、知识卡和 Chroma vector store 保证；
6. 应用启动时必须检查 full release 是否存在；
7. 缺少 full release 时页面提示用户运行 `download_full_release.py`；
8. 系统不自动切换到其他小型库；
9. 用户上传 PDF 作为增量数据进入 upload collection；
10. upload collection 不污染 base full release collection。

该方案能够在公开展示时兼顾三点：

- Demo 效果尽量接近本地完整库；
- GitHub 仓库保持轻量、清晰、可维护；
- PDF 原文版权和大文件分发风险得到控制。
