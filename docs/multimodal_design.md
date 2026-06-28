# 注塑企业论文知识库 Agent 的多模态和多类型数据处理方案

## 1. 文档定位

本文档用于说明“注塑企业论文知识库 Agent”在处理论文 PDF、表格、图标题、公式、缺陷图片等多模态和多类型数据时的整体方案。

本项目的知识库主要来源于注塑成型相关学术论文，而不是普通 FAQ。因此，多模态处理的目标不是追求“把所有内容都识别出来”，而是围绕企业真实业务问题，把论文中可追溯、可验证、可引用的证据转化为可检索、可回答、可人工复核的知识单元。

核心原则如下：

1. **证据优先**：所有回答尽量绑定论文来源、页码、段落、表格或图标题。
2. **结构化优先**：表格、实验条件、工艺参数、缺陷类型、质量指标等信息优先转成结构化 chunk。
3. **可追溯优先**：不把模型生成内容当作事实，回答必须能回到原始论文证据。
4. **渐进式多模态**：前期优先做好文本、表格、图标题和图附近段落，不强行做复杂 OCR 和图像理解。
5. **人机协同**：低置信度、高风险、证据冲突、生产参数建议等场景必须进入人工复核。

---

## 2. 论文 PDF 中不同内容类型的处理方案

论文 PDF 中的信息不是单一文本，而是由正文、标题、表格、图标题、公式、参考文献、附录等多种内容构成。不同内容对 RAG 和 Agent 的价值不同，处理方式也应不同。

### 2.1 正文文本处理

正文文本是知识库的基础数据，主要用于回答概念解释、机理分析、工艺影响、缺陷原因、实验结论、方法对比等问题。

处理流程：

1. 使用 PDF 解析工具抽取正文文本。
2. 清理页眉、页脚、页码、版权声明、重复换行、断词和乱码。
3. 按论文结构识别标题层级，例如 Abstract、Introduction、Methods、Results、Discussion、Conclusion。
4. 以“章节 + 段落语义”为基础切分 chunk。
5. 为每个 chunk 增加 metadata。

建议 metadata 字段：

```json
{
  "paper_id": "paper_001",
  "title": "论文标题",
  "authors": ["作者1", "作者2"],
  "year": 2024,
  "journal": "期刊或会议名称",
  "section": "Results and Discussion",
  "page_start": 5,
  "page_end": 6,
  "chunk_type": "text",
  "keywords": ["injection molding", "warpage", "holding pressure"],
  "evidence_level": "paragraph"
}
```

正文文本 chunk 的典型用途：

- 回答“某个缺陷的形成机理是什么”。
- 回答“某个工艺参数对质量有什么影响”。
- 总结某篇论文的研究目标、方法和结论。
- 支撑缺陷诊断、参数影响分析、方法对比等 Agent 工具调用。

### 2.2 表格处理

表格通常包含实验设计、工艺参数范围、材料属性、缺陷指标、模型性能、优化结果等高价值信息。表格不适合简单当作普通段落处理，因为普通文本 chunk 容易丢失行列关系。

表格处理流程：

1. 识别表格标题，例如 Table 1、Table 2。
2. 提取表格所在页码和附近正文说明。
3. 将表格转换为结构化数据。
4. 保留原始表头、单位、行列含义和注释。
5. 为表格生成两种 chunk：
   - **结构化 table chunk**：保留行列数据，适合精确检索和程序处理。
   - **自然语言摘要 chunk**：描述表格展示了什么，适合语义检索。

结构化 table chunk 示例：

```json
{
  "chunk_id": "paper_001_table_002",
  "paper_id": "paper_001",
  "chunk_type": "table",
  "table_id": "Table 2",
  "caption": "Process parameters and their levels used in the experiment",
  "page": 4,
  "columns": [
    {"name": "Parameter", "unit": null, "meaning": "工艺参数名称"},
    {"name": "Level 1", "unit": null, "meaning": "低水平"},
    {"name": "Level 2", "unit": null, "meaning": "中水平"},
    {"name": "Level 3", "unit": null, "meaning": "高水平"}
  ],
  "rows": [
    {
      "Parameter": "Melt temperature",
      "Level 1": "220°C",
      "Level 2": "230°C",
      "Level 3": "240°C"
    },
    {
      "Parameter": "Holding pressure",
      "Level 1": "60 MPa",
      "Level 2": "70 MPa",
      "Level 3": "80 MPa"
    }
  ],
  "nearby_text": "The selected parameters were chosen based on industrial practice and preliminary experiments.",
  "quality_targets": ["warpage", "shrinkage"],
  "defect_terms": ["warpage", "sink mark"],
  "parameter_terms": ["melt temperature", "holding pressure"],
  "evidence_level": "table"
}
```

自然语言摘要 chunk 示例：

```text
Table 2 reports the injection molding process parameters used in the experiment, including melt temperature and holding pressure. The table provides three levels for each parameter and is related to warpage and shrinkage analysis.
```

这种双 chunk 设计的好处是：

- 用户问“保压压力范围是多少”时，可以检索结构化表格。
- 用户问“哪些论文研究了保压压力对翘曲的影响”时，可以检索自然语言摘要。
- Agent 做 evidence table 时，可以直接引用表格字段，而不是让大模型重新猜测表格含义。

### 2.3 图标题处理

图像本身在前期不做复杂视觉理解，但图标题通常是非常有价值的文本证据。论文中的图标题往往明确说明图中展示的是缺陷形貌、实验流程、模型框架、参数影响曲线、误差对比或显著性分析。

图标题处理流程：

1. 识别 Figure、Fig.、图等编号和标题。
2. 保存图标题原文、页码和编号。
3. 抽取图标题前后若干段正文，形成“图标题 + 图附近段落”证据块。
4. 标记图的可能类型，例如方法框架图、实验流程图、缺陷图片、参数影响曲线、性能对比图。
5. 关联论文中的方法、参数、缺陷和质量指标关键词。

图标题 chunk 示例：

```json
{
  "chunk_id": "paper_003_fig_005",
  "paper_id": "paper_003",
  "chunk_type": "figure_caption",
  "figure_id": "Fig. 5",
  "caption": "Effect of holding pressure on warpage under different mold temperatures",
  "page": 7,
  "nearby_paragraphs": [
    "The warpage decreases as holding pressure increases within the tested range.",
    "However, excessive holding pressure may increase residual stress."
  ],
  "figure_type": "parameter_effect_curve",
  "parameter_terms": ["holding pressure", "mold temperature"],
  "quality_targets": ["warpage"],
  "evidence_level": "caption_with_context"
}
```

图标题适合支持的问题：

- 某个参数对缺陷或质量指标的趋势影响。
- 某种算法或工艺优化方法的流程说明。
- 某篇论文是否提供了缺陷形貌、实验曲线或对比结果。
- 论文中某个图是否能作为回答的辅助证据。

### 2.4 图附近段落处理

图附近段落通常解释图中现象，比单独图标题更适合作为 RAG 证据。例如图标题只写“Effect of holding pressure on warpage”，但正文会解释“保压压力升高会降低收缩差异，但过高会引入残余应力”。

处理方式：

1. 将图标题前后 1 到 3 个相关段落与图标题绑定。
2. 如果正文中出现 “as shown in Fig. 5”“Fig. 5 indicates”“如图 5 所示”等引用关系，则优先绑定该段落。
3. 将该证据块标记为 `caption_with_context`，区别于普通正文段落。
4. 回答时优先引用“图标题 + 附近解释段落”，而不是直接让模型凭图标题推断。

这种设计可以解决两个问题：

- 图像暂时不理解，也能利用论文作者对图的解释。
- 避免模型直接“看图说话”产生幻觉。

### 2.5 公式处理

公式在注塑论文中常见于质量预测模型、优化算法、物理模型、损失函数、评价指标和统计分析。公式不一定都需要进入向量检索，但关键公式需要保留。

公式处理流程：

1. 尽量保留公式原文或 LaTeX 表达。
2. 将公式与公式编号、所在段落、变量解释绑定。
3. 对公式进行角色标注，例如评价指标、损失函数、优化目标、物理关系、算法更新规则。
4. 不要求大模型自动推导复杂公式，只要求能说明公式用途和变量含义。

公式 chunk 示例：

```json
{
  "chunk_id": "paper_004_eq_003",
  "paper_id": "paper_004",
  "chunk_type": "formula",
  "equation_id": "Eq. (3)",
  "formula_text": "RMSE = sqrt(1/n * sum((y_i - y_hat_i)^2))",
  "page": 6,
  "formula_role": "evaluation_metric",
  "variables": {
    "y_i": "measured quality value",
    "y_hat_i": "predicted quality value",
    "n": "number of samples"
  },
  "nearby_text": "The root mean square error was used to evaluate prediction accuracy.",
  "evidence_level": "formula_with_context"
}
```

公式处理的边界：

- 可以解释公式含义、变量含义和用途。
- 可以比较不同论文使用了哪些评价指标或优化目标。
- 不自动验证公式推导是否严格正确。
- 不根据公式直接给出未经验证的生产参数。

---

## 3. 为什么前期不强行做复杂 OCR 和图像理解

项目前期不建议把复杂 OCR、图像理解、缺陷图自动识别作为主线功能，原因如下。

### 3.1 论文 PDF 的主要价值首先在文本和结构化证据

注塑论文中的关键知识通常已经由作者写在正文、表格、图标题和图附近段落中。例如：

- 参数范围通常在表格中。
- 缺陷机理通常在正文中。
- 图像结论通常在图附近段落中。
- 算法性能通常在表格和结果分析中。

因此，前期把这些内容解析好，已经可以覆盖大部分论文问答、缺陷诊断、参数影响分析和方法对比需求。

### 3.2 OCR 和图像理解成本高、误差难控

复杂 OCR 和图像理解会引入额外风险：

1. 扫描版 PDF 的 OCR 可能出现字符错误、单位错误、数字错误。
2. 表格 OCR 容易破坏行列关系。
3. 曲线图 OCR 很难稳定识别坐标轴、图例、曲线趋势和数值。
4. 缺陷图片识别需要标注数据和专业评估，不能仅靠通用视觉模型。
5. 图像理解结果如果无法追溯，很难作为企业决策证据。

对于企业级应用，错误的数值、单位或趋势可能导致错误工艺判断，因此前期不应把不稳定视觉识别作为核心证据来源。

### 3.3 项目展示需要突出工程取舍

面试或项目答辩时，不能只说“我做了多模态”，而要说明为什么先做文本、表格、图标题和附近段落，再逐步扩展图像能力。

合理表述是：

> 本项目采用渐进式多模态路线。第一阶段优先处理论文中最稳定、最可追溯的文本、表格、图标题和公式，保证 RAG 证据可靠；第二阶段再扩展缺陷图片识别、图表 OCR 和视觉问答，但这些结果只作为候选证据，不能直接替代工艺工程师判断。

---

## 4. 表格如何转成结构化 chunk

表格是注塑论文知识库中最重要的数据来源之一，尤其适合支持参数建议、实验条件检索、模型性能对比和证据表生成。

### 4.1 表格结构化目标

表格转 chunk 时，不只是把表格转成一段文本，而是要保留以下信息：

1. 表格编号和标题。
2. 所在论文、页码和章节。
3. 表头、单位和注释。
4. 每一行的语义。
5. 表格对应的参数、质量指标、缺陷类型、材料和设备条件。
6. 表格附近正文解释。
7. 表格的证据类型。

### 4.2 推荐字段设计

```json
{
  "chunk_id": "唯一 chunk ID",
  "paper_id": "论文 ID",
  "chunk_type": "table",
  "table_id": "Table 编号",
  "caption": "表格标题",
  "page": "页码",
  "section": "所在章节",
  "columns": "列定义",
  "rows": "行数据",
  "notes": "表格脚注或单位说明",
  "nearby_text": "表格前后解释段落",
  "material": "材料，例如 PP、PC、PMMA、ABS",
  "process": "工艺类型，例如 injection molding、micro injection molding",
  "parameters": "涉及的工艺参数",
  "quality_targets": "质量指标，例如 warpage、shrinkage、weight",
  "defect_terms": "缺陷关键词",
  "method_terms": "算法或优化方法关键词",
  "evidence_level": "table"
}
```

### 4.3 表格 chunk 的粒度

表格 chunk 可以分为三种粒度：

#### 4.3.1 整表 chunk

适合较小表格，例如参数水平表、模型性能对比表。

优点：保留完整上下文。

缺点：如果表格很大，检索时可能噪声较多。

#### 4.3.2 行级 chunk

适合每一行代表一个实验组、一个缺陷、一个参数组合、一个算法结果的表格。

示例：

```json
{
  "chunk_type": "table_row",
  "table_id": "Table 4",
  "row_meaning": "GA optimization result for warpage reduction",
  "method": "Genetic Algorithm",
  "target": "warpage",
  "result": "warpage reduced by 18.5%",
  "source_table_caption": "Optimization results of different algorithms"
}
```

#### 4.3.3 单元格证据 chunk

适合需要精确回答具体数值的问题，例如“PC 数据集中透过率 RMSE 降低了多少”。

单元格 chunk 不建议作为默认方案，因为数量会非常大。它更适合在 evidence_extract_tool 中按需生成。

### 4.4 表格 chunk 的检索方式

表格应同时支持三种检索：

1. **语义检索**：用户问“保压压力对翘曲有什么影响”。
2. **关键词检索**：用户问“holding pressure、warpage、PC”。
3. **结构化过滤**：用户限定材料、缺陷、参数、年份、论文类型。

因此，表格 chunk 既要进入向量库，也要保留结构化字段供过滤和证据整理使用。

---

## 5. 图标题和图附近段落如何作为证据

图像内容本身可能难以直接理解，但图标题和图附近段落是稳定的文本证据，适合作为论文知识库的重要数据类型。

### 5.1 证据组合方式

推荐组合为：

```text
图编号 + 图标题 + 正文引用句 + 图前后解释段落 + 页码 + 章节
```

例如：

```json
{
  "figure_id": "Fig. 6",
  "caption": "Comparison of predicted and measured warpage values",
  "linked_sentences": [
    "As shown in Fig. 6, the proposed model achieves better agreement with the measured values."
  ],
  "nearby_paragraphs": [
    "The prediction errors are smaller under most process conditions, indicating that the model captures the nonlinear relationship between process parameters and warpage."
  ],
  "evidence_summary": "Fig. 6 and its nearby paragraph support that the proposed model improves warpage prediction accuracy."
}
```

### 5.2 图标题证据的适用场景

图标题和附近段落可以用于：

1. 判断某篇论文是否包含某类实验结果。
2. 支撑“趋势类”回答，例如温度升高、压力变化、冷却时间变化对缺陷的影响。
3. 支撑“方法流程类”回答，例如模型结构图、优化流程图、实验流程图。
4. 支撑“结果对比类”回答，例如预测值与真实值对比、不同算法误差对比。
5. 支撑“缺陷形貌类”回答，例如某论文展示了气泡、短射、翘曲、缩痕图片。

### 5.3 图标题证据的边界

不能只凭图标题得出过度结论。例如：

- 图标题写“Effect of melt temperature on shrinkage”，不能直接推断“熔体温度越高收缩一定越大”。
- 必须结合附近段落、表格或正文结论。
- 如果图标题和正文解释不足，应返回“证据不足”，并触发 knowledge_gap_tool 或 human_review_tool。

---

## 6. 缺陷图片识别作为扩展功能

缺陷图片识别可以作为后续扩展能力，但不能作为前期核心功能，也不能直接给出生产结论。

### 6.1 功能定位

缺陷图片识别的定位是：

> 根据用户上传的注塑缺陷图片，给出可能的候选缺陷类型，并结合用户提供的文字描述、材料、设备、工艺参数和论文证据，辅助工程师进一步判断。

它只能输出候选缺陷，不能直接输出“最终缺陷结论”或“生产放行结论”。

### 6.2 输入信息

建议输入包括：

```json
{
  "image": "用户上传的缺陷图片",
  "text_description": "用户对缺陷现象的描述",
  "material": "材料，例如 PC、PP、ABS、PMMA",
  "part_geometry": "制件结构描述",
  "machine_info": "设备信息，可选",
  "mold_info": "模具信息，可选",
  "process_parameters": {
    "melt_temperature": "熔体温度",
    "mold_temperature": "模具温度",
    "injection_speed": "注射速度",
    "holding_pressure": "保压压力",
    "cooling_time": "冷却时间"
  }
}
```

### 6.3 输出结果

输出应采用候选列表，而不是单一结论。

```json
{
  "candidate_defects": [
    {
      "defect_name": "sink mark",
      "confidence": 0.68,
      "visual_evidence": "局部凹陷区域与典型缩痕外观相似",
      "possible_causes": ["保压不足", "冷却不均", "局部壁厚过大"],
      "need_human_confirmation": true
    },
    {
      "defect_name": "short shot",
      "confidence": 0.31,
      "visual_evidence": "边缘区域疑似未完全充满，但图片角度不足",
      "possible_causes": ["注射压力不足", "熔体温度偏低", "排气不良"],
      "need_human_confirmation": true
    }
  ],
  "final_decision": "not_allowed",
  "message": "该结果仅为候选缺陷识别，不能作为最终生产结论。请结合现场工艺记录和工程师复核。"
}
```

### 6.4 与论文知识库结合

缺陷图片识别不能孤立使用，应与论文知识库检索结合：

1. 视觉模型给出候选缺陷。
2. defect_diagnosis_tool 检索该缺陷在论文中的可能原因。
3. parameter_effect_tool 检索相关参数对该缺陷的影响。
4. evidence_extract_tool 汇总论文证据。
5. human_review_tool 在低置信度或高风险情况下触发人工确认。

推荐回答形式：

> 图片外观与“缩痕”和“短射”都有一定相似性，其中缩痕候选置信度较高。但由于缺少制件结构、壁厚分布和工艺参数记录，系统不能给出最终缺陷结论。建议补充材料、熔体温度、保压压力、保压时间、冷却时间和缺陷位置描述，并由工艺工程师确认。

---

## 7. 识别出错时的补救机制

多模态识别一定会出现错误，因此系统必须内置补救机制，而不是把模型输出包装成确定结论。

### 7.1 人工确认

以下情况必须人工确认：

1. 缺陷图片识别置信度低。
2. 多个候选缺陷置信度接近。
3. 图片模糊、角度不佳、光照不均。
4. 缺陷涉及批量生产问题。
5. 系统建议调整关键工艺参数。
6. 论文证据存在冲突。
7. 用户准备把结果用于质量放行或客户报告。

人工确认不是系统失败，而是企业级 Agent 的必要安全设计。

### 7.2 低置信度标记

系统应对结果打置信度标签：

- `high_confidence`：证据充分，但仍需说明来源。
- `medium_confidence`：有一定证据，但需要用户补充信息。
- `low_confidence`：证据不足，只能给候选方向。
- `conflict_evidence`：不同论文或不同证据之间存在冲突。
- `human_review_required`：必须人工复核。

示例：

```json
{
  "answer_confidence": "low_confidence",
  "reason": "图片模糊，且缺少材料、工艺参数和缺陷位置描述",
  "action": "request_more_information_and_human_review"
}
```

### 7.3 重新上传

当图片质量不足时，系统应提示用户重新上传：

- 上传更清晰的局部缺陷图。
- 上传整体制件图，便于判断缺陷位置。
- 上传不同角度、不同光照下的图片。
- 如果有条件，上传尺寸标尺或显微图。
- 上传对应批次的工艺参数记录。

### 7.4 结合文字描述

缺陷图片必须结合文字描述使用。用户至少应补充：

1. 缺陷发生位置。
2. 缺陷外观描述。
3. 材料类型。
4. 制件结构和壁厚变化。
5. 近期是否换料、换模、换设备或换工艺窗口。
6. 当前工艺参数。
7. 缺陷出现频率和批次范围。

图片只能提供外观线索，文字和工艺记录才能提供工程判断依据。

---

## 8. 绝对不能完全信任大模型的场景

在企业级注塑场景中，大模型不能被当作最终决策者。以下场景绝对不能完全信任大模型，必须由专业人员、实验验证或企业流程确认。

### 8.1 安全相关场景

包括设备安全、人员安全、高温高压操作、模具损坏风险、材料分解风险等。

系统可以提供风险提示和参考资料，但不能替代安全规范、现场 SOP 和专业工程师判断。

### 8.2 批量生产参数调整

大模型不能直接决定批量生产参数，例如：

- 熔体温度调整多少。
- 保压压力调整多少。
- 注射速度是否大幅提高。
- 冷却时间是否缩短。
- 是否切换新的工艺窗口。

系统只能给出基于论文证据的候选方向，例如“文献中通常认为提高保压压力可能降低缩痕，但过高可能增加残余应力，实际参数需要结合材料、模具和试模验证”。

### 8.3 质量放行

质量放行必须依据企业质量标准、检测数据、客户要求和质量工程师审核。Agent 不能直接判断某批产品是否合格，更不能替代检测报告。

### 8.4 设备异常

设备异常涉及传感器、液压、电气、机械、控制系统等复杂因素。大模型不能仅凭描述判断设备是否可继续运行。

如果用户描述设备异常，系统应提示：

- 停止高风险操作。
- 查看设备报警代码。
- 联系设备工程师或供应商。
- 结合设备日志、传感器曲线和维护记录排查。

### 8.5 法规标准

涉及国家标准、行业标准、客户标准、医疗器械、汽车零部件、食品接触材料等场景时，不能只依赖大模型总结。

系统应提示用户查阅标准原文、企业规范和客户文件，并由质量或法规人员确认。

---

## 9. 多模态 Agent 工作流设计

### 9.1 论文入库工作流

```text
上传论文 PDF
  ↓
解析文本、表格、图标题、公式
  ↓
清洗页眉页脚、乱码、断行
  ↓
识别章节结构
  ↓
生成 text chunk、table chunk、figure_caption chunk、formula chunk
  ↓
提取 metadata：材料、参数、缺陷、质量指标、方法、页码
  ↓
写入向量库和结构化索引
  ↓
抽样人工检查解析质量
```

### 9.2 用户问答工作流

```text
用户提问
  ↓
判断问题类型：论文问答 / 缺陷诊断 / 参数影响 / 方法对比 / 图片识别
  ↓
调用对应工具检索文本、表格、图标题、公式证据
  ↓
rerank 和证据筛选
  ↓
evidence_extract_tool 生成 evidence table
  ↓
判断置信度和风险等级
  ↓
低风险：生成可引用回答
高风险：回答候选方向 + 触发 human_review_tool
证据不足：触发 knowledge_gap_tool
```

### 9.3 缺陷图片扩展工作流

```text
用户上传缺陷图片 + 文字描述
  ↓
视觉模型给出候选缺陷
  ↓
判断图片质量和置信度
  ↓
检索论文中相关缺陷、参数、材料和原因
  ↓
生成候选缺陷解释和证据表
  ↓
提示用户补充工艺参数和现场信息
  ↓
必须人工确认，不直接给生产结论
```

---

## 10. 多模态数据与工具调用关系

| 数据类型 | 主要来源 | 推荐 chunk 类型 | 适合调用的工具 | 主要用途 | 是否可自动决策 |
|---|---|---|---|---|---|
| 正文文本 | PDF 正文章节 | text | search_papers_tool、defect_diagnosis_tool、parameter_effect_tool | 机理解释、论文问答、参数影响 | 低风险问答可自动回答 |
| 表格 | 参数表、实验表、结果表 | table、table_row | evidence_extract_tool、method_compare_tool、parameter_effect_tool | 参数范围、性能对比、实验条件 | 只能作为证据，不能直接定参数 |
| 图标题 | Fig./Figure caption | figure_caption | search_papers_tool、parameter_effect_tool | 趋势、流程、结果辅助证据 | 不能单独决策 |
| 图附近段落 | 图前后说明 | caption_with_context | evidence_extract_tool | 解释图中趋势和结论 | 可辅助回答 |
| 公式 | 模型公式、指标公式 | formula | method_compare_tool、evidence_extract_tool | 方法解释、指标解释 | 不自动推导生产结论 |
| 缺陷图片 | 用户上传图片 | image_candidate | defect_diagnosis_tool、human_review_tool | 候选缺陷识别 | 必须人工确认 |

---

## 11. 面试时如何讲这个多模态设计

面试中不建议泛泛地说“我做了多模态 RAG”。更好的讲法是从业务问题、数据特点、工程取舍和风险控制四个层次说明。

### 11.1 一句话介绍

> 我的项目不是简单把论文 PDF 切块做问答，而是针对注塑论文中的正文、表格、图标题、公式和缺陷图片设计了分类型处理方案，把可追溯证据转成结构化 chunk，再让 Agent 根据问题类型调用不同工具完成缺陷诊断、参数影响分析和论文方法对比。

### 11.2 说明为什么不是一上来做复杂 OCR

可以这样讲：

> 我没有在第一阶段强行做复杂 OCR 和图像理解，因为企业级知识库最重要的是证据可靠。注塑论文里的核心信息通常已经存在于正文、表格、图标题和图附近段落中，先把这些稳定信息解析好，就能覆盖大部分问答和诊断需求。复杂 OCR、曲线图识别和缺陷图片识别误差较难控制，所以我把它们设计成第二阶段扩展能力，而不是核心决策依据。

### 11.3 说明表格处理亮点

可以这样讲：

> 表格没有简单拼成普通文本，而是转成结构化 table chunk。每个表格保留表号、标题、页码、表头、单位、行数据、附近解释段落，并抽取材料、工艺参数、缺陷和质量指标。这样用户问参数范围、实验条件或模型性能时，系统可以直接返回结构化证据，而不是让大模型凭上下文猜。

### 11.4 说明图标题和图附近段落的价值

可以这样讲：

> 对于图像，我前期不直接让模型看图下结论，而是利用图标题和图附近段落。论文作者通常会在图标题和正文中解释图展示的参数趋势、模型对比或缺陷形貌，所以我把它们绑定成 caption_with_context chunk。这样既利用了图的信息，又保持了证据可追溯。

### 11.5 说明缺陷图片识别的边界

可以这样讲：

> 缺陷图片识别被设计为扩展功能，只输出候选缺陷类型，例如缩痕、短射、飞边、气泡等，并给出置信度和需要补充的信息。它不能直接给生产结论，必须结合材料、制件结构、工艺参数、设备状态和论文证据，并在低置信度或高风险情况下进入人工复核。

### 11.6 说明安全和人工兜底

可以这样讲：

> 我在系统中明确区分“可自动回答”和“必须人工介入”的场景。普通论文问答、方法总结和低风险知识解释可以自动回答；但安全、批量生产参数、质量放行、设备异常和法规标准相关问题不能完全信任大模型，必须触发人工复核或提示用户查阅企业标准和现场 SOP。

### 11.7 面试中的完整回答模板

> 这个项目的多模态设计采用渐进式路线。第一阶段重点处理论文 PDF 中最稳定、最可追溯的正文、表格、图标题和公式。正文用于机理解释和论文问答；表格转成结构化 chunk，用于参数范围、实验条件和模型性能检索；图标题和图附近段落绑定成证据块，用于解释参数趋势、缺陷形貌和方法流程；公式则保留编号、变量解释和上下文，用于方法解释和指标对比。  
> 
> 我没有一开始强行做复杂 OCR 和图像理解，因为企业级 RAG 更看重证据可靠性。OCR 可能识别错数字和单位，图像理解也很难稳定判断曲线趋势或缺陷类型。因此前期先利用论文中已经写清楚的文本证据，后续再扩展缺陷图片识别。缺陷图片识别只给候选缺陷和置信度，不能直接给生产结论，必须结合文字描述、工艺参数和人工确认。  
> 
> 整个设计的重点是让 Agent 不只是“能回答”，而是知道证据来自哪里、哪些问题可以自动回答、哪些问题必须交给工程师复核。这样更符合注塑企业中安全、质量和批量生产的真实约束。

---

## 12. 总结

本项目的多模态处理方案不是盲目追求复杂视觉能力，而是围绕企业论文知识库的核心需求，优先处理可靠、可追溯、可结构化的证据。

第一阶段重点建设：

1. 正文文本解析与语义 chunk。
2. 表格结构化 chunk。
3. 图标题与图附近段落证据块。
4. 公式与上下文绑定。
5. 高风险场景人工复核机制。

第二阶段再扩展：

1. 缺陷图片候选识别。
2. 图表 OCR 和曲线趋势辅助识别。
3. 多模态 evidence table。
4. 缺陷图片、工艺参数和论文证据联合诊断。

最终目标不是让大模型替代工程师，而是让 Agent 成为一个可检索论文证据、可组织结构化知识、可提示风险边界、可辅助工程判断的企业级知识助手。
