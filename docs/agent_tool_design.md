# 注塑企业论文知识库 Agent 工具设计

> 文件路径：`docs/agent_tool_design.md`  
> 项目名称：注塑企业论文知识库 Agent  
> 设计目标：为企业级 RAG + Agent 应用定义可调用工具边界，使系统既能完成论文证据检索、缺陷诊断、参数影响分析、方法对比和结构化证据整理，又能在低置信度、高风险、证据冲突或知识不足时进入人工复核。

---

## 1. 工具设计总原则

本项目的 Agent 不应被设计成“让大模型直接凭经验回答”的系统，而应被设计成“以论文证据为核心、以工具调用为路径、以人工兜底为边界”的企业级知识库应用。

Agent 的基本工作流如下：

```text
用户问题
  ↓
意图识别与风险判断
  ↓
选择工具：检索 / 诊断 / 参数影响 / 方法对比 / 证据抽取
  ↓
生成带来源的中间结果
  ↓
判断置信度、证据一致性、风险等级
  ↓
可自动回答 或 进入人工复核 / 记录知识缺口
```

所有工具都必须遵循以下原则：

1. **证据优先**：凡是涉及论文结论、工艺参数影响、缺陷原因、算法效果，都必须尽量返回论文来源、页码、段落、表格或图号。
2. **不凭空给结论**：当论文库没有足够证据时，Agent 不能强行给出确定性答案，应调用 `knowledge_gap_tool`。
3. **高风险人工兜底**：凡是涉及真实生产参数调整、批量生产决策、模具修改、设备安全、材料更换、停线处理，必须调用 `human_review_tool` 或给出“需要人工确认”的标记。
4. **模型可做信息整理，不可越权做生产决策**：模型可以总结、对比、归纳、生成候选原因和建议方向，但不能直接替代工艺工程师下达最终参数修改指令。
5. **所有输出必须可追溯**：工具输出应保留 `evidence_id`、`paper_id`、`chunk_id`、`source_location`、`confidence_score` 等字段，便于后续审计和面试展示。

---

## 2. 模型自动决策与人工介入边界

### 2.1 模型可以自动完成的任务

以下场景允许 Agent 自动完成，并直接给出结果：

| 场景 | 是否可自动完成 | 说明 |
|---|---:|---|
| 按关键词或语义检索相关论文 | 是 | 只做证据检索，不涉及生产决策。 |
| 总结某篇论文的方法、数据集、结论 | 是 | 必须带来源，不应扩展到论文未支持的结论。 |
| 整理某类缺陷的常见原因 | 是 | 只能作为“可能原因”，不能作为最终诊断。 |
| 分析某个参数对质量的影响趋势 | 是 | 需要标明“基于论文证据”，不能直接给最终设定值。 |
| 对比几种算法或工艺优化方法 | 是 | 适合用于研发选型、论文调研、面试展示。 |
| 生成 evidence table | 是 | 属于结构化整理任务。 |
| 判断当前证据是否不足 | 是 | 可调用 `knowledge_gap_tool` 记录缺口。 |

### 2.2 必须人工介入的任务

以下场景必须进入人工复核，不能让模型独立决策：

| 场景 | 必须人工介入原因 | 应调用工具 |
|---|---|---|
| 用户要求给出可直接用于生产的参数设定值 | 涉及产品质量、设备安全和批量风险 | `human_review_tool` |
| 用户要求立即调整注塑机参数 | 属于现场工艺决策 | `human_review_tool` |
| 证据之间结论冲突 | 可能受材料、模具、设备、实验条件影响 | `human_review_tool` |
| 检索结果置信度低 | 容易误导生产 | `human_review_tool` / `knowledge_gap_tool` |
| 论文库中没有足够证据 | 不能凭经验补全 | `knowledge_gap_tool` |
| 涉及模具结构修改、设备维护、安全风险 | 影响成本和安全 | `human_review_tool` |
| 涉及客户投诉、批量报废、停线决策 | 业务风险高 | `human_review_tool` |

### 2.3 风险等级建议

| 风险等级 | 典型问题 | Agent 处理方式 |
|---|---|---|
| Low | “帮我找保压压力相关论文” | 自动检索并返回证据。 |
| Medium | “保压压力升高对缩水有什么影响？” | 自动分析趋势，但标注适用条件和证据来源。 |
| High | “现在缩水严重，直接告诉我参数怎么改” | 给候选排查方向，同时调用人工复核。 |
| Critical | “是否应该停线、换模具、修改设备参数上限？” | 不给最终决策，必须人工介入。 |

---

# 3. 工具列表

---

## 3.1 search_papers_tool

### 工具名

`search_papers_tool`

### 解决的业务问题

用于从注塑成型相关论文库中按关键词、语义、参数、缺陷、材料、工艺阶段或算法方法检索证据。它是整个 RAG 系统的基础工具，主要解决以下问题：

- 工程师想快速找到某个缺陷、参数或材料相关的论文证据。
- 研发人员想查找某类算法、模型或工艺优化方法的研究基础。
- 新员工想了解某个注塑概念背后的论文来源。
- 后续诊断、参数分析、方法对比工具需要先获得候选证据。

### 输入 schema

```json
{
  "query": "string, 用户问题或检索语句，例如：保压压力 对 缩水 的影响",
  "search_type": "enum: keyword | semantic | hybrid, 默认 hybrid",
  "filters": {
    "material": "string | null, 例如 PC、PMMA、PP、ABS",
    "defect_type": "string | null, 例如 warpage、sink mark、short shot、flash",
    "process_parameter": "string | null, 例如 injection pressure、packing pressure、mold temperature",
    "method_type": "string | null, 例如 machine learning、DOE、GA、PSO、simulation",
    "year_range": ["integer | null", "integer | null"],
    "paper_ids": ["string"]
  },
  "top_k": "integer, 默认 10",
  "rerank": "boolean, 默认 true",
  "return_chunks": "boolean, 默认 true",
  "language": "enum: zh | en | auto, 默认 auto"
}
```

### 输出 schema

```json
{
  "query": "string",
  "search_type": "string",
  "results": [
    {
      "evidence_id": "string",
      "paper_id": "string",
      "title": "string",
      "authors": ["string"],
      "year": "integer | null",
      "chunk_id": "string",
      "source_location": {
        "page": "integer | null",
        "section": "string | null",
        "table": "string | null",
        "figure": "string | null"
      },
      "matched_text": "string",
      "matched_keywords": ["string"],
      "relevance_score": "float, 0-1",
      "rerank_score": "float, 0-1 | null",
      "evidence_quality": "enum: high | medium | low",
      "citation": "string"
    }
  ],
  "overall_confidence": "float, 0-1",
  "warnings": ["string"]
}
```

### 什么时候调用

应在以下场景调用：

- 用户询问某个注塑概念、缺陷、参数、材料、算法或论文结论。
- 其他工具需要先获取论文证据，例如缺陷诊断、参数影响分析、方法对比。
- 用户要求“找论文”“查资料”“有没有相关研究”“论文中怎么说”。
- 用户问题较宽泛，需要先检索再判断是否有足够证据。

### 什么时候不能调用

不应在以下场景单独调用或直接使用其结果下结论：

- 用户要求立即给出生产参数调整方案。
- 用户描述了复杂现场故障，但没有材料、设备、模具、产品结构、缺陷位置等上下文。
- 检索结果明显不足或分数很低。
- 检索结果互相冲突，需要人工判断适用条件。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 无检索结果 | 改写 query，尝试中英文同义词、缺陷别名、参数别名。 |
| 结果过少 | 放宽 filters，提高 top_k，使用 hybrid retrieval。 |
| 结果噪声过多 | 增加参数、材料、缺陷类型过滤，启用 rerank。 |
| 结果冲突 | 标记冲突点，并调用 `human_review_tool`。 |
| 仍无证据 | 调用 `knowledge_gap_tool` 记录知识缺口。 |

### 是否需要人工兜底

通常不需要人工兜底，因为该工具只做检索。但在以下情况需要人工兜底：

- 检索结果将被用于真实生产参数调整。
- 多篇论文结论互相矛盾。
- 检索结果涉及高风险工艺决策。
- 用户要求“直接按检索结果执行”。

---

## 3.2 defect_diagnosis_tool

### 工具名

`defect_diagnosis_tool`

### 解决的业务问题

用于根据注塑缺陷现象，结合论文库证据，检索和整理可能原因、相关工艺参数、质量影响机制和排查方向。该工具面向工艺工程师、质量工程师和新员工，主要解决以下问题：

- 产品出现缩水、翘曲、飞边、短射、银纹、烧焦、熔接痕等缺陷时，快速定位可能原因。
- 把“缺陷现象”映射到“材料、模具、设备、参数、工艺阶段”的可能因素。
- 为工程师提供基于论文证据的排查顺序，而不是凭经验随意试参。

### 输入 schema

```json
{
  "defect_description": "string, 用户描述的缺陷现象",
  "defect_type": "string | null, 标准化缺陷类型，例如 sink mark、warpage、flash、short shot",
  "product_context": {
    "material": "string | null",
    "part_geometry": "string | null",
    "defect_location": "string | null",
    "mold_type": "string | null",
    "machine_info": "string | null"
  },
  "process_context": {
    "injection_speed": "number | string | null",
    "injection_pressure": "number | string | null",
    "packing_pressure": "number | string | null",
    "packing_time": "number | string | null",
    "melt_temperature": "number | string | null",
    "mold_temperature": "number | string | null",
    "cooling_time": "number | string | null"
  },
  "retrieval_top_k": "integer, 默认 10",
  "risk_level": "enum: low | medium | high | critical, 默认 medium"
}
```

### 输出 schema

```json
{
  "standardized_defect_type": "string",
  "possible_causes": [
    {
      "cause_id": "string",
      "cause_category": "enum: material | mold | machine | process_parameter | product_design | environment | unknown",
      "cause_description": "string",
      "related_parameters": ["string"],
      "mechanism_explanation": "string",
      "supporting_evidence_ids": ["string"],
      "confidence_score": "float, 0-1",
      "applicability_conditions": ["string"],
      "limitations": ["string"]
    }
  ],
  "suggested_checks": [
    {
      "check_item": "string",
      "reason": "string",
      "priority": "enum: high | medium | low",
      "requires_human": "boolean"
    }
  ],
  "not_final_decision_notice": "string",
  "need_human_review": "boolean",
  "knowledge_gap": "boolean"
}
```

### 什么时候调用

应在以下场景调用：

- 用户描述了具体缺陷现象，例如“产品有缩水”“边缘飞边严重”“透明件有银纹”。
- 用户希望知道某类缺陷可能由哪些参数、材料或模具因素导致。
- 用户希望得到缺陷排查思路，而不是单纯查论文。
- 用户输入中包含缺陷名称、质量异常、外观问题或尺寸问题。

### 什么时候不能调用

不能在以下场景中直接给最终结论：

- 用户没有提供任何缺陷细节，只说“产品有问题”。
- 用户要求直接给出最终生产参数。
- 缺陷可能涉及设备故障、模具损伤、材料批次异常或安全风险。
- 论文证据不足，无法支持特定缺陷与特定参数之间的关系。
- 缺陷诊断结果将被用于批量生产调整。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 缺陷类型无法识别 | 返回需要补充的信息，例如缺陷图片、位置、材料、工艺参数。 |
| 证据不足 | 调用 `search_papers_tool` 扩展检索；仍不足则调用 `knowledge_gap_tool`。 |
| 多个原因置信度接近 | 给出候选原因排序，并调用 `human_review_tool`。 |
| 证据冲突 | 明确列出冲突来源，不给唯一结论，调用 `human_review_tool`。 |
| 用户要求直接改参数 | 只给排查方向，并调用 `human_review_tool`。 |

### 是否需要人工兜底

需要。缺陷诊断与真实生产强相关，以下情况必须人工兜底：

- `risk_level` 为 high 或 critical。
- `confidence_score` 低于 0.65。
- 证据少于 2 篇论文或来源单一。
- 涉及真实设备参数调整。
- 用户要求把建议直接用于现场试模、量产或客户问题处理。

---

## 3.3 parameter_effect_tool

### 工具名

`parameter_effect_tool`

### 解决的业务问题

用于分析某个注塑工艺参数对产品质量、缺陷、成型稳定性或过程变量的影响。它主要解决以下问题：

- 工程师想知道某个参数升高或降低可能带来什么影响。
- 研发人员想整理论文中关于参数敏感性、参数优化和质量预测的结论。
- 新员工想学习参数与缺陷之间的基本关系。
- 面试展示中需要体现“从论文证据到工艺知识卡”的能力。

### 输入 schema

```json
{
  "parameter_name": "string, 例如 packing pressure、mold temperature、injection speed",
  "target_quality_or_defect": "string | null, 例如 weight、warpage、sink mark、transmittance",
  "material": "string | null",
  "process_stage": "string | null, 例如 filling、packing、cooling",
  "analysis_type": "enum: trend | mechanism | sensitivity | optimization_hint | all, 默认 all",
  "evidence_scope": {
    "paper_ids": ["string"],
    "top_k": "integer, 默认 10",
    "min_relevance_score": "float, 默认 0.5"
  },
  "risk_level": "enum: low | medium | high | critical, 默认 medium"
}
```

### 输出 schema

```json
{
  "parameter_name": "string",
  "target_quality_or_defect": "string | null",
  "effect_summary": "string",
  "effect_direction": "enum: increase_positive | increase_negative | non_monotonic | condition_dependent | unclear",
  "mechanisms": [
    {
      "mechanism": "string",
      "process_stage": "string | null",
      "supporting_evidence_ids": ["string"],
      "confidence_score": "float, 0-1"
    }
  ],
  "parameter_interactions": [
    {
      "related_parameter": "string",
      "interaction_description": "string",
      "supporting_evidence_ids": ["string"]
    }
  ],
  "applicability_conditions": ["string"],
  "unsafe_or_unverified_claims": ["string"],
  "can_auto_answer": "boolean",
  "need_human_review": "boolean"
}
```

### 什么时候调用

应在以下场景调用：

- 用户询问“某个参数对某个质量指标或缺陷有什么影响”。
- 用户希望整理参数影响机制，例如保压压力、模温、熔体温度、冷却时间、注射速度等。
- 用户想知道某个参数属于哪个工艺阶段，可能影响哪些缺陷。
- 用户要求做参数知识卡、参数影响表、参数敏感性总结。

### 什么时候不能调用

不能在以下场景中直接输出生产建议：

- 用户要求“把保压压力调到多少”。
- 用户提供的参数范围超出设备、材料或论文实验范围。
- 参数影响依赖材料、模具结构、产品厚度、浇口位置等条件，而用户未提供这些上下文。
- 论文证据只支持实验趋势，不支持实际量产建议。
- 多个参数存在强耦合，单独分析一个参数会误导判断。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 未检索到参数相关证据 | 改用参数同义词和中英文名称再次检索。 |
| 只找到单篇论文 | 输出低置信度结论，并建议人工确认。 |
| 影响方向不一致 | 标记为 condition_dependent，并列出不同条件。 |
| 用户要求具体数值 | 不直接给数值，转入 `human_review_tool`。 |
| 参数超出论文范围 | 标记为 unsafe_or_unverified_claims。 |

### 是否需要人工兜底

视风险而定。以下情况必须人工兜底：

- 用户要把结论用于真实生产调参。
- 输出涉及具体参数数值、上下限或调整幅度。
- 论文证据之间存在冲突。
- 参数影响与材料、产品结构、设备能力强相关。
- `effect_direction` 为 `condition_dependent` 或 `unclear`，但用户仍要求决策。

---

## 3.4 method_compare_tool

### 工具名

`method_compare_tool`

### 解决的业务问题

用于对比论文中的算法方法、工艺优化方法、缺陷诊断方法或质量预测方法。它主要服务于研发人员、算法工程师和求职面试展示，解决以下问题：

- 不同论文方法之间的输入、输出、模型结构、数据要求、优缺点难以快速比较。
- 企业需要判断某种算法是否适合落地到注塑工艺场景。
- 面试项目展示中需要体现“论文方法对比、技术选型、工程适配”的能力。

### 输入 schema

```json
{
  "methods": [
    {
      "method_name": "string, 例如 GA、PSO、random forest、CNN、RAG、simulation optimization",
      "paper_id": "string | null"
    }
  ],
  "comparison_dimensions": [
    "enum: task | input_data | output | model_structure | optimization_objective | dataset | metrics | advantages | limitations | engineering_feasibility | interpretability"
  ],
  "application_context": {
    "task_type": "enum: quality_prediction | defect_diagnosis | parameter_optimization | process_monitoring | literature_review | unknown",
    "material": "string | null",
    "available_data": ["string"],
    "deployment_requirement": "string | null"
  },
  "retrieval_top_k": "integer, 默认 10",
  "output_format": "enum: table | narrative | both, 默认 both"
}
```

### 输出 schema

```json
{
  "comparison_table": [
    {
      "method_name": "string",
      "task_type": "string",
      "input_data": ["string"],
      "output": ["string"],
      "core_idea": "string",
      "advantages": ["string"],
      "limitations": ["string"],
      "required_data_volume": "enum: low | medium | high | unclear",
      "engineering_feasibility": "enum: high | medium | low | unclear",
      "interpretability": "enum: high | medium | low | unclear",
      "supporting_evidence_ids": ["string"],
      "confidence_score": "float, 0-1"
    }
  ],
  "recommendation_for_context": {
    "best_fit_method": "string | null",
    "reason": "string",
    "not_decision_notice": "string"
  },
  "evidence_conflicts": ["string"],
  "need_human_review": "boolean"
}
```

### 什么时候调用

应在以下场景调用：

- 用户要求“对比几篇论文的方法”。
- 用户询问“GA、PSO、机器学习、仿真优化哪个更适合注塑参数优化”。
- 用户想整理方法综述、技术路线、论文调研表。
- 用户希望把论文方法映射到企业应用场景。
- 用户准备面试，需要讲清楚为什么选择某种技术路线。

### 什么时候不能调用

不能在以下场景中过度给出确定性推荐：

- 企业真实数据条件未知，但用户要求直接选型。
- 方法对比缺少相同数据集、相同指标或相同任务背景。
- 论文实验条件与企业现场差异很大。
- 用户要求直接替代专家做采购、上线或项目立项决策。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 方法名称无法匹配论文 | 先调用 `search_papers_tool` 用同义词扩展检索。 |
| 不同论文指标不可比 | 标注“不可直接横向比较”，只做定性对比。 |
| 缺少工程部署信息 | 输出“工程可行性未知”，并建议补充数据规模、设备条件和实时性要求。 |
| 方法优劣存在冲突 | 调用 `human_review_tool` 或标记为需要专家判断。 |
| 检索结果不足 | 调用 `knowledge_gap_tool`。 |

### 是否需要人工兜底

一般不需要人工兜底，因为该工具主要用于论文调研和技术对比。但在以下情况需要人工兜底：

- 输出将被用于企业真实技术选型、项目立项或上线决策。
- 需要判断算法是否满足生产实时性、安全性、可靠性要求。
- 需要结合企业内部数据质量、设备接口、算力资源和维护成本。

---

## 3.5 evidence_extract_tool

### 工具名

`evidence_extract_tool`

### 解决的业务问题

用于把检索结果、论文片段、诊断结果或方法对比结果整理成结构化 evidence table，便于 RAG 回答、知识卡生成、评测集构建和企业审计。它主要解决以下问题：

- 检索结果是零散文本，不方便用于后续问答。
- 论文中的结论、实验条件、参数、质量指标需要结构化沉淀。
- 企业希望每个回答都能追溯到证据表，而不是只有自然语言总结。
- 面试展示中需要体现“从非结构化论文到结构化知识”的能力。

### 输入 schema

```json
{
  "raw_results": [
    {
      "evidence_id": "string | null",
      "paper_id": "string",
      "chunk_id": "string",
      "text": "string",
      "source_location": {
        "page": "integer | null",
        "section": "string | null",
        "table": "string | null",
        "figure": "string | null"
      }
    }
  ],
  "extract_schema": "enum: defect | parameter_effect | method | experiment | metric | general_evidence",
  "target_fields": ["string"],
  "deduplicate": "boolean, 默认 true",
  "normalize_terms": "boolean, 默认 true",
  "language": "enum: zh | en | bilingual, 默认 zh"
}
```

### 输出 schema

```json
{
  "evidence_table": [
    {
      "evidence_id": "string",
      "paper_id": "string",
      "claim": "string",
      "evidence_text": "string",
      "evidence_type": "enum: conclusion | experiment_result | parameter_setting | mechanism | limitation | metric | dataset | method_description",
      "entities": {
        "materials": ["string"],
        "defects": ["string"],
        "parameters": ["string"],
        "quality_metrics": ["string"],
        "methods": ["string"]
      },
      "source_location": {
        "page": "integer | null",
        "section": "string | null",
        "table": "string | null",
        "figure": "string | null"
      },
      "applicability_conditions": ["string"],
      "limitations": ["string"],
      "confidence_score": "float, 0-1"
    }
  ],
  "deduplication_report": {
    "original_count": "integer",
    "final_count": "integer",
    "merged_items": ["string"]
  },
  "low_confidence_items": ["string"],
  "need_human_review": "boolean"
}
```

### 什么时候调用

应在以下场景调用：

- `search_papers_tool` 返回了多个相关论文片段，需要整理成证据表。
- `defect_diagnosis_tool` 需要把原因、机制、参数和来源结构化。
- `parameter_effect_tool` 需要输出参数影响表。
- `method_compare_tool` 需要输出论文方法对比表。
- 用户要求“整理成表格”“抽取知识卡”“做结构化 evidence table”。

### 什么时候不能调用

不应在以下场景中单独使用：

- 没有任何原始证据文本输入。
- 用户问题需要先检索，而不是直接抽取。
- 输入文本质量很差、来源不明或明显不是论文内容。
- 用户要求模型补全论文中没有的信息。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 原始结果为空 | 返回错误，并建议先调用 `search_papers_tool`。 |
| 字段无法抽取 | 将字段标记为 null，不要编造。 |
| 证据重复 | 启用 deduplicate，合并相同 claim。 |
| 证据质量低 | 放入 low_confidence_items，必要时调用 `human_review_tool`。 |
| 抽取结果冲突 | 保留冲突项，标记 conflict，不强行合并。 |

### 是否需要人工兜底

通常不需要人工兜底，因为它是结构化整理工具。但以下情况需要人工介入：

- 抽取结果将进入企业正式知识库。
- evidence table 将作为生产决策依据。
- 同一 claim 存在多个相互冲突的证据。
- 低置信度条目比例较高。

---

## 3.6 human_review_tool

### 工具名

`human_review_tool`

### 解决的业务问题

用于在低置信度、高风险、证据冲突、生产相关决策、模型不确定或用户越权要求时，把任务转交给人工专家复核。它是企业级 Agent 的安全边界工具，主要解决以下问题：

- 防止模型把论文趋势误用为现场生产决策。
- 防止低置信度诊断误导工程师。
- 对证据冲突、实验条件不一致、参数范围不清的情况进行人工确认。
- 为企业系统保留审计记录，说明为什么需要人工介入。

### 输入 schema

```json
{
  "case_id": "string",
  "trigger_reason": "enum: low_confidence | high_risk | evidence_conflict | production_decision | missing_context | user_requested_final_decision | safety_risk | other",
  "user_question": "string",
  "agent_intermediate_result": "object",
  "evidence_ids": ["string"],
  "risk_level": "enum: low | medium | high | critical",
  "confidence_score": "float, 0-1",
  "required_expert_role": "enum: process_engineer | quality_engineer | mold_engineer | material_engineer | algorithm_engineer | manager | unknown",
  "review_questions": ["string"],
  "deadline": "string | null"
}
```

### 输出 schema

```json
{
  "review_ticket_id": "string",
  "status": "enum: pending | approved | rejected | need_more_info | resolved",
  "assigned_role": "string",
  "review_summary": "string | null",
  "expert_comments": ["string"],
  "approved_actions": ["string"],
  "rejected_actions": ["string"],
  "additional_information_needed": ["string"],
  "final_decision_owner": "string",
  "audit_log": {
    "created_at": "string",
    "trigger_reason": "string",
    "risk_level": "string",
    "evidence_ids": ["string"]
  }
}
```

### 什么时候调用

必须在以下场景调用：

- 证据冲突：不同论文对同一参数或缺陷给出不同趋势。
- 低置信度：整体置信度低于系统阈值，例如 0.65。
- 高风险：用户要求把建议用于真实生产、试模、量产、停线、返工、客户投诉处理。
- 上下文缺失：缺少材料、设备、模具、产品结构、当前参数、缺陷位置等关键信息。
- 参数越界：用户提供或要求的参数范围超出论文实验范围或设备安全范围。
- 模型发现自己只能给出假设，不能给出确定结论。
- 用户要求“直接告诉我怎么调，不要解释”。

### 什么时候不能调用

不应滥用在以下低风险场景：

- 用户只是查论文、做学习、做综述或面试准备。
- 用户明确表示只是做理论分析，不用于生产。
- 证据充分、风险较低、输出只是总结性内容。
- 普通的格式转换、表格整理、术语解释。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 无法分配专家角色 | 标记为 unknown，并默认分配给工艺工程师或知识库管理员。 |
| 人工复核系统不可用 | 返回“需要人工复核但工单创建失败”，保留本地日志。 |
| 专家要求补充信息 | 返回 need_more_info，并列出需要用户补充的字段。 |
| 用户拒绝人工复核 | Agent 只能输出非决策性、低风险解释，不给最终参数。 |
| 复核超时 | 标记 pending，不自动升级为通过。 |

### 是否需要人工兜底

该工具本身就是人工兜底工具。它应作为所有高风险链路的最后安全出口。

---

## 3.7 knowledge_gap_tool

### 工具名

`knowledge_gap_tool`

### 解决的业务问题

用于在论文库没有足够证据、证据覆盖不完整、检索不到相关内容或现有证据无法回答用户问题时，记录知识缺口。它主要解决以下问题：

- 防止模型在论文库不足时编造答案。
- 帮助企业持续建设知识库，知道还缺哪些论文、实验数据或专家规则。
- 为后续数据采集、论文补充、人工标注和评测集扩展提供依据。
- 在面试展示中体现系统具备“不知道就记录缺口”的工程化能力。

### 输入 schema

```json
{
  "gap_id": "string | null",
  "user_question": "string",
  "missing_information_type": "enum: no_relevant_paper | insufficient_evidence | missing_material_context | missing_parameter_range | missing_defect_case | missing_method_detail | conflicting_evidence | other",
  "attempted_queries": ["string"],
  "retrieved_evidence_ids": ["string"],
  "reason_for_gap": "string",
  "suggested_next_actions": [
    "enum: add_papers | ask_expert | collect_factory_cases | add_metadata | improve_chunking | add_synonyms | build_eval_question | other"
  ],
  "priority": "enum: high | medium | low"
}
```

### 输出 schema

```json
{
  "gap_id": "string",
  "status": "enum: recorded | duplicate | updated | ignored",
  "gap_summary": "string",
  "priority": "enum: high | medium | low",
  "recommended_actions": ["string"],
  "linked_questions": ["string"],
  "linked_evidence_ids": ["string"],
  "created_at": "string",
  "owner_role": "enum: knowledge_engineer | process_engineer | quality_engineer | algorithm_engineer | unknown"
}
```

### 什么时候调用

应在以下场景调用：

- `search_papers_tool` 检索不到有效论文证据。
- `defect_diagnosis_tool` 无法支持某个缺陷的原因判断。
- `parameter_effect_tool` 无法找到某参数对某质量指标的可靠影响结论。
- `method_compare_tool` 缺少足够论文或指标不可比。
- 论文库只有单篇弱相关论文，不足以支持结论。
- 用户问题暴露了知识库覆盖盲区，例如新材料、新缺陷、新设备、新工艺。

### 什么时候不能调用

不应在以下场景调用：

- 只是用户问题表达不清，但可以通过追问解决。
- 已有足够证据，只是需要整理或总结。
- 用户问的是项目功能、系统使用方法或普通术语解释。
- 工具失败是由于检索 query 太差，而不是知识库真的缺内容；此时应先改写 query 重新检索。

### 失败时如何处理

| 失败类型 | 处理方式 |
|---|---|
| 缺口已存在 | 返回 duplicate，并关联已有 gap_id。 |
| 缺口描述过宽 | 自动拆分为更具体的子缺口。 |
| 缺少 attempted_queries | 要求上游工具补充检索记录。 |
| 无法判断优先级 | 默认 medium，并交给知识库管理员确认。 |
| 记录系统不可用 | 在本地日志保留 gap 信息，后续补写。 |

### 是否需要人工兜底

需要阶段性人工兜底。知识缺口记录本身可以自动完成，但后续是否补论文、做实验、加专家规则、改 chunk 策略，需要人工确认。

---

# 4. 推荐的 Agent 调用策略

## 4.1 普通论文问答

```text
用户问：有没有关于模温影响翘曲的论文？

调用顺序：
search_papers_tool
  → evidence_extract_tool
  → 生成带引用的回答
```

模型可以自动回答，但必须标明证据来源。

---

## 4.2 缺陷诊断

```text
用户问：产品出现缩水，可能是什么原因？

调用顺序：
defect_diagnosis_tool
  → search_papers_tool
  → evidence_extract_tool
  → 判断风险等级
  → 若涉及真实调参，则 human_review_tool
```

模型可以给出“候选原因”和“排查方向”，但不能直接给最终生产参数。

---

## 4.3 参数影响分析

```text
用户问：提高保压压力会不会减少缩水？

调用顺序：
parameter_effect_tool
  → search_papers_tool
  → evidence_extract_tool
  → 输出影响趋势、机制、适用条件
```

模型可以分析趋势，但必须说明：不同材料、模具结构、产品厚度和参数范围下结论可能不同。

---

## 4.4 方法对比

```text
用户问：GA、PSO 和机器学习方法在注塑参数优化中有什么区别？

调用顺序：
method_compare_tool
  → search_papers_tool
  → evidence_extract_tool
  → 输出方法对比表
```

模型可以自动对比，但如果用户要用于企业真实技术选型，需要人工复核。

---

## 4.5 证据不足

```text
用户问：某种新型材料在特殊模具下如何调参？

调用顺序：
search_papers_tool
  → 若证据不足：knowledge_gap_tool
  → 告知用户当前论文库证据不足
```

模型不能凭空补全答案，应明确记录知识缺口。

---

# 5. 工具选择决策表

| 用户意图 | 首选工具 | 后续工具 | 是否可能人工介入 |
|---|---|---|---|
| 找论文、找证据 | `search_papers_tool` | `evidence_extract_tool` | 通常否 |
| 缺陷原因分析 | `defect_diagnosis_tool` | `search_papers_tool`, `evidence_extract_tool` | 经常需要 |
| 参数影响分析 | `parameter_effect_tool` | `search_papers_tool`, `evidence_extract_tool` | 视风险而定 |
| 方法对比 | `method_compare_tool` | `search_papers_tool`, `evidence_extract_tool` | 技术选型时需要 |
| 结构化证据表 | `evidence_extract_tool` | 无或 `human_review_tool` | 低置信度时需要 |
| 低置信度 / 高风险 / 冲突 | `human_review_tool` | 无 | 必须 |
| 证据不足 / 无法回答 | `knowledge_gap_tool` | `search_papers_tool` | 后续需要 |

---

# 6. 自动决策阈值建议

| 指标 | 建议阈值 | 处理方式 |
|---|---:|---|
| `overall_confidence >= 0.80` 且风险为 Low/Medium | 可自动回答 | 输出证据、结论和适用条件。 |
| `0.65 <= overall_confidence < 0.80` | 谨慎回答 | 标注不确定性，必要时建议人工确认。 |
| `overall_confidence < 0.65` | 不应给确定结论 | 调用 `human_review_tool` 或 `knowledge_gap_tool`。 |
| 支撑论文数少于 2 篇 | 低证据覆盖 | 输出低置信度提醒。 |
| 出现证据冲突 | 高不确定性 | 调用 `human_review_tool`。 |
| 用户要求真实生产执行 | 高风险 | 必须人工复核。 |
| 用户仅用于学习、综述、面试 | 低风险 | 可自动回答，但仍需带证据来源。 |

---

# 7. 输出回答时的安全模板

当 Agent 自动回答时，建议使用以下表达：

```text
根据当前论文库证据，可以得到以下候选结论……
这些结论主要适用于论文中的材料、模具结构和参数范围。
如果要用于真实生产调参，需要由工艺工程师结合现场条件复核。
```

当证据不足时，建议使用以下表达：

```text
当前论文库中没有检索到足够证据支持这个问题的确定回答。
我已将该问题记录为知识缺口，建议后续补充相关论文、现场案例或专家规则。
```

当需要人工复核时，建议使用以下表达：

```text
这个问题涉及真实生产决策或证据存在冲突，不能仅由模型给出最终结论。
我可以先整理候选原因、支持证据和需要人工确认的问题，并提交给对应工程师复核。
```

---

# 8. 面试展示时的讲法

这个工具体系可以在面试中概括为：

> 我没有把 Agent 设计成一个直接给答案的聊天机器人，而是把它设计成一个有工具边界和风险控制的企业级论文知识库 Agent。它先通过 `search_papers_tool` 检索论文证据，再根据任务类型调用缺陷诊断、参数影响分析或方法对比工具，最后通过 `evidence_extract_tool` 形成结构化证据表。如果证据不足，就进入 `knowledge_gap_tool`；如果涉及低置信度、高风险或真实生产决策，就进入 `human_review_tool`。这样既能发挥大模型的信息整理能力，又能避免模型越权替代工程师做生产决策。

---

# 9. 后续可扩展工具

虽然当前版本必须包含 7 个核心工具，但后续可以扩展以下工具：

| 工具名 | 用途 |
|---|---|
| `paper_metadata_tool` | 读取论文题名、作者、期刊、年份、DOI、关键词。 |
| `paper_summary_tool` | 自动生成论文结构化摘要。 |
| `knowledge_card_tool` | 把 evidence table 转换为缺陷卡、参数卡、方法卡。 |
| `eval_question_tool` | 根据论文证据生成评测问题和标准答案。 |
| `citation_check_tool` | 检查回答中的引用是否真实支持结论。 |
| `case_memory_tool` | 记录企业内部缺陷案例和处理经验。 |
| `parameter_safety_tool` | 检查建议参数是否超出设备、材料或工艺安全范围。 |

---

# 10. 总结

本工具设计的核心不是让大模型“更像专家”，而是让大模型在企业知识库中“知道什么时候该检索、什么时候该总结、什么时候该承认不知道、什么时候必须交给人”。

对于“注塑企业论文知识库 Agent”而言，最重要的边界是：

1. **论文问答、证据整理、方法对比可以由模型自动完成。**
2. **真实生产调参、缺陷最终判定、停线返工、模具修改必须人工介入。**
3. **证据不足时必须记录知识缺口，不能编造答案。**
4. **每一个结论都应能回溯到论文证据或人工复核记录。**
