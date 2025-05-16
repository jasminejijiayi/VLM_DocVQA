# Lab3：基于视觉语言模型的视觉问答系统

## 1. 背景介绍

视觉语言模型（Vision Language Models, VLM）作为多模态人工智能的核心技术，在视觉问答（Visual Question Answering, VQA）任务中展现出强大的跨模态理解能力。文档视觉问答（Document VQA）作为VQA的重要分支，在金融票据处理、医疗报告分析等现实场景中具有重要应用价值。

当前主流数据集如DocVQA（单页文档）和MP-DocVQA（多页文档）提出了独特挑战：

- 文档特有的布局结构（如表格、公式）需要空间感知能力；
- 跨页信息关联要求长注意力依赖;
- 细粒度文字识别与语义理解的协同需求。

尽管Qwen2.5-VL等最新开源模型在通用VQA任务中表现优异，但其在专业文档场景下的能力尚未得到系统验证。现有研究存在两个关键问题：

- 表格等结构化信息的理解不足；
- 上下文依赖长、噪声干扰多。

本实验旨在基于现有开源VLM，探索其在Document VQA任务上的表现，分析模型优势与不足，并尝试简单的优化策略。具体目标包括：

- 评估开源VLM在DocVQA和MP-DocVQA上的基线表现；
- 通过优化策略提升VLM对VQA的表现。

## 2. 任务目标

- 使用开源VLM模型完成 DocVQA 和 MP-DocVQA 数据集上的图像问答任务。
- 比较和分析不同策略对模型的影响。
- 优化策略，提高模型的表现。

## 3. 数据集
本次实验主要使用DocVQA（单页文档视觉问答数据集）和MP-DocVQA：（多页文档视觉问答数据集）。为便于实验, 采样其中的 100 条作为本次实验的数据集。

### DocVQA主要字段如下：

- **question**: 自然语言问题
- **image**: 文档图像
- **answers**: 标准答案集合

### MP-DocVQA主要字段如下：

- **question**: 自然语言问题
- **answers**: 标准答案集合
- **answer_page_idx**: 答案所在文档图像
- **image_1**: 文档图像1
- **image_2**: 文档图像2
- ...
- **image_20**: 文档图像20

数据集加载使用datasets库，详细见附件code.zip对应代码中的`load_data`函数，可以直接使用。

## 4. 评价指标

**Pass Rate**：模型在给定问题上返回的答案正确率。

评估函数详细见附件code.zip对应代码中的`evaluate_results`函数，可以直接使用。

## 5. 具体任务

### 5.1 DocVQA

1. 参考code.zip中对应代码实现，评估VLM在DocVQA上的性能（baseline）

2. 使用可能的优化策略提升VLM在DocVQA上的性能，例如：
   - Prompt优化
   - 图像预处理/增强
   - 其他创新策略

### 5.2 MP-DocVQA

1. 参考code.zip中对应代码实现，比较VLM在不同设置下在MP-DocVQA上的性能，包括：
   - 只使用正确图像下，VLM的性能
   - 只使用第一张图像下，VLM的性能（baseline）

2. 使用可能的优化策略提升VLM在MP-DocVQA上的性能，例如：
   - 多图像拼接/增强 （注意图像大小，不要超过20 * 420 * 420）
   - OCR处理+RAG选取图像
   - 其他创新策略

### 5.3 报告

提交一份简洁的报告, 清晰描述在不同数据集中不同策略下的效果。 报告建议不超过 5 页, 主要讲清楚自己的不同策略的实现和结果分析/改进和启发。

如果使用或参考了外部资源的代码, 请在报告中注明。

## 6. 示例代码

我们提供了code.zip，文件格式如下：

```
code
├── docvqa/
│       └── baseline.py
└── mp_docvqa/
          └── baseline.py
```

baseline.py均可直接运行，用于复现baseline结果。请基于示例代码进行改进，每个策略一个单独文件。

示例代码对应函数功能如下：

- 数据加载: `load_data`
- 图像处理: `preprocess_image`
- 结果生成: `generate_answer`
- 结果评估: `evaluate_results`

## 7. 模型访问

为了更顺利地完成实验，建议使用该课程部署的 Qwen2.5-VL-3B-Instruct 模型和示例代码, 但需要注意的是：

- 该服务器的显存有限，请参考示例代码逐个生成，并每次输入图像的大小不要超过20 * 420 *420；
- 该服务器的并发能力有限, 请合理安排使用时间, 避免在截止日期临近时集中使用导致请求响应缓慢。

## 8. 提交要求

提交名为 学号_姓名.zip 的压缩文件, 文件组织示例：

```
project
├── code/
│     ├── docvqa/
│     │      ├──results/
│     │      │     ├──baseline.json
│     │      │     └── methodx.json
│     │      ├──baseline.py
│     │      └── methodx.py
│     ├── mp_docvqa/
│     │      ├──results/
│     │     │     ├──baseline.json
│     │     │     └── methodx.json
│     │     ├──baseline.py
│     │     └── methodx.py
│     └── README.md (简要版本即可)
└── report.pdf
```

注：

- 提交文件务必包含运行结果xxx.json，具体methodx.py的数量取决于自行尝试的数量，可以提交并讨论所有的方法；
- 不要提交数据集以及模型源文件等内容。

## 9. 评分标准

| 分数范围 | 描述 |
| --- | --- |
| 90-100% | 功能：复现完整的baseline; 改进：尝试优化策略，相较于baseline有明显提升（DocVQA准确率>=0.90, MP-DocVQA准确率>=0.60）; 报告：报告完整、清晰详细讲解了不同优化策略、深度分析提升的原因。 |
| 80-89% | 功能：复现完整的baseline; 改进：尝试优化策略，相较于baseline有一定提升（DocVQA准确率>=0.85, MP-DocVQA准确率>=0.50）; 报告：报告完整、清晰详细讲解了不同优化策略。 |
| 70-79% | 功能：复现完整的baseline; 改进：尝试优化策略，但相较于baseline没有提升; 报告：报告完整、讲解了不同优化策略。 |
| 60-69% | 功能：复现完整的baseline; 改进：没有尝试优化策略; 报告：报告简陋，难以理解。 |
| 低于 60% | 功能：复现完整的baseline; 改进：没有尝试优化策略; 报告：报告存在严重缺陷。 |

## 10. 截止日期
作业截止日期：2025-05-31 23:59:59

## 11. 提示和建议

快速了解两个数据集可以参考下述链接：

- DocVQA： [https://huggingface.co/datasets/lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA)
- MP-DocVQA： [https://huggingface.co/datasets/lmms-lab/MP-DocVQA](https://huggingface.co/datasets/lmms-lab/MP-DocVQA)

快速了解Qwen2.5-VL可以参考该链接：
[https://github.com/QwenLM/Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)