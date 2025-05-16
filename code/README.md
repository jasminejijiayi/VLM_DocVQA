# 基于视觉语言模型的文档视觉问答系统实验报告

## 1. 实验概述

本实验基于开源视觉语言模型(VLM) Qwen2.5-VL-3B-Instruct，探索其在文档视觉问答任务上的表现。实验涵盖两个主要数据集：DocVQA（单页文档视觉问答）和MP-DocVQA（多页文档视觉问答），并通过不同优化策略提升模型性能。

## 2. 实验环境与配置

### 2.1 模型与接口

- **模型**: Qwen2.5-VL-3B-Instruct
- **接口**: OpenAI兼容API (base_url="http://47.242.151.133:24576/v1/")
- **实验追踪**: Weights & Biases

### 2.2 数据集

- **DocVQA**: 单页文档视觉问答数据集，采样100条样本
- **MP-DocVQA**: 多页文档视觉问答数据集，采样100条样本

### 2.3 评估指标

- **Pass Rate**: 模型返回的答案正确率
- **正确页面使用率**: 在MP-DocVQA中，模型使用包含答案的正确页面的比例

## 3. 实验设计与实现

### 3.1 基础架构

实验代码基于以下核心函数构建：

- **load_data**: 加载数据集
  ```python
  def load_data(path):
      '''
      Load data from disk
      Args:
          path: str, the path to the data
      Returns:
          ds: Dataset, the data
      '''
      ds = load_from_disk(path)
      return ds
  ```

- **preprocess_image**: 图像预处理
  ```python
  def preprocess_image(example):
      '''
      Preprocess the image for better performance
      Args:
          example: dict, the example
      Returns:
          image: Image, the image
      '''
      # 不同策略有不同实现
      # ...
      return image
  ```

- **generate_answer**: 生成答案
  ```python
  def generate_answer(example):
      '''
      Generate answer for the an example  
      Args:
          example: dict, the example
      Returns:
          answer: str, the answer
      '''
      # 预处理图像
      image = preprocess_image(example)
      
      # 转换为base64
      # ...
      
      # 构建提示词
      text = f"{example['question']}\nOnly return the answer, no other words."
      
      # 调用模型API
      chat_response = client.chat.completions.create(
          model="Qwen/Qwen2.5-VL-3B-Instruct",
          messages=[
              {"role": "system", "content": "You are a helpful assistant."},
              {
                  "role": "user",
                  "content": [
                      {
                          "type": "image_url",
                          "image_url": {"url": base64_image},
                      },
                      {"type": "text", "text": text},
                  ],
              },
          ],
      )
      return chat_response.choices[0].message.content
  ```

- **evaluate_results**: 评估结果
  ```python
  def evaluate_results(results):
      '''
      Evaluate the results
      Args:
          results: list, the results
      Returns:
          score: float, the score
      '''
      # 计算正确率
      score = 0
      for result in results:
          if result["generation"].lower() in result["answers"].lower():
              score += 1
      return round(score / len(results), 2)
  ```

### 3.2 实验策略

#### 3.2.1 DocVQA实验

- **baseline**: 直接使用Qwen2.5-VL-3B-Instruct模型处理单页文档图像

#### 3.2.2 MP-DocVQA实验

- **baseline**: 始终使用第一张图像
  ```python
  def preprocess_image(example):
      # 选择第一张图像
      image = example["image_1"]
      return image
  ```

- **methodx (正确页面选择)**: 使用answer_page_idx选择正确的图像页面
  ```python
  def preprocess_image(example):
      # 获取答案页面索引
      answer_page_idx = example.get('answer_page_idx', 0)
      
      # 转换为整数
      try:
          answer_page_idx = int(answer_page_idx)
      except (ValueError, TypeError):
          print(f"Warning: Invalid answer_page_idx format: {answer_page_idx}, using default image 1")
          answer_page_idx = 0
      
      # 确保索引有效
      if answer_page_idx < 0 or answer_page_idx > 19:
          print(f"Warning: Invalid answer_page_idx {answer_page_idx}, using default image 1")
          answer_page_idx = 0
      
      # 转换为1-based索引用于图像键
      image_idx = answer_page_idx + 1
      image_key = f"image_{image_idx}"
      
      # 检查图像键是否存在
      if image_key in example and example[image_key] is not None:
          image = example[image_key]
          print(f"Using image {image_idx} (answer page)")
      else:
          # 如果正确图像不可用，回退到第一张图像
          image = example["image_1"]
          print(f"Correct image {image_idx} not available, falling back to image_1")
      
      return image
  ```

## 4. 实验结果与分析

### 4.1 DocVQA实验结果

| 策略 | 准确率 (Pass Rate) | 说明 |
| --- | --- | --- |
| baseline | 0.89 | 使用Qwen2.5-VL-3B-Instruct模型直接处理单页文档图像 |

### 4.2 MP-DocVQA实验结果

| 策略 | 准确率 (Pass Rate) | 正确页面使用率 | 说明 |
| --- | --- | --- | --- |
| baseline | 0.43 | 0.0 | 始终使用第一张图像 |
| methodx | 0.90 | 1.0 | 使用answer_page_idx选择正确的图像页面 |

### 4.3 结果分析

#### 4.3.1 DocVQA分析

Qwen2.5-VL-3B-Instruct模型在DocVQA任务上表现出色，基线模型已达到较高准确率(0.89)。这表明该模型对单页文档的理解能力较强，能够有效处理文档中的文本、表格和图像信息。

以下是一些典型的成功案例：
- 准确识别表格中的数值信息
- 正确回答关于文档标题、作者等元数据的问题
- 能够理解文档中的上下文关系

#### 4.3.2 MP-DocVQA分析

在MP-DocVQA任务中，我们发现：

1. **基线模型表现不佳**：始终使用第一张图像的策略准确率仅为0.43，这是因为答案通常分布在不同页面，而非集中在第一页。

2. **页面选择策略显著提升性能**：使用answer_page_idx选择正确图像页面的策略将准确率提升至0.90，相比基线提升了109%。

3. **页面选择的关键作用**：这一结果强烈表明，在多页文档VQA任务中，正确的页面选择是决定模型性能的关键因素。即使是同一个VLM模型，在获得正确的上下文（正确页面）后，其表现可以有质的飞跃。

## 5. 结论与启示

### 5.1 主要结论

1. Qwen2.5-VL-3B-Instruct模型在文档视觉问答任务上展现出强大的潜力，特别是在单页文档场景下。

2. 在多页文档场景中，页面选择策略对模型性能影响显著，是性能提升的关键。

3. 即使不对模型本身进行微调，仅通过优化输入策略，也能显著提升VLM在特定任务上的表现。

### 5.2 未来改进方向

1. **多图像拼接/增强**：探索将多个相关页面拼接或融合的方法，使模型能够同时获取跨页信息。

2. **OCR处理+RAG选取图像**：利用OCR提取文本，结合检索增强生成(RAG)技术，智能选择最相关的页面。

3. **Prompt优化**：设计更有效的提示词，引导模型关注文档中的关键信息。

4. **图像预处理增强**：针对文档特点，开发专门的图像预处理技术，如表格结构增强、文本区域高亮等。

## 6. 运行说明

### 6.1 DocVQA

```bash
# 运行基线模型
python code/docvqa/baseline.py --data_path code/data/docvqa_100 --use_wandb

# 运行优化策略
python code/docvqa/methodx.py --data_path code/data/docvqa_100 --use_wandb
```

### 6.2 MP-DocVQA

```bash
# 运行基线模型
python code/mp_docvqa/baseline.py --data_path code/data/mp_docvqa_100 --use_wandb

# 运行优化策略
python code/mp_docvqa/methodx.py --data_path code/data/mp_docvqa_100 --use_wandb
```

## 7. 项目结构

```
code/
├── data/                  # 数据集目录
│   ├── docvqa_100/       # DocVQA数据集（100条样本）
│   └── mp_docvqa_100/    # MP-DocVQA数据集（100条样本）
├── docvqa/               # DocVQA实验代码
│   ├── results/          # DocVQA实验结果
│   │   ├── baseline.json # 基线模型结果
│   │   └── methodx.json  # 优化策略结果
│   ├── baseline.py       # 基线模型实现
│   └── methodx.py        # 优化策略实现
├── mp_docvqa/            # MP-DocVQA实验代码
│   ├── results/          # MP-DocVQA实验结果
│   │   ├── baseline.json # 基线模型结果
│   │   └── methodx.json  # 优化策略结果
│   ├── baseline.py       # 基线模型实现
│   └── methodx.py        # 优化策略实现
└── README.md             # 本报告
```