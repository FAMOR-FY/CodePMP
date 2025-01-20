
# CodePMP：基于代码偏好的预训练模型以增强大语言模型的推理能力

## 概述

在大语言模型（LLMs）的对齐训练中，虽然基于人类反馈的强化学习（RLHF）已被证明是有效的，但其性能高度依赖于奖励模型（RM）的能力。在数学和逻辑推理等复杂推理领域，获取高质量的偏好数据成本高昂且标注困难。为解决这一挑战，我们提出了CodePMP方法，该方法利用GitHub上公开的源代码数据来合成大规模、多样化的代码偏好数据。这使得偏好模型的可扩展预训练成为可能，从而提高了推理RM微调的样本效率，并减少了对大量高质量人工标注数据的依赖。

## 主要特点

- 利用来自GitHub的高质量代码片段生成多样化的代码提示。
- 提高了推理RM微调的样本效率。
- 减少了对大量手动标注数据的需求。

## 开始使用

### 安装

1. **克隆仓库：**
   ```bash
   git clone https://github.com/FAMOR-FY/CodePMP.git
   cd CodePMP
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

### 使用说明

1. **数据准备：**
   - 注意：PMP和RM微调所需的训练数据并未包含在此仓库中。如需获取数据，请联系作者。

2. **PMP训练：**
   - 运行PMP训练脚本：
     ```bash
     bash pmp_rm1_lm1_7b_1b_all_wsd_qwen_7b.sh
     ```

3. **RM微调：**
   - 运行RM微调脚本：
     ```bash
     bash RM_finetune_MathShepherd_40k_pmp_rm1_lm1_7b_1b_all_wsd_qwen7b.sh
     ```

## 评估

评估脚本已集成到RM微调脚本中。请参考那些脚本获取评估流程。

## 引用

```latex
@article{Yu2024CodePMPSP,
  title={CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning},
  author={Huimu Yu and Xing Wu and Weidong Yin and Debing Zhang and Songlin Hu},
  journal={ArXiv},
  year={2024},
  volume={abs/2410.02229},
  url={https://api.semanticscholar.org/CorpusID:273098454}
}
```

## 常见问题

- **缺少数据：**
  - 请确保已获取所有必要的数据集，如"数据准备"部分所述。
