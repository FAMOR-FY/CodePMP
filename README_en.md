# CodePMP: Code-based Preference Model Pre-training for Enhancing LLM Inference Capabilities

## Overview

In the alignment training of Large Language Models (LLMs), while Reinforcement Learning from Human Feedback (RLHF) has proven effective, its performance heavily depends on the capabilities of the Reward Model (RM). In complex domains such as mathematics and logical reasoning, acquiring high-quality preference data is costly and laborious. To address this challenge, we introduce CodePMP, a method that leverages publicly available source code data from GitHub to synthesize large-scale, diverse code preference data. This enables scalable preference model pre-training, thereby enhancing the sample efficiency of inference RM fine-tuning and reducing reliance on extensive high-quality human-labeled data.
<img width="1053" alt="image" src="https://github.com/user-attachments/assets/8d8e0201-5356-4291-8ad0-bc3cf9fee017" />

## Key Features

- Utilizes high-quality code snippets from GitHub to generate diverse code prompts.
- Enhances the sample efficiency of inference RM fine-tuning.
- Reduces the need for large amounts of manually labeled data.

## Getting Started

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/FAMOR-FY/CodePMP.git
   cd CodePMP
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage Instructions

1. **Data Preparation:**
   - Note: The training data required for PMP and RM fine-tuning is not included in this repository. Please contact the authors to obtain the necessary datasets.

2. **PMP Training:**
   - Run the PMP training script:
     ```bash
     bash pmp_rm1_lm1_7b_1b_all_wsd_qwen_7b.sh
     ```

3. **RM Fine-tuning:**
   - Run the RM fine-tuning script:
     ```bash
     bash RM_finetune_MathShepherd_40k_pmp_rm1_lm1_7b_1b_all_wsd_qwen7b.sh
     ```

## Evaluation

Evaluation scripts are integrated into the RM fine-tuning scripts. Please refer to those scripts for evaluation procedures.

## Citation

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

## Troubleshooting

- **Missing Data:**
  - Ensure you have obtained all necessary datasets as specified in the "Data Preparation" section.
