#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#    Modified by Zheng Yuan and Hongyi Yuan

import os

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ['RANK'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['WORLD_SIZE'] = '1'

# import deepspeed

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from reward_trainer import RewardTrainer
from transformers.trainer_utils import get_last_checkpoint
import pyarrow.parquet as pq

from torch.distributed import init_process_group
import datetime
# timeout_in_seconds = 120 * 60
# init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=5400))

import random
import numpy as np
import json

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(transformers.__version__, '==__version__==')

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_deepseek": (
        "Instruction:{input}\n\nResponse:"
    ),
    "prompt_mistral": (
        "Instruction:{input}\n\nResponse:"
    ),
    "prompt_math_stack": (
        "Instruction:{question}\n\nResponse:"
    ),
    "prompt_codeultrafeedback": (
        "Instruction:{instruction}\n\nResponse:"
    ),
    "prompt_math_step_dpo": (
        "Instruction:{prompt}\n\nResponse:"
    ),
    "drop": (
        "Instruction: Please answer the following questions based on the passages.\n\nPassage: {passage}\n\nQuestion: {question}\n\nResponse:"
    ),
    "reclor": (
        "Instruction: Please answer the following questions based on the passages.\n\nPassage: {context}\n\nQuestion: {question}\n\nResponse:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    resumed_model_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    overwrite_output_dir: bool = field(default=False)
    remove_unused_columns:  bool = field(default=False)
    dataset_name: str = field(default="math_stack")
    data_percentage: int = field(default=100)
    rm_coef: float = field(default=1.0)
    lm_coef: float = field(default=0.0)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        # output_embeddings = model.model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        # output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        # output_embeddings[-num_new_tokens:] = output_embeddings_avg

class NCEDataset_MetaMath_DPO(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_percentage):
        super(NCEDataset_MetaMath_DPO, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'train' in fp:
                dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval[:len(dataset_for_eval)*data_percentage//100]
        print(len(self.df), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        # source = PROMPT_DICT['prompt_code'].format_map(data_dict)
        source = data_dict['prompt']
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_MS(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_MS, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'train' in fp:
                dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_math_stack'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_MS_40k(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_MS_40k, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'train' in fp:
                dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_math_stack'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_chenhao(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_chenhao, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'jsonl' and 'train' in fp:
                with open(os.path.join(data_path, fp), 'r')as f:
                    dataset_for_eval += [json.loads(row) for row in f.readlines()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_math_step_dpo'].format_map(data_dict)
        real_target = data_dict['positive_response']
        fake_target = data_dict['negative_response']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict


class NCEDataset_drop_20k(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_drop_20k, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "result_no_repeat.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['drop'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_drop(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_drop, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "result.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['drop'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_reclor(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_reclor, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "train.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['reclor'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_logiqa2(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_logiqa2, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "train.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        print(len(dataset_for_eval), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['reclor'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict


class NCEDataset_math_stack(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_math_stack, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = jload(data_path)
        self.df = dataset_for_eval[:int(len(dataset_for_eval)*0.9)]
        print(len(self.df), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_math_stack'].format_map(data_dict)
        # source = data_dict['prompt']
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_CodeUltraFeedback(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_CodeUltraFeedback, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'jsonl' and 'gpt4o-maas_' in fp:
                with open(os.path.join(data_path, fp), 'r')as f:
                    dataset_for_eval += [json.loads(row) for row in f.readlines()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_codeultrafeedback'].format_map(data_dict)
        real_target = data_dict['chosen'] if data_dict['flag'] else data_dict['rejected']
        fake_target = data_dict['rejected'] if data_dict['flag'] else data_dict['chosen']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_Math_Step_DPO(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_Math_Step_DPO, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'train' in fp:
                dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval[:int(len(dataset_for_eval)*0.9)]
        print(len(self.df), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_math_step_dpo'].format_map(data_dict)
        real_target = data_dict['initial_reason_steps'] + data_dict['full_chosen']
        fake_target = data_dict['initial_reason_steps'] + data_dict['full_rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class NCEDataset_UltraFeedback(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset_UltraFeedback, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'train' in fp:
                with open(os.path.join(data_path, fp), 'r')as f:
                    dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval
        print(len(self.df), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        source_feature = self.tokenizer(
            source,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        source_ids_lens = source_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
        real_feature = self.tokenizer(
            real,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        fake_feature = self.tokenizer(
            fake,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        output_dict = dict(input_ids_chosen=real_feature.input_ids[0], 
                    attention_mask_chosen=real_feature.attention_mask[0], 
                    input_ids_rejected=fake_feature.input_ids[0],
                    attention_mask_rejected=fake_feature.attention_mask[0],
                    source_ids_lens=source_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_codeultrafeedback'].format_map(data_dict)
        real_target = data_dict['chosen_response']
        fake_target = data_dict['rejected_response']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict


@dataclass
class RewardDataCollatorWithPadding(object):
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """
    tokenizer: transformers.PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: Sequence[Dict]) -> Dict[str, Any]:
        features_all = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
                or "source_ids_lens" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_all.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )

            features_all.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )

        batch_features = self.tokenizer.pad(
            features_all,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        labels = copy.deepcopy(batch_features["input_ids"])
        labels[batch_features["input_ids"]==self.tokenizer.pad_token_id] = IGNORE_INDEX
        for i in range(labels.shape[0]):
            labels[i, :features[i//2]['source_ids_lens']] = IGNORE_INDEX

        batch = {
            "input_ids": batch_features["input_ids"],
            "labels": labels,
            "attention_mask": batch_features["attention_mask"],
            "return_loss": True,
        }
        return batch

def make_nce_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if training_args.dataset_name == 'math_stack':
        train_dataset = NCEDataset_math_stack(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'MetaMath_DPO':
        train_dataset = NCEDataset_MetaMath_DPO(tokenizer=tokenizer, data_path=data_args.data_path, data_percentage=training_args.data_percentage)
    elif training_args.dataset_name == 'CodeUltraFeedback':
        train_dataset = NCEDataset_CodeUltraFeedback(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'MathShepherd':
        train_dataset = NCEDataset_MS(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'MathShepherd_40k':
        train_dataset = NCEDataset_MS_40k(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'MathStepDPO':
        train_dataset = NCEDataset_Math_Step_DPO(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'DROP_20k':
        train_dataset = NCEDataset_drop_20k(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'DROP':
        train_dataset = NCEDataset_drop(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'reclor':
        train_dataset = NCEDataset_reclor(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'UltraFeedback':
        train_dataset = NCEDataset_UltraFeedback(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'logiqa2':
        train_dataset = NCEDataset_logiqa2(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'reclor_NR':
        train_dataset = NCEDataset_reclor(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'logiqa2_NR':
        train_dataset = NCEDataset_logiqa2(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'reclor+logiqa2_NR':
        train_dataset = NCEDataset_reclor(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'reclor+logiqa2':
        train_dataset = NCEDataset_reclor(tokenizer=tokenizer, data_path=data_args.data_path)
    elif training_args.dataset_name == 'chenhao':
        train_dataset = NCEDataset_chenhao(tokenizer=tokenizer, data_path=data_args.data_path)
    
    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=training_args.model_max_length)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)
    print(training_args, '==training_args==')
    print(data_args, '==data_args==')

    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # else:
    #     last_checkpoint = None
    # print(last_checkpoint)
    
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    # config._attn_implementation = 'flash_attention_2'
    config.num_labels = 1
    config._attn_implementation_internal = "eager"

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True
        # torch_dtype=torch.float32,
        # num_labels=1
    )

    print('model loaded')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
        trust_remote_code=True,
        # add_prefix_space=False,
    )
    model.config.pad_token_id = model.config.eos_token_id
    
    print(tokenizer.pad_token, '===pad_token====', model.config.pad_token_id)

    data_module = make_nce_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    
    from torch.utils.data import DataLoader

    test_loader = DataLoader(data_module['train_dataset'],
                              collate_fn=data_module['data_collator'],
                              sampler=None,
                              batch_size=1)
    print('==begin to decode for verification==', len(data_module['train_dataset']))
    for idx, d in enumerate(test_loader):
        input_ids = d['input_ids']
        print(tokenizer.batch_decode(input_ids))
        print(d['attention_mask'])
        sequence_lengths = torch.eq(d['input_ids'], tokenizer.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % d['input_ids'].shape[-1]
        sequence_lengths = sequence_lengths
        print(sequence_lengths)
        if idx >= 1:
            break
    
    trainer = RewardTrainer(model=model, tokenizer=tokenizer, args=training_args, 
                         **data_module)
    if model_args.resumed_model_name_or_path:
        print('==resume_from_checkpoint==', model_args.resumed_model_name_or_path)
        trainer.train(resume_from_checkpoint=model_args.resumed_model_name_or_path)
    else:
        trainer.train()
    trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    trainer.save_model(output_dir=training_args.output_dir)
    
    # lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model.module.state_dict())
    # if torch.distributed.get_rank() == 0:
    #     model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
