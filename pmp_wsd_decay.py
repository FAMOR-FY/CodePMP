#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# os.environ['WORLD_SIZE'] = '1'
# os.environ['NCCL_DEBUG'] = 'INFO'
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
from reward_trainer import RewardTrainer, RewardTrainer_power_save
from transformers.trainer_utils import get_last_checkpoint
from torch import nn
import math
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
    "prompt_code": (
        "### Instruction: Please read the following code description carefully and generate the actual code implementation based on this description. Please make sure the code is logical and readable.\n\n"
        #"### Code Style Example (not actual code, just style reference):\n"
        # """```import pandas as pd \nfrom sklearn import linear_model \n  \ndef displayLowEducationCourseEnrollment(df): \n    x = df[["SAT Math Score", "ACT English Score"]] \n    y = df["Low Education"] \n      \n    regr = linear_model.LinearRegression() \n    regr.fit(x,y) \n      \n    intercept = regr.intercept_ \n    coefficients = regr.coef_ \n      \n    print("Intercept:", intercept) \n    print("Coefficients:") \n    for i,j in zip(x.columns,coefficients): \n        print("{{}}:{{}}".format(i, j)) \n      \n    predicted_values = regr.predict([[1200,30]]) \n      \n    if (predicted_values >= 0.5): \n        print("\\\nUser does NOT qualify for this program") \n    else: \n        print("\nUser DOES qualify for this program") \n          \nif __name__ == '__main__': \n    df = pd.read_csv('data.csv') \n      \n    displayLowEducationCourseEnrollment(df) \n```\n\n"""
        "### Code Description:\n"
        "{query}\n\n"
        "### Response:"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    resumed_model_name_or_path: Optional[str] = field(default="")
    attention_dropout: Optional[float] = field(default=0.0)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    non_engin_ratio: float = field(default=1.0)
    engin_ratio: float = field(default=1.0)

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
    rm_coef: float = field(default=1.0)
    lm_coef: float = field(default=0.0)
    save_interval: float = field(default=0)


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

class NCEDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args, tokenizer: transformers.PreTrainedTokenizer):
        super(NCEDataset, self).__init__()
        logging.warning("Loading data...")
        data_path = data_args.data_path
        self.tokenizer = tokenizer
        if os.path.isfile(data_path):
            if data_path.split('.')[-1] == 'parquet':
                dataset_for_eval = [row for i, row in  pq.read_table(data_path).to_pandas().iterrows()]
            elif data_path.split('.')[-1] == 'json':
                dataset_for_eval = jload(data_path)
            else:
                with open(data_path, 'r') as f:
                    lines = f.readlines()
                dataset_for_eval = [json.loads(line.strip()) for line in lines]
        else:
            dataset_for_eval = []
            fps = [os.path.join(data_path, fp) for fp in os.listdir(data_path)]
            for fp in fps:
                if fp.split('.')[-1] == 'parquet':
                    dataset_for_eval += [row for i, row in  pq.read_table(fp).to_pandas().iterrows()]
                elif fp.split('.')[-1] == 'json':
                    dataset_for_eval += jload(fp)
                else:
                    with open(fp, 'r') as f:
                        lines = f.readlines()
                    dataset_for_eval += [json.loads(line.strip()) for line in lines]
        
        if data_args.non_engin_ratio + data_args.engin_ratio < 2.0 and data_args.non_engin_ratio != data_args.engin_ratio:
            type2data = {
                'engineering code': [],
                'non-engineering code': []
            }
            for item in dataset_for_eval:
                if item['better'] != "``````" and item['better'] != "" and item['worse'] != "``````" and item['worse'] != "" and item['query_type'] != None and item['confidence'] != None:
                    type2data[item['query_type']].append(item)
            for key in type2data.keys():
                type2data[key] = sorted(type2data[key], key=lambda item:item['confidence'], reverse=True)
            
            self.df = type2data['non-engineering code'][:int(len(type2data['non-engineering code']) * data_args.non_engin_ratio)] + type2data['engineering code'][-int(len(type2data['engineering code']) * data_args.engin_ratio):]
            import random
            random.seed(1)
            random.shuffle(self.df)
        else:
            self.df = [item for item in dataset_for_eval if item['better'] != "``````" and item['better'] != "" and item['worse'] != "``````" and item['worse'] != ""]
            self.df = self.df[:int(len(self.df) * data_args.non_engin_ratio)]
        print(len(self.df), '==size of sources==')
        
    def __len__(self):
        return len(self.df)
    
    def encoder(self, source, real_target, fake_target):
        real = source + real_target
        fake = source + fake_target
        real_target_feature = self.tokenizer(
            real_target,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        real_target_ids_lens = real_target_feature.input_ids[0].ne(self.tokenizer.pad_token_id).sum().item()
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
                    real_target_ids_lens=real_target_ids_lens)
        return output_dict

    def __getitem__(self, i):
        data_dict = self.df[i]
        source = PROMPT_DICT['prompt_code'].format_map(data_dict)
        real_target = data_dict['better']
        fake_target = data_dict['worse']
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
                or "real_target_ids_lens" not in feature
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
            labels[i, :-features[i//2]['real_target_ids_lens']] = IGNORE_INDEX

        batch = {
            "input_ids": batch_features["input_ids"],
            "labels": labels,
            "attention_mask": batch_features["attention_mask"],
            "return_loss": True,
        }
        return batch

def make_nce_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = NCEDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=training_args.model_max_length)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def find_latest_checkpoint(directory):
    checkpoints = [d for d in os.listdir(directory) if d.startswith('checkpoint-') and d[len('checkpoint-'):].isdigit()]
    return os.path.join(directory, max(checkpoints, key=lambda x: int(x[len('checkpoint-'):]), default=None)) if checkpoints else None

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(training_args.seed)
    
    print(data_args, '==data_args==')

    resume_ckpt = find_latest_checkpoint(training_args.output_dir)
    if resume_ckpt:
        model_args.model_name_or_path = resume_ckpt
        model_args.resumed_model_name_or_path = resume_ckpt
    training_args_path = os.path.join(model_args.resumed_model_name_or_path, "training_args.bin")
    resume_training_args = torch.load(training_args_path)

    training_args.max_steps = resume_training_args.max_steps
    training_args.lr_scheduler_kwargs = resume_training_args.lr_scheduler_kwargs
    training_args.warmup_steps = int(27458 * 0.03)

    # if os.path.isdir(training_args.output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    # else:
    #     last_checkpoint = None
    # print(last_checkpoint)
    
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    # config._attn_implementation = 'flash_attention_2'
    config.num_labels = 1
    config.attention_dropout = model_args.attention_dropout

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16
    )

    lm_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16
        # torch_dtype=torch.float32,
    )

    model.lm_head = lm_model.lm_head
    del lm_model
    
    # import os
    # from safetensors.torch import load_file
    # from safetensors import safe_open
    # tmp_ckpt = {}
    # for fp in os.listdir(model_args.model_name_or_path):
    #     if fp.split('.')[-1] == 'safetensors':        
    #         ckpt = load_file(os.path.join(model_args.model_name_or_path, fp))
    #         for k in ckpt.keys():
    #             if 'lm_head' in k:
    #                 tmp_ckpt['.'.join(k.split('.')[1:])] = ckpt[k]
    # import ipdb;ipdb.set_trace()
    # model.lm_head.load_state_dict(tmp_ckpt, strict=True)

    print('model loaded')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
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
    
    if training_args.lr_scheduler_type == "warmup_stable_decay":
    #     print(f"Proc Num: {int(os.environ['WORLD_SIZE'])}")
        total_steps = math.ceil(len(data_module['train_dataset']) / (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * int(os.environ["WORLD_SIZE"])))
        warmup_steps = training_args.warmup_steps if training_args.warmup_steps > 0 else math.ceil(total_steps * training_args.warmup_ratio)
    #     training_args.lr_scheduler_kwargs['num_decay_steps'] = math.ceil(total_steps * training_args.lr_scheduler_kwargs['decay_ratio'])
    #     training_args.lr_scheduler_kwargs['num_stable_steps'] = total_steps - training_args.lr_scheduler_kwargs['num_decay_steps'] - warmup_steps
    #     del training_args.lr_scheduler_kwargs['decay_ratio']
    #     print(f"total_steps: {total_steps}")
    #     print(f"total_steps: {total_steps}")
        if training_args.save_interval > 0:
            n = warmup_steps
            save_steps_value = []
            while n <= training_args.lr_scheduler_kwargs['num_stable_steps'] + warmup_steps:
                save_steps_value.append(n)
                n = int(n * training_args.save_interval + 0.5)
            save_steps_value.append(training_args.lr_scheduler_kwargs['num_stable_steps'] + warmup_steps)
            training_args.save_steps_value = save_steps_value
    print(training_args, '==training_args==')
    if training_args.lr_scheduler_type == "warmup_stable_decay" and training_args.save_interval > 0:
        trainer = RewardTrainer_power_save(model=model, tokenizer=tokenizer, args=training_args, 
                            **data_module)
    else:
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