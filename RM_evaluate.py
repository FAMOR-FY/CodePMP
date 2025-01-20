import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import transformers
from RM_finetune import RewardDataCollatorWithPadding
import argparse
from torch.utils.data import Dataset
import logging
import os
import io
import json
import pyarrow.parquet as pq
from tqdm import tqdm
import logging
import random
import numpy as np

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

class EvalDataset_MetaMath_DPO(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_MetaMath_DPO, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'eval' in fp:
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
        # source = PROMPT_DICT['prompt_code'].format_map(data_dict)
        source = data_dict['prompt']
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_MS(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_MS, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'test' in fp:
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
        # source = PROMPT_DICT['prompt_code'].format_map(data_dict)
        source = PROMPT_DICT['prompt_math_stack'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_MS_40k(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_MS_40k, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'test' in fp:
                dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval[:40000]
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
        source = PROMPT_DICT['prompt_math_stack'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_chenhao(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_chenhao, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'jsonl' and 'test' in fp:
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
        source = PROMPT_DICT['prompt_math_step_dpo'].format_map(data_dict)
        real_target = data_dict['positive_response']
        fake_target = data_dict['negative_response']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict


class EvalDataset_drop_20k(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_drop_20k, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "test_result.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
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
        source = PROMPT_DICT['drop'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_reclor(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_reclor, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "val.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
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
        source = PROMPT_DICT['reclor'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_logiqa2(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_logiqa2, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        with open(os.path.join(data_path, "test.jsonl"), 'r')as f:
            dataset_for_eval = [json.loads(line) for line in f.readlines()]
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
        source = PROMPT_DICT['reclor'].format_map(data_dict)
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_CodeUltraFeedback(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_CodeUltraFeedback, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'jsonl' and 'gpt4o-maas' in fp:
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
        # source = PROMPT_DICT['prompt_code'].format_map(data_dict)
        source = PROMPT_DICT['prompt_codeultrafeedback'].format_map(data_dict)
        real_target = data_dict['chosen'] if data_dict['flag'] else data_dict['rejected']
        fake_target = data_dict['rejected'] if data_dict['flag'] else data_dict['chosen']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict

class EvalDataset_Math_Step_DPO(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_Math_Step_DPO, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = []
        for fp in os.listdir(data_path):
            if fp.split('.')[-1] == 'parquet' and 'train' in fp:
                dataset_for_eval += [row for i, row in  pq.read_table(os.path.join(data_path, fp)).to_pandas().iterrows()]
        self.df = dataset_for_eval[int(len(dataset_for_eval)*0.9):]
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

class EvalDataset_math_stack(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset_math_stack, self).__init__()
        logging.warning("Loading data...")
        self.tokenizer = tokenizer
        dataset_for_eval = jload(data_path)
        self.df = dataset_for_eval[int(len(dataset_for_eval)*0.8):]
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
        real_target = data_dict['chosen']
        fake_target = data_dict['rejected']
        output_dict = self.encoder(source, real_target, fake_target)
        return output_dict


def evaluate_preference_acc(model, eval_dataset, tokenizer, training_args):
    model.eval()
    total_correct = 0
    total_samples = 0
    eval_dataloader = DataLoader(eval_dataset, batch_size=training_args['batch_size'], collate_fn=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=training_args['model_max_length']))
    
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            batch['input_ids'] = batch['input_ids'].to(model.device)
            batch['attention_mask'] = batch['attention_mask'].to(model.device)
            rewards = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )[0]
        
            rewards_chosen = rewards[::2]
            rewards_rejected = rewards[1::2]
            
            # 计算正确预测的数量
            correct_predictions = torch.sum((rewards_chosen > rewards_rejected).float()).item()
            total_samples += batch['input_ids'].size(0) / 2
            total_correct += correct_predictions
            # import ipdb; ipdb.set_trace()
    
    preference_acc = total_correct / total_samples
    return preference_acc

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the eval set and save results.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the eval data.")
    parser.add_argument("--out_path", type=str, required=True, help="Path to the output file where results will be saved.")
    parser.add_argument("--model_max_length", type=int, default=1024, help="Maximum length of the sequence.")
    parser.add_argument("--batch_size", type=int, default=2, help="Maximum length of the sequence.")
    parser.add_argument("--dataset_name", type=str, default='math_stack', help="")
    args = parser.parse_args()
    seed_everything(42)

    # 检查是否有可用的GPU，并将模型移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    # config._attn_implementation = 'flash_attention_2'
    # config.num_labels = 1


    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.model_max_length)
    model.config.pad_token_id = model.config.eos_token_id
    # 加载评估数据集
    if args.dataset_name == 'math_stack':
        eval_dataset = EvalDataset_math_stack(data_path=args.data_path, tokenizer=tokenizer)
    elif args.dataset_name == 'MetaMath_DPO':
        eval_dataset = EvalDataset_MetaMath_DPO(data_path=args.data_path, tokenizer=tokenizer)
    elif args.dataset_name == 'CodeUltraFeedback':
        eval_dataset = EvalDataset_CodeUltraFeedback(data_path=args.data_path, tokenizer=tokenizer)
    elif args.dataset_name == 'MathShepherd':
        eval_dataset = EvalDataset_MS(data_path=args.data_path, tokenizer=tokenizer)
    elif args.dataset_name == 'MathShepherd_40k':
        eval_dataset = EvalDataset_MS_40k(data_path=args.data_path, tokenizer=tokenizer)
    elif args.dataset_name == 'MathStepDPO':
        eval_dataset = EvalDataset_Math_Step_DPO(data_path=args.data_path, tokenizer=tokenizer)
    elif args.dataset_name == 'DROP_20k':
        eval_dataset = EvalDataset_drop_20k(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'DROP':
        eval_dataset = EvalDataset_drop(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'reclor':
        eval_dataset = EvalDataset_reclor(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'logiqa2':
        eval_dataset = EvalDataset_logiqa2(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'reclor_NR':
        eval_dataset = EvalDataset_reclor(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'logiqa2_NR':
        eval_dataset = EvalDataset_logiqa2(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'reclor+logiqa2_NR':
        eval_dataset = EvalDataset_reclor(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'reclor+logiqa2':
        eval_dataset = EvalDataset_reclor(tokenizer=tokenizer, data_path=args.data_path)
    elif args.dataset_name == 'chenhao':
        eval_dataset = EvalDataset_chenhao(tokenizer=tokenizer, data_path=args.data_path)

    # 评估模型
    training_args = {"batch_size": args.batch_size, 'model_max_length': args.model_max_length}  # 如果需要，可以在这里设置训练参数
    preference_acc = evaluate_preference_acc(model, eval_dataset, tokenizer, training_args)

    # 保存结果到文件
    output_file = os.path.join(args.out_path, 'preference_acc.txt')
    with open(output_file, 'w') as f:
        f.write(f"Preference Accuracy on Eval Set: {preference_acc}\n")
    print(f"Preference Accuracy on Eval Set: {preference_acc}\n")
    logging.warning(f"Preference Accuracy on Eval Set: {preference_acc}\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()