import os
import sys

import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List, Union
import json
import torch.distributed as dist
from safetensors.torch import load_file
from safetensors import safe_open

from tqdm import tqdm
import copy

from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'
# os.environ['RANK'] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['WORLD_SIZE'] = '1'

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
PROMPT_DICT = {
    "prompt_input": (
        "Instruction:\n{instruction}\n\nInput:\n{input}\n\nResponse:"
    ),
    "prompt_no_input": (
        "Instruction:\n{query}\n\nResponse:"
    ),
    "prompt_no_input_with_context": (
        "Instruction: Please answer the following questions based on the passages.\n\nPassage: {context}\n\nQuestion: {question}\n\nResponse:"
    ),
    "prompt_no_input_with_text": (
        "Instruction: Please answer the following questions based on the passages.\n\nPassage: {text}\n\nQuestion: {question}\n\nResponse:"
    ),
    "prompt_no_input_v2": (
        "Instruction:\n{input}\n\nResponse:"
    ),
}


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
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings)
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids)


class SupervisedDataset_logiqa2(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset_logiqa2, self).__init__()

        # dataset_for_eval = load_dataset(data_path)['train']
        dataset_for_eval = []

        with open(data_path, 'r') as f:
            for line in f.readlines():
                dataset_for_eval.append(json.loads(line))

        
        sources = [PROMPT_DICT['prompt_no_input_with_text'].format_map(item) + option for item in dataset_for_eval for option in item['options']]
        labels = [item['options'][int(item['answer'])] for item in dataset_for_eval for option in item['options']]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"] + data_dict["input_ids"][-100:]
        self.labels = labels + labels[-100:]
        self.ids = [item['id'] for item in dataset_for_eval for option in item['options']]
        self.ids = self.ids + self.ids[-100:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=self.ids[i])

class SupervisedDataset_reclor(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset_reclor, self).__init__()

        # dataset_for_eval = load_dataset(data_path)['train']
        dataset_for_eval = []

        with open(data_path, 'r') as f:
            dataset_for_eval = json.load(f)

        
        sources = [PROMPT_DICT['prompt_no_input_with_context'].format_map(item) + option for item in dataset_for_eval for option in item['answers']]
        labels = [item['answers'][int(item['label'])] for item in dataset_for_eval for option in item['answers']]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"] + data_dict["input_ids"][-100:]
        self.labels = labels + labels[-100:]
        self.ids = [item['id_string'] for item in dataset_for_eval for option in item['answers']]
        self.ids = self.ids + self.ids[-100:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], id=self.ids[i])


def padding(inputs, padding_token, cutoff=None):
    num_elems = len(inputs)
    if cutoff is None:
        cutoff = max([len(item) for item in inputs])
    else:
        cutoff = min(max([len(item) for item in inputs]), cutoff)

    tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
    for i in range(num_elems):
        toks = inputs[i]
        length = min(cutoff, len(toks))
        tokens[i, -length:] = toks[-length:]
    return tokens


def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim=-1)
    gathered_s = [torch.ones_like(s) * pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", 'id'))
        batch_features = self.tokenizer.pad(
            {
                'input_ids': list(input_ids),
            },
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        return dict(
            input_ids=batch_features['input_ids'],
            id=ids,
            attention_mask=batch_features['input_ids'].ne(self.tokenizer.pad_token_id),
            labels = labels,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if "reclor" in data_path:
        eval_dataset = SupervisedDataset_reclor(tokenizer=tokenizer, data_path=data_path)
    elif "logiqa2" in data_path:
        eval_dataset = SupervisedDataset_logiqa2(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main(args):
    # dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    # world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
    )

    model = model.to(accelerator.device)
    # model = model.half()
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
    )
    print('model loaded')
    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path)
    dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size
    )
    model, dataloader = accelerator.prepare(model, dataloader)
    print('data loaded')
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    print(num_processes, '==num_processes==', process_index, '==process_index==')
    # data_path = os.path.join(args.out_path, 'gsm8k_test_{}.json'.format(process_index))
    # data_path = os.path.join(args.out_path, 'gsm8k_test_dist_{}.json'.format(process_index))
    # data_path = os.path.join(args.out_path, 'gsm8k_test_dist_v1_{}.json'.format(process_index))
    # data_path = os.path.join(args.out_path, 'gsm8k_test_dist_v2_{}.json'.format(process_index))
    # data_path = os.path.join(args.out_path, 'gsm8k_test_dist_v2_sample_{}.json'.format(process_index))
    os.makedirs(args.out_path, exist_ok=True)
    # fp_name = os.listdir(data_path)[0]
    if "reclor" in data_path:
        dataset_name = 'reclor'
    elif "logiqa2" in data_path:
        dataset_name = 'logiqa2'
    data_path = os.path.join(args.out_path, 'MO_{}_test_rank_{}_{}.json'.format(dataset_name, 4, process_index))

    @accelerator.on_process(process_index=process_index)
    def write2json(fwobj, tmp):
        fwobj.write(json.dumps(tmp, ensure_ascii=False) + '\n')

    @accelerator.on_process(process_index=process_index)
    def close_fwobj(fwobj):
        fwobj.close()

    @accelerator.on_process(process_index=process_index)
    def open_fwobj(data_path):
        return open(data_path, 'w')

    fwobj = open_fwobj(data_path)
    print(fwobj, '==', process_index)

    import numpy as np
    for step, batch in tqdm(enumerate(dataloader)):
        # if step > 10:
        #     break
        # print(batch.pop('id'))
        # print(dataset_for_eval[step]['prompt'])
        if step == 0:
            print(batch)
        input_ids = batch['input_ids'].to(accelerator.device)
        attention_mask = batch['attention_mask'].to(accelerator.device)
        labels = batch['labels']
        with torch.no_grad():
            rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )[0]
        inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        outputs_ids = batch['id']

        if np.mod(step, 100) == 0:
            print(inputs_string, '==inputs_string==')

        for idx in range(len(inputs_string)):
            tmp = {
                'id': outputs_ids[idx],
                'input': inputs_string[idx],
                'rewards': (rewards[idx].flatten()).tolist(),
                'answer': labels[idx],
            }
            write2json(fwobj, tmp)

    close_fwobj(fwobj)
    accelerator.wait_for_everyone()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    args = parser.parse_args()

    main(args)
