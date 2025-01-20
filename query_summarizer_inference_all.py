import os
import sys

import argparse
import torch
import transformers
# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import io
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Sequence, List
import json
import torch.distributed as dist

from tqdm import tqdm
import copy
import re
from accelerate import DistributedDataParallelKwargs
from accelerate import Accelerator
import pyarrow.parquet as pq

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Please generate a corresponding instruction based on this code snippet. In the instruction you need to analyze the intent and function of this code snippet (describe the operation the code performs and how it achieves its function).\n\n"
        "#### Code snippet:\n"
        "```{source_code}```\n\n"
        "#### Instructions:"
    ),
}

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
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, machine_rank, machine_num, code_type):
        super(SupervisedDataset, self).__init__()

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
            fps = [os.path.join(data_path, fp) for fp in os.listdir(data_path) if fp.split('_')[0] == code_type.split('_')[0]]
            fps = fps[int(len(fps) / machine_num * machine_rank ): int(len(fps) / machine_num * (machine_rank+1))]
            for fp in fps:
                if fp.split('.')[-1] == 'parquet':
                    dataset_for_eval += [row for i, row in  pq.read_table(fp).to_pandas().iterrows()]
                elif fp.split('.')[-1] == 'json':
                    dataset_for_eval += jload(fp)
                else:
                    with open(fp, 'r') as f:
                        lines = f.readlines()
                    dataset_for_eval += [json.loads(line.strip()) for line in lines]


        # print(len(dataset_for_eval), '==dataset_for_eval==')
        self.ids = []
        sources = []
        query_types = []
        confidences = []

        for item in tqdm(dataset_for_eval):
            source_code = item['text']
            d = {'source_code': source_code}
            template = PROMPT_DICT["prompt_input"]
            sources.append(template.format_map(d)+'\n')
            query_types.append(item['query_type'])
            confidences.append(item['confidence'])
        self.ids = list(range(len(sources)))

        # data_dict = preprocess(sources, tokenizer)
        # print(preprocess(sources[:2], tokenizer), '==preprocss==')
        # print(tokenizer.batch_decode(preprocess(sources[:2], targets[:2], tokenizer)['input_ids'], skip_special_tokens=True))

        if os.path.isfile(data_path):
            self.sources = sources[int(len(sources) / machine_num * machine_rank ): int(len(sources) / machine_num * (machine_rank+1))]
            self.ids = self.ids[int(len(sources) / machine_num * machine_rank ): int(len(sources) / machine_num * (machine_rank+1))]
            self.query_types = query_types[int(len(sources) / machine_num * machine_rank ): int(len(sources) / machine_num * (machine_rank+1))]
            self.confidences = confidences[int(len(sources) / machine_num * machine_rank ): int(len(sources) / machine_num * (machine_rank+1))]
        else:
            self.sources = sources
            self.query_types = query_types
            self.confidences = confidences
        # self.targets = targets[
        #            int(len(sources) / machine_num * machine_rank): int(len(sources) / machine_num * (machine_rank + 1))]

        # print(len(self.input_ids), '===', len(self.ids))
        
        # assert len(self.input_ids) == len(self.ids)

        completed_len = 0
        if os.path.exists(f"/root_path/CodePMP/data/code2query_all/deepseek_coder_1.3b/{code_type}_code2query_{machine_rank}-of-{machine_num}.jsonl"):
            with open(f"/root_path/CodePMP/data/code2query_all/deepseek_coder_1.3b/{code_type}_code2query_{machine_rank}-of-{machine_num}.jsonl", 'r') as f:
                completed_len = len(f.readlines())
        self.sources = self.sources[completed_len:]
        self.query_types = self.query_types[completed_len:]
        self.confidences = self.confidences[completed_len:]
        self.ids = self.ids[completed_len:]

        print(len(self.sources))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.sources[i], id=self.ids[i], query_type=self.query_types[i], confidence=self.confidences[i])


def padding(inputs, padding_token, cutoff = None):
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
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, ids, query_type, confidence = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", 'id', 'query_type', 'confidence'))
        # input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 1024)
        # labels = padding(labels, IGNORE_INDEX, cutoff = 1024)

        return dict(
            input_ids=input_ids,
            id=ids,
            query_type=query_type,
            confidence=confidence,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, machine_rank, machine_num, code_type) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, machine_rank=machine_rank,
                                               machine_num=machine_num, code_type=code_type)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


def main(args):

    # dist.init_process_group("nccl")
    torch.manual_seed(args.seed)
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size

    code_type = args.code_type
    num_itera = args.num_itera

    machine_rank = args.machine_rank
    machine_num = args.machine_num

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model,
    #     )
    # if "llama" in base_model:
    #     tokenizer.add_special_tokens(
    #         {
    #             "eos_token": DEFAULT_EOS_TOKEN,
    #             "bos_token": DEFAULT_BOS_TOKEN,
    #             "unk_token": DEFAULT_UNK_TOKEN,
    #         }
    #     )
    # tokenizer.truncation_side = 'left'

    # print('model loaded')

    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path, machine_rank, machine_num, code_type)
    dataloader = DataLoader(
            eval_dataset, 
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=batch_size
    )

    print('data loaded')

    dataloader = accelerator.prepare(dataloader)

    import os
    num_processes = accelerator.num_processes
    process_index = accelerator.process_index
    print(num_processes, '==num_processes==', process_index, '==process_index==')

    from vllm import LLM, SamplingParams
    import os
    os.environ['VLLM_USE_MODELSCOPE'] = 'False'
    model = base_model
    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
                   "Response:", "Response"]
    tensor_parallel_size = 1
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, swap_space=64, dtype='bfloat16', tokenizer_pool_size=1, max_model_len=2048)

    import numpy as np

    os.makedirs(args.out_path, exist_ok = True)
    rank = torch.distributed.get_rank()
    data_path = os.path.join(args.out_path, '{}_code2query_{}-of-{}.jsonl'.format(code_type, machine_rank, machine_num))
    
    @accelerator.on_process(process_index=process_index)
    def write2json(fwobj, tmp):
        fwobj.write(json.dumps(tmp, ensure_ascii=False)+'\n')
    
    @accelerator.on_process(process_index=process_index)
    def close_fwobj(fwobj):
        fwobj.close()
        
    @accelerator.on_process(process_index=process_index)
    def open_fwobj(data_path):
        return open(data_path, 'w')

    fwobj = open_fwobj(data_path)
    print(fwobj, '==', process_index)

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        prompt = batch['input_ids']
        idx = batch['id']
        query_types = batch['query_type']
        confidences = batch['confidence']
        sampling_params = SamplingParams(n=num_itera, temperature=0.7, top_p=1, max_tokens=1024,
                                         use_beam_search=False, stop=stop_tokens)
        completions = llm.generate(prompt, sampling_params)
        for output, id_, query_type, confidence in zip(completions, idx, query_types, confidences):
            prompt_temp = output.prompt
            for o in output.outputs:
                tmp = {
                    'id':id_,
                    'source_code':'\n'.join(prompt_temp.split('\n')[3:-3]),
                    'query':o.text,
                    'query_type': query_type,
                    'confidence': confidence,
                }
                write2json(fwobj, tmp)
    #
    # for _ in range(num_itera):
    #     for tempera in [0.7]:
    #         generation_config = GenerationConfig(
    #             # temperature=0.8 if args.diverse_beam > 1 else 1.0,
    #             temperature=tempera,
    #             # num_beam_groups=args.diverse_beam,
    #             # diversity_penalty=1.0,
    #             do_sample=True,
    #             num_beams=return_seq_num,
    #             max_new_tokens=512,
    #             num_return_sequences=return_seq_num,
    #         )
    #
    #         all_outputs = []
    #         for step, batch in tqdm(enumerate(dataloader)):
    #             # if step > 10:
    #             #     break
    #             # print(batch.pop('id'))
    #             # print(dataset_for_eval[step]['prompt'])
    #             input_ids = batch['input_ids'].to(accelerator.device)
    #             attention_mask = batch['attention_mask'].to(accelerator.device)
    #             generation_output = None
    #             with torch.no_grad():
    #                 # try:
    #                 if hasattr(model, 'generate'):
    #                     generation_output = model.generate(
    #                         input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         generation_config=generation_config,
    #                         return_dict_in_generate=True,
    #                         synced_gpus=True,
    #                     )
    #                 else:
    #                     generation_output = model.module.generate(
    #                         input_ids=input_ids,
    #                         attention_mask=attention_mask,
    #                         generation_config=generation_config,
    #                         return_dict_in_generate=True,
    #                         synced_gpus=True,
    #                     )
    #                 # except:
    #                 #     inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #                 #     print(inputs_string, '==invalid==')
    #                 #     continue
    #
    #                 if generation_output:
    #                     sequences = generation_output.sequences
    #                     outputs_string = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    #                     inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    #                     # outputs_ids = batch['id'].data.cpu().numpy().tolist()
    #
    #             if np.mod(step, 100) == 0:
    #                 print(inputs_string, '==inputs_string==', outputs_string)
    #
    #             for idx in range(len(inputs_string)):
    #                 tmp = {
    #                     # 'id':int(outputs_ids[idx]),
    #                     'input':inputs_string[idx],
    #                     'output':outputs_string[idx],
    #                 }
    #                 write2json(fwobj, tmp)
        
    close_fwobj(fwobj)
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--code_type", default="", type=str, help="config path")
    parser.add_argument("--num_itera", default=1, type=int, help="config path")

    parser.add_argument("--machine_rank", default=0, type=int, help="config path")
    parser.add_argument("--machine_num", default=1, type=int, help="config path")
    args = parser.parse_args()

    main(args)
