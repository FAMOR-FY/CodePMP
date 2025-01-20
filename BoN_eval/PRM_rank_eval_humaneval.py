import json
import os
import re
from tqdm import tqdm
from math import log, exp
from math_equivalence import last_boxed_only_string, is_equiv, _strip_string
import argparse
import random
from vllm import LLM, SamplingParams

random.seed(123)
chat_prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>human
{}<|im_end|>
<|im_start|>gpt
"""

compare_prompt_template = """## 任务描述\n\n你是一个数学老师，学生提交了题目的解题步骤，你需要参考`题干`，`解析`和`答案`，判断`学生解题步骤`的结果是否正确。忽略`学生解题步骤`中的错误，只关注最后的答案。答案可能出现在`解析`中，也可能出现在`答案`中。\n\n## 输入内容\n\n题干:\n\n```\n{question}\n```\n\n解析:\n\n```\n{analysis}\n\n```\n\n答案:\n\n```\n{answer}\n```\n\n学生解题步骤:\n\n```\n{pred_step}\n```\n\n输出:"""
def generate_compare_prompt(question, answer, pred):
    return compare_prompt_template.format_map({
        "question": question,
        "analysis": answer,
        "answer": "",
        "pred_step": pred
    })

def generate_base_prompt(compare_prompt):
    return chat_prompt.format(compare_prompt)

def get_query(input):
    query = re.findall('Instruction:(.+?)Response:', input, re.DOTALL)[0].strip()
    return query

def get_steps(input, prefix_len):
    res = re.findall('Response:(.*)', input, re.DOTALL)[0].strip()
    if res:
        prefix_steps = res[:prefix_len]
        step = res[prefix_len:].strip()
        return prefix_steps, step, len(res)
    else:
        return '', '', 0

def aggregation_min(rewards_list):
    if rewards_list:
        return min(rewards_list)
    else:
        return -1

def no_aggregation(rewards_list):
    if rewards_list:
        return rewards_list[0]
    else:
        return -1

def aggregation_max(rewards_list):
    if rewards_list:
        return max(rewards_list)
    else:
        return -1

def aggregation_sum_logprob(rewards_list):
    if rewards_list:
        return sum([log(r+1e-5) for r in rewards_list])
    else:
        return -1

def aggregation_sum(rewards_list):
    if rewards_list:
        return sum(rewards_list)
    else:
        return -1

def aggregation_sum_logit(rewards_list):
    if rewards_list:
        return sum([log((r+1e-5)/(1-r+1e-5)) for r in rewards_list])
    else:
        return -1

def aggregation_mean_odd(rewards_list):
    if rewards_list:
        return sum([log((r+1e-5)/(1-r+1e-5)) for r in rewards_list])/len(rewards_list)
    else:
        return -1

def aggregation_only_last(rewards_list):
    if rewards_list:
        return rewards_list[-1]
    else:
        return -1


def main(root_path, dataset_name, generator_name, BoN=256):
    # model_path = '/root_path/models/DeepSeek-7B-Math-Compare-Answer'
    # llm = LLM(model=model_path, tensor_parallel_size=1, swap_space=64, dtype='bfloat16', tokenizer_pool_size=1, max_model_len=2048, gpu_memory_utilization=0.9, trust_remote_code=True)
    
    fps = [fp for fp in os.listdir(root_path) if f'{dataset_name}_test_rank' in fp and generator_name in fp]
    qid2data = {}
    fps.sort(key=lambda x: int(x.split('.')[-2].split('_')[-1]))
    for fp in fps:
        with open(os.path.join(root_path, fp), 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line)
                if line['id'] in qid2data:
                    qid2data[line['id']].append(line)
                elif line['id'] not in qid2data:
                    qid2data[line['id']] = [line]

    # for qid in qid2data.keys():
    #     if len(qid2data[qid]) > BoN:
    #         res_set = set()
    #         l = len(qid2data[qid])
    #         for i in range(1, len(qid2data[qid]) + 1):
    #             if qid2data[qid][l - i]['input'] in res_set:
    #                 del qid2data[qid][l - i]
    #             else:
    #                 res_set.add(qid2data[qid][l - i]['input'])
    #         assert len(qid2data[qid]) == BoN

    data = []
    all_base_prompts = []
    for qid in qid2data.keys():
        items = random.sample(qid2data[qid], args.BoN)
        id = items[0]['id']
        # print(items)
        try:
            if 'reclor' in dataset_name or 'DROP' in dataset_name or 'logiqa2' in dataset_name:
                query = items[0]['input'].split('\n\nResponse:')[0]
            else:
                query = get_query(items[0]['input'])
        except:
            print(items[0]['input'])
            continue    
        answer = items[0]['answer']
        solution_rewards = []
        for item_id in range(len(items)):
            item = items[item_id]
            assert item['id'] == id
            if args.aggregation_function == 'min':
                reward = aggregation_min(item['rewards'])
            elif args.aggregation_function == 'max':
                reward = aggregation_max(item['rewards'])
            elif args.aggregation_function == 'mean_odd':
                reward = aggregation_mean_odd(item['rewards'])
            elif args.aggregation_function == 'sum_logit':
                reward = aggregation_sum_logit(item['rewards'])
            elif args.aggregation_function == 'sum':
                reward = aggregation_sum(item['rewards'])
            elif args.aggregation_function == 'sum_logprob':
                reward = aggregation_sum_logprob(item['rewards'])
            elif args.aggregation_function == 'only_last':
                reward = aggregation_only_last(item['rewards'])
            elif args.aggregation_function == 'no_aggregation':
                reward = no_aggregation(item['rewards'])
            else:
                raise EOFError
            solution_rewards.append((item_id, reward))
        solution_rewards.sort(key=lambda x: x[1], reverse=True)
        try:
            if 'reclor' in dataset_name or 'DROP' in dataset_name or 'logiqa2' in dataset_name:
                best_res = items[solution_rewards[0][0]]['input'].split('\n\nResponse:')[1]
            else:
                best_res = re.findall('Response:(.*)', items[solution_rewards[0][0]]['input'], re.DOTALL)[
                0].strip().replace(' ки', '')
                match = re.search(r'```python(.*?)```', best_res, re.DOTALL)
                best_res = match.group(1).strip() if match else best_res
        except:
            print(items[solution_rewards[0][0]]['input'])
            continue    
        # compare_prompt = generate_compare_prompt(query, answer, best_res)
        # base_prompt = generate_base_prompt(compare_prompt)
        # all_base_prompts.append(base_prompt)

        tmp = {
            'task_id': id,
            'completion': best_res,
            # 'question': query,
            # 'response': best_res,
            # 'amswer': "",
            # 'is_correct': None,
        }
        data.append(tmp)

    # sampling_params = SamplingParams(
    #     temperature=0.0,
    #     top_p=1.0,
    #     max_tokens=256,
    #     repetition_penalty=1.0,
    #     skip_special_tokens=False,
    #     stop=["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction",
    #                "Response:", "Response", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<|im_end|>"],
    # )
    
    # outputs = llm.generate(all_base_prompts, sampling_params)

    # for output, d in zip(outputs, data):
    #     d['is_correct'] = (output.outputs[0].text == '<correct>')

    # acc = sum([d['is_correct'] for d in data]) / len(data)
    with open(os.path.join(root_path, '{}_test_best_of_{}_generated_by_{}_{}.jsonl'.format(dataset_name, BoN, generator_name, args.aggregation_function)), 'w') as f:
        f.write('\n'.join([json.dumps(d) for d in data]))
    # json.dump({'acc': acc}, open(os.path.join(root_path, '{}_test_acc_best_of_{}_generated_by_{}_{}.json'.format(dataset_name, BoN, generator_name, args.aggregation_function)), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--root_path", default="", type=str, help="model path")
    parser.add_argument("--dataset_name", default="", type=str, help="model path")
    parser.add_argument("--generator_name", default="", type=str, help="model path")
    parser.add_argument("--BoN", default=256, type=int, help="model path")
    parser.add_argument("--aggregation_function", default='min', type=str, help="model path")
    args = parser.parse_args()
    main(root_path=args.root_path, dataset_name=args.dataset_name, BoN=args.BoN, generator_name=args.generator_name)
