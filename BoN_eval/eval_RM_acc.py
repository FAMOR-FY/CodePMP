import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

pmp_scaling_list = [
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-51_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-73_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-103_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-146_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-206_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-292_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-412_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-582_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-824_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-980_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-1165_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-1385_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-1647_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-1959_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-2330_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-2771_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-3295_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-3918_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-4659_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-5540_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-6588_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-7834_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-9316_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-11079_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-13175_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-15668_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-18632_decay_ckpt',
    # 'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-22157_decay_ckpt',
    'pmp_rm1_lm1_7b_1b_all_wsd_decay_checkpoint-24712_decay_ckpt',
]

# 解析命令行参数
parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--root_path', type=str, required=True, help='Root path for the experiment')
parser.add_argument('--scaling_path', type=str, required=True, help='Scaling path for the experiment')
parser.add_argument('--total', type=int)
parser.add_argument('--rank', type=int)
args = parser.parse_args()

root_path = args.root_path
scaling_path = args.scaling_path

# 定义exper2path字典
exper2path = {
    # 'main': [
    #     os.path.join(root_path, 'qwen_1.5b_Qwen2-1.5B'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'qwen_7b_Qwen2-7B'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen7b_qwen_7b'),
    # ],
    # 'data_construction': [
    #     os.path.join(root_path, 'pmp_rm1_lm1_1b_14M_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_1b_des_clip_14M_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_14M_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_1b_1b_des_clip_14M_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_des_clip_14M_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_14M_wsd_qwen_1.5b'),
    # ],
    # 'loss': [
    #     os.path.join(root_path, 'pmp_lm1_7b_1b_all_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_7b_1b_all_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b'),
    # ],
    # 'data_source': [
    #     os.path.join(root_path, 'pmp_rm1_lm1_stack_wsd_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b'),
    # ],
    # 'lr_shechuler': [
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_qwen_1.5b'),
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b'),
    # ],
    # 'special_token': [
    #     os.path.join(root_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b'),
    #     os.path.join(root_path.replace('RM_finetune', 'RM_finetune_EOC'), 'pmp_EOC_rm1_lm1_7b_1b_all_wsd_qwen_1.5b'),
    # ],
    # 'pmp_scaling_1.5b': [
    #     os.path.join(root_path, i) for i in pmp_scaling_list
    # ],
    # 'pmp_scaling_7b': [
    #     os.path.join(root_path, i.replace('decay_ckpt', 'qwen7b_decay_ckpt')) for i in pmp_scaling_list
    # ],
    # 'RM_scaling_1.5b': [
    #     os.path.join(scaling_path, 'qwen_1.5b_Qwen2-1.5B', str(i)) for i in sorted([int(i) for i in os.listdir(os.path.join(scaling_path, 'qwen_1.5b_Qwen2-1.5B')) if 'eval' not in i and 'test' not in i and 'run' not in i])
    # ],
    # 'RM_scaling_1.5b_pmp': [
    #     os.path.join(scaling_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b', str(i)) for i in sorted([int(i) for i in os.listdir(os.path.join(scaling_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen_1.5b')) if 'eval' not in i and 'test' not in i and 'run' not in i])
    # ],
    'RM_scaling_7b': [
        os.path.join(scaling_path, 'qwen_7b_Qwen2-7B', str(i)) for i in sorted([int(i) for i in os.listdir(os.path.join(scaling_path, 'qwen_7b_Qwen2-7B')) if 'eval' not in i and 'test' not in i and 'run' not in i])
    ],
    'RM_scaling_7b_pmp': [
        os.path.join(scaling_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen7b_qwen_7b', str(i)) for i in sorted([int(i) for i in os.listdir(os.path.join(scaling_path, 'pmp_rm1_lm1_7b_1b_all_wsd_qwen7b_qwen_7b')) if 'eval' not in i and 'test' not in i and 'run' not in i])
    ],
}

# 替换路径中的占位符
root_path = args.root_path
scaling_path = args.scaling_path

# 去重并创建目录
unique_paths = set()
for key, paths in exper2path.items():
    paths = [path.format(root_path=root_path, scaling_path=scaling_path) for path in paths]
    unique_paths.update(paths)
unique_paths = list(unique_paths)[args.rank * (len(unique_paths) // args.total):(args.rank + 1) * (len(unique_paths) // args.total) + (1 if args.rank < len(unique_paths) % args.total else 0)]

def process_path(path, rank):
    try:
        # 创建目录
        os.makedirs(os.path.join(path, 'eval'), exist_ok=True)
        os.makedirs(os.path.join(path, 'test'), exist_ok=True)
        
        # 运行Shell脚本
        subprocess.run(['bash', '/root_path/CodePMP/RM_ft_scaling_qwen_7b_logiqa2_recover.sh', path, str(rank)], check=True)
        # subprocess.run(['bash', '/root_path/CodePMP/RM_ft_scaling_qwen_7b_logiqa2_recover.sh', path, str(rank)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run script for path: {path}. Error: {e}")

# # 使用ThreadPoolExecutor和队列来管理rank
# rank_queues = [Queue() for _ in range(8)]

# with ThreadPoolExecutor(max_workers=8) as executor:
#     futures = []
#     for i, path in enumerate(unique_paths):
#         rank = i % 8  # 计算相对rank
#         rank_queues[rank].put(path)
    
#     for rank in range(8):
#         while not rank_queues[rank].empty():
#             path = rank_queues[rank].get()
#             futures.append(executor.submit(process_path, path, rank))
    
#     for future in as_completed(futures):
#         try:
#             future.result()
#         except Exception as e:
#             print(f"An error occurred: {e}")

from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# 假设 unique_paths 和 process_path 已经定义
# unique_paths = [...]
# def process_path(path, rank):
#     ...

# 创建8个队列
rank_queues = [Queue() for _ in range(8)]

# 将路径分配到队列中
for i, path in enumerate(unique_paths):
    rank = i % 8  # 计算相对rank
    rank_queues[rank].put(path)

def process_queue(queue, rank):
    while not queue.empty():
        path = queue.get()
        process_path(path, rank)

# 使用 ThreadPoolExecutor 来管理8个线程，每个线程处理一个队列
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for rank in range(8):
        futures.append(executor.submit(process_queue, rank_queues[rank], rank))
    
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            print(f"An error occurred: {e}")