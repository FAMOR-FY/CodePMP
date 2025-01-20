# 指定目录
base_dir="/root_path/CodePMP/output/pmp_rm1_lm1_7b_1b_all_wsd/qwen_7b"
mkdir /root_path/CodePMP/logs/pmp_rm1_lm1_7b_1b_all_wsd_decay_qwen_7b/

# 获取所有子目录并按数字升序排序
subdirs=$(find "$base_dir" -type d -name "checkpoint-*" | sort -V)
subdirs_array=($subdirs)

# 遍历每个子目录
for ((i=0; i<${#subdirs_array[@]}-1; i++)); do
    dir=${subdirs_array[i]}
    checkpoint=$(basename "$dir")
    bash /root_path/CodePMP/pmp_rm1_lm1_7b_1b_all_wsd_decay_qwen_7b.sh $checkpoint  2>&1 | tee -a /root_path/CodePMP/logs/pmp_rm1_lm1_7b_1b_all_wsd_decay_qwen_7b/pmp_rm1_lm1_7b_1b_all_wsd_decay_$checkpoint_qwen_7b_rank${RANK}.log
done