# ====================== Config GPU Environment ====================== #
export GLOO_SOCKET_IFNAME=eth0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ====================== Config Model and Training ====================== #
current_datetime=$(date '+%Y-%m-%d-%H-%M-%S')
MODEL_DIR=/root_path
SAVE_DIR=/root_path/CodePMP

N_GPUS=8
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# pip3 install wandb  -i https://mirrors.cloud.tencent.com/pypi/simple 
pip3 install transformers --upgrade
pip3 install accelerate --upgrade
# pip3 install deepspeed==0.13.0

export WANDB_API_KEY=XXX
wandb login $WANDB_API_KEY
echo "wandb is configured and ready to use."

mkdir -p /root_path/CodePMP/output/pmp_rm1_lm1_7b_1b_all_wsd/qwen_7b

accelerate launch \
--main_process_ip="${MASTER_ADDR}" \
--main_process_port="${MASTER_PORT:-"2333"}" \
--machine_rank="${RANK}" \
--num_processes="${NUM_PROCESSES}" \
--num_machines="${WORLD_SIZE}" \
    /root_path/CodePMP/pmp.py \
--do_train \
--model_name_or_path /root_path/models/Qwen2-7B \
--data_path /root_path/CodePMP/data/query2code_all/deepseek_coder_6.7b_deepseek_coder_1.3b \
--bf16 True \
--output_dir /root_path/CodePMP/output/pmp_rm1_lm1_7b_1b_all_wsd/qwen_7b \
--num_train_epochs 1 \
--model_max_length 1024 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_interval 1.1892 \
--save_total_limit 100 \
--learning_rate 1e-6 \
--weight_decay 0.1 \
--warmup_ratio 0.03 \
--lr_scheduler_kwargs '{"num_stable_steps": -1, "decay_ratio": 0.1, "min_lr_ratio": 0}' \
--rm_coef 1.0 \
--lm_coef 1.0 \
--lr_scheduler_type "warmup_stable_decay" \
--logging_steps 10 \
--gradient_checkpointing True \
--run_name pmp_rm1_lm1_7b_1b_all_wsd_qwen_7b_${current_datetime} \
--deepspeed /root_path/CodePMP/config/zero2_config.json \
--tf32 True
