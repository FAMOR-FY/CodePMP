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
pip3 install transformers==4.45.1
pip3 install accelerate --upgrade
# pip3 install deepspeed==0.13.0

export WANDB_API_KEY=XXX
wandb login $WANDB_API_KEY
echo "wandb is configured and ready to use."
mkdir -p /root_path/CodePMP/output/RM_finetune_logiqa2_NR/qwen_7b_Qwen2-7B/eval
MODEL_PATH=/root_path/models/Qwen2-7B
DATA_PATH=/root_path/CodePMP/data/RM_data/logiqa2/processed_data_no_repeat
DATASET_NAME=logiqa2_NR
MODEL_PATH=/root_path/models/Qwen2-7B
accelerate launch \
    --main_process_ip="${MASTER_ADDR}" \
    --main_process_port="${MASTER_PORT:-"2333"}" \
    --machine_rank="${RANK}" \
    --num_processes="${NUM_PROCESSES}" \
    --num_machines="${WORLD_SIZE}" \
    /root_path/CodePMP/RM_finetune.py \
    --do_train \
    --model_name_or_path /root_path/models/Qwen2-7B \
    --data_path /root_path/CodePMP/data/RM_data/logiqa2/processed_data_no_repeat \
    --bf16 True \
    --output_dir /root_path/CodePMP/output/RM_finetune_logiqa2_NR/qwen_7b_Qwen2-7B \
    --dataset_name logiqa2_NR \
    --num_train_epochs 1 \
    --model_max_length 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 3e-6 \
    --weight_decay 0. \
    --max_grad_norm 1 \
    --warmup_ratio 0.25 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --run_name RM_finetune_logiqa2_NR_qwen_7b_Qwen2-7B_${current_datetime} \
    --deepspeed /root_path/CodePMP/config/zero2_config_cf.json \
    --tf32 True && \
python /root_path/CodePMP/RM_evaluate.py \
--model_path /root_path/CodePMP/output/RM_finetune_logiqa2_NR/qwen_7b_Qwen2-7B \
--data_path /root_path/CodePMP/data/RM_data/logiqa2/processed_data_no_repeat \
--out_path /root_path/CodePMP/output/RM_finetune_logiqa2_NR/qwen_7b_Qwen2-7B/eval \
--dataset_name logiqa2_NR