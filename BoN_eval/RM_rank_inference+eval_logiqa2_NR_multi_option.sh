# ====================== Config GPU Environment ====================== #
export GLOO_SOCKET_IFNAME=eth0
export CUDA_DEVICE_MAX_CONNECTIONS=1

# ====================== Config Model and Training ====================== #
current_datetime=$(date '+%Y-%m-%d-%H:%M:%S')
MODEL_DIR=/root_path/huggingface
SAVE_DIR=/root_path/CodePMP

N_GPUS=8
NUM_PROCESSES=$(expr $N_GPUS \* $WORLD_SIZE)
echo "Number of process: $NUM_PROCESSES"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# pip3 install wandb  -i https://mirrors.cloud.tencent.com/pypi/simple 
# pip3 install transformers --upgrade
# pip3 install accelerate --upgrade
# pip3 install deepspeed==0.13.0

export WANDB_API_KEY=XXX
wandb login $WANDB_API_KEY
echo "wandb is configured and ready to use."
echo $1
mkdir -p $1/test
CUDA_VISIBLE_DEVICES=$2 accelerate launch ${SAVE_DIR}/BoN_eval/RM_rank_inference_multi_option.py \
    --base_model $1 \
    --data_path /root_path/CodePMP/data/RM_data/logiqa2/data/test.jsonl \
    --batch_size 8 \
    --out_path $1/test && \

CUDA_VISIBLE_DEVICES=$2 python ${SAVE_DIR}/BoN_eval/PRM_rank_eval_multi_option.py \
--root_path $1/test \
--dataset_name logiqa2 \
--generator_name mistral_7b \
--aggregation_function no_aggregation

echo "All processes have completed."