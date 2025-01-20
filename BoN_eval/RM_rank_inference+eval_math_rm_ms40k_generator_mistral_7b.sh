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

mkdir -p $1/test
BoN=(2 4 8 16 32 64 128 256)
accelerate launch ${SAVE_DIR}/BoN_eval/RM_rank_inference.py \
    --base_model $1 \
    --data_path /root_path/repos/hezuo/huimu/DG-PRM-master/DG-PRM/data/verify_data_math_500/mistral_7b \
    --batch_size 8 \
    --out_path $1/test \
    --port 2049 && \
for LOCAL_RANK in $(seq 0 $((N_GPUS - 1)))
do  
    SPLIT=${SPLITS[${LOCAL_RANK}]}
    CUDA_VISIBLE_DEVICES=$LOCAL_RANK python ${SAVE_DIR}/BoN_eval/PRM_rank_eval_by_model.py \
    --root_path $1/test \
    --dataset_name MATH \
    --generator_name mistral_7b \
    --BoN ${BoN[${LOCAL_RANK}]} \
    --aggregation_function no_aggregation &
    pids+=($!)
    sleep 10s
done

# 等待所有后台进程结束
for pid in ${pids[*]}; do
    wait $pid
done

echo "All processes have completed."