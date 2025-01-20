current_datetime=$(date '+%Y-%m-%d-%H:%M:%S')
MODEL_DIR=/root_path
SAVE_DIR=/root_path/CodePMP
ROOT_DIR=/root_path/CodePMP

N_GPUS=8

MACHINE_NUM=$1
MACHINE_RANK=$2
NUM_PROCESSES=$(expr $N_GPUS \* $MACHINE_NUM)

echo "Number of machine: $NUM_MACHINE"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "RANK: $RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

export PYTHONSOURCE=mirrors.cloud.tencent.com
# pip3 install wandb
# pip3 install -r ./requirements.txt
# pip3 install transformers --upgrade
# pip3 install accelerate --upgrade
# pip3 install deepspeed==0.13.0
# pip3 install trl --upgrade

export WANDB_API_KEY=XXX
wandb login $WANDB_API_KEY
echo "wandb is configured and ready to use."
mkdir ${ROOT_DIR}/data/code2query_all/
mkdir ${ROOT_DIR}/data/code2query_all/deepseek_coder_1.3b

# 用于存放子进程的PID
pids=()
set -x
for LOCAL_RANK in $(seq 0 $((N_GPUS - 1)))
do  
    SPLIT=$(expr $LOCAL_RANK \+ $MACHINE_RANK \* 8)
    CUDA_VISIBLE_DEVICES=$LOCAL_RANK python ${ROOT_DIR}/query_summarizer_inference_all.py \
    --base_model ${MODEL_DIR}/models/deepseek-coder-1.3b-instruct\
    --data_path ${ROOT_DIR}/data/code2type_all/qwen2_1.5b \
    --batch_size 1024 \
    --out_path ${ROOT_DIR}/data/code2query_all/deepseek_coder_1.3b \
    --code_type notebook \
    --machine_rank ${SPLIT} \
    --machine_num ${NUM_PROCESSES} &
    pids+=($!)
    sleep 10s
done

# 等待所有后台进程结束
for pid in ${pids[*]}; do
    wait $pid
done

echo "All processes have completed."