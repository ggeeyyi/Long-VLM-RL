#!/bin/bash
train_script=$1

DEFAULT_MASTER_ADDR="127.0.0.1"
DEFAULT_MASTER_PORT=25001

echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"

NNODES=${SLURM_JOB_NUM_NODES:-1}
echo "NNODES = $NNODES"

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
echo "NODES = $NODES"

NODE_RANK=${SLURM_PROCID:-0}
echo "NODE_RANK = $NODE_RANK"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=${MASTER_ADDR:-$DEFAULT_MASTER_ADDR}
echo "MASTER_ADDR = $MASTER_ADDR"

if [ $NODE_RANK -eq 0 ]; then

ray start --block --head --port=6379 &
sleep 10
nnodes=$NNODES bash $train_script

else

# Wait until head node is ready
until nc -z ${MASTER_ADDR} 6379; do
    echo "Waiting for Ray head at ${MASTER_ADDR}:6379..."
    sleep 2
done

ray start --block --address=${MASTER_ADDR}:6379

fi