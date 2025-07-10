ACCOUNT=nvr_elm_llm
PARTITIONS=polar3,polar,batch_block1,batch_block2,batch_block3
train_script=$1
nnodes=$2
exp_name=$(basename "$train_script" .sh)
srun -o logs/${exp_name}_%J_%t.txt -p $PARTITIONS -A $ACCOUNT -N ${nnodes} -t 4:00:00 -J nvr_elm_llm:${exp_name} --ntasks-per-node=1 --gres gpu:8 --cpus-per-task=32 --exclusive bash scripts/train_multi_nodes.sh ${train_script}
