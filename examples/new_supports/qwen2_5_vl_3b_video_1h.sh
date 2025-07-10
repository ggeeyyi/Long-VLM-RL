#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
nnodes=${nnodes:-1}

MODEL_PATH=path_to/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
VIDEO_PATH=$1
EXP_NAME=qwen2_5_vl_3b_video_1h
num_video_frames=3600

num_video_frames=$1
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=LongVideo-Reason/longvideo-reason@train \
    data.val_files=LongVideo-Reason/longvideo-reason@validation \
    data.video_dir=$VIDEO_PATH \
    data.num_workers=1 \
    data.rollout_batch_size=$((8 * $nnodes)) \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.actor.padding_free=true \
    worker.actor.ulysses_size=8 \
    worker.actor.global_batch_size=$((8 * $nnodes)) \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.num_video_frames=$num_video_frames \
    worker.rollout.tokens_per_frame=60 \
    worker.rollout.max_num_batched_tokens=256000 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.num_chunk_seq=60 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=1 \
    trainer.load_checkpoint_path=checkpoints/easy_r1/${EXP_NAME} \
    trainer.nnodes=${nnodes} \
    trainer.val_before_train=false \
    trainer.val_freq=-1
