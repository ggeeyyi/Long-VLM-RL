#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH=$1  # replace it with your local file path
VIDEO_PATH=$2
cp verl/utils/vila_remote_code/* ${MODEL_PATH}

ulysses_size=4
num_video_frames=256
tokens_per_frame=257
max_num_batched_tokens=$((num_video_frames*tokens_per_frame+8192))

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=LongVideo-Reason/longvideo-reason@train \
    data.val_files=LongVideo-Reason/longvideo-reason@validation \
    data.video_dir=$VIDEO_PATH \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.vila_model=true \
    worker.rollout.num_video_frames=$num_video_frames \
    worker.rollout.tokens_per_frame=$tokens_per_frame \
    worker.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    worker.rollout.trust_remote_code=true \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    worker.reward.reward_type=sequential \
    worker.vila_model=true \
    worker.actor.model.trust_remote_code=true \
    worker.actor.ulysses_size=$ulysses_size \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=longvila_7b_video \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.val_freq=-1