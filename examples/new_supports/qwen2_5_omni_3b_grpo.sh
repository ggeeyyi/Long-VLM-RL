#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-Omni-3B # replace it with your local file path
export VLLM_USE_V1=0
DATA_PATH=$1
VIDEO_PATH=$2

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=Perflow-Shuai/videos_for_audio@train \
    data.val_files=Perflow-Shuai/videos_for_audio@train \
    data.video_dir=$VIDEO_PATH \
    data.is_omni=true \
    data.audio_max_length=10000 \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.is_omni=true \
    worker.actor.padding_free=false \
    worker.actor.ulysses_size=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_model_len=20000\
    worker.rollout.max_num_batched_tokens=100000 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_5_omni_3b \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=8
