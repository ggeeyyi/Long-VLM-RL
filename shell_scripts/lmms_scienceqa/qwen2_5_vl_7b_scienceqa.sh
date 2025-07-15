#!/bin/bash

set -x

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # Actor model path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=GY2233/lmms-ScienceQA@train \
    data.val_files=GY2233/lmms-ScienceQA@test \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    worker.actor.padding_free=false \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:sqa_compute_score \
    trainer.experiment_name=qwen2_5_vl_3b_7b_lmms_scienceqa \
    trainer.n_gpus_per_node=4
