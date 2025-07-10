#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Wan-AI/Wan2.1-T2V-1.3B-Diffusers

python3 -m verl.trainer.main \
    config=examples/config_video_diffusion.yaml \
    worker.actor.model.trust_remote_code=true \
    worker.rollout.trust_remote_code=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=video_generation_grpo \
    trainer.n_gpus_per_node=8
