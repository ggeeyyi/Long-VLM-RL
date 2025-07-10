#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=stabilityai/stable-diffusion-3.5-medium

python3 -m verl.trainer.main \
    config=examples/config_diffusion.yaml \
    worker.actor.model.trust_remote_code=true \
    worker.rollout.trust_remote_code=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=image_diffusion_grpo \
    trainer.n_gpus_per_node=8
