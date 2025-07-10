set -x

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct  # replace it with your local file path
DATA_PATH=$1 # For example, OpenR1-Math-220k

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.max_response_length=2048 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    trainer.experiment_name=qwen2_5_3b_math_grpo \
    trainer.save_freq=100