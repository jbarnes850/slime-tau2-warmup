# Tau2 Telecom Warmup on slime

This folder contains a minimal Tau2 telecom warmup on top of slime, using Qwen3-4B-Instruct and the Tau2 dual-control environment.

The core pieces are:
- `tau2_reward_shaping.py` – partial-credit reward shaping using Tau2 `action_checks`, `communicate_checks`, and `env_assertions`.
- `tau2_mock.py` – preprocess Tau2 telecom tasks into JSONL for slime.
- `evaluate_tau2.py` – evaluate a trained checkpoint on Tau2 telecom test tasks (pass@1 + partial scores).

## 1. Environment and data

Follow `examples/tau-bench/SETUP_GUIDE.md` up through:
- cloning `THUDM/slime` and `sierra-research/tau2-bench`
- installing `tau2-bench` and required Python packages
- downloading and converting `Qwen/Qwen3-4B-Instruct`
- running `tau2_mock.py` to produce `telecom_train_tasks.jsonl` / `telecom_test_tasks.jsonl`

Key command:

```bash
cd /root/slime
python examples/tau-bench/tau2_mock.py --local_dir /root/tau2-bench/
```

## 2. SFT warmup (telecom teacher traces)

Use the SFT job from `SETUP_GUIDE.md` to train a small SFT model on teacher traces:

```bash
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh

RUNTIME_ENV_JSON='{"working_dir": "/root/slime", "env_vars": {"PYTHONPATH": "/root/Megatron-LM/:/root/slime/examples/tau-bench", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}'

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --colocate \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-4B-Instruct-2507/ \
  --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/ \
  --load /root/Qwen3-4B-Instruct-2507_slime/ \
  --save /root/Qwen3-4B-Instruct-2507_sft_telecom/ \
  --prompt-data /root/teacher_traces.jsonl \
  --input-key prompt \
  --num-rollout 49 \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 2048 \
  --rollout-temperature 1.0 \
  --global-batch-size 8 \
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --optimizer adam \
  --lr 2e-5 \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.1 \
  --weight-decay 0.01 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.7 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  ${DISTRIBUTED_ARGS[@]}
```

This produces SFT checkpoints under `/root/Qwen3-4B-Instruct-2507_sft_telecom/`.

## 3. GRPO baseline + reward shaping

For RL, use the Tau2 telecom setup from `SETUP_GUIDE.md`, then enable reward shaping:

```bash
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh

export TAU2_DATA_DIR=/root/tau2-bench/data
export GEMINI_API_KEY="your-gemini-api-key"

RUNTIME_ENV_JSON='{
  "working_dir": "/root/slime",
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/:/root/slime/examples/tau-bench",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "TAU2_DATA_DIR": "/root/tau2-bench/data",
    "GEMINI_API_KEY": "'${GEMINI_API_KEY}'"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 2 \
  --rollout-num-gpus 2 \
  --colocate \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen3-4B-Instruct-2507/ \
  --ref-load /root/Qwen3-4B-Instruct-2507_torch_dist/ \
  --load /root/Qwen3-4B-Instruct-2507_sft_telecom/iter_0000009/ \
  --save /root/Qwen3-4B-Instruct-2507_rl_telecom/ \
  --prompt-data /root/tau2-bench/telecom_train_tasks.jsonl \
  --input-key index \
  --num-rollout 74 \
  --rollout-batch-size 8 \
  --n-samples-per-prompt 4 \
  --rollout-max-response-len 4096 \
  --rollout-temperature 0.7 \
  --global-batch-size 8 \
  --custom-generate-function-path generate_with_tau.generate \
  --custom-reward-post-process-path tau2_reward_shaping.tau2_reward_post_process \
  --tensor-model-parallel-size 2 \
  --sequence-parallel \
  --pipeline-model-parallel-size 1 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.1 \
  --weight-decay 0.01 \
  --rollout-num-gpus-per-engine 1 \
  --sglang-mem-fraction-static 0.7 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32
```

This uses `tau2_reward_shaping.py` to add partial credit based on Tau2 environment signals while keeping the original binary reward for evaluation.

## 4. Evaluation on Tau2 telecom

After RL, evaluate a checkpoint on Tau2 telecom test tasks with `evaluate_tau2.py`:

```bash
cd /root/slime/examples/tau-bench

python evaluate_tau2.py \
  --checkpoint /root/Qwen3-4B-Instruct-2507_rl_telecom/iter_0000100/ \
  --output-path /root/tau2_eval_results.json \
  --num-tasks 25 \
  --sglang-port 30000 \
  --domain telecom \
  --task-split test
```

The script reports:
- pass@1 success rate
- average reward
- average partial score (using Tau2 `reward_info`)

