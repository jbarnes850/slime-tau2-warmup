"""Reward shaping for tau2-bench using environment signals.

Usage:
    --custom-reward-post-process-path tau2_reward_shaping.tau2_reward_post_process

Formula: R_train = task_reward + alpha * partial_score
Weights are renormalized over present components (action=0.7, communicate=0.1, assertion=0.2).
"""

import json
import logging
from typing import TYPE_CHECKING, List, Union

import torch

if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

ALPHA = 0.5
ACTION_WEIGHT = 0.7
COMMUNICATE_WEIGHT = 0.1
ASSERTION_WEIGHT = 0.2


def compute_partial_score(reward_info: dict) -> float:
    """Compute partial credit from tau2 reward_info with dynamic weight normalization."""
    scores = []
    weights = []

    action_checks = reward_info.get("action_checks") or []
    if action_checks:
        matched = sum(1 for ac in action_checks if ac.get("action_match"))
        scores.append(matched / len(action_checks))
        weights.append(ACTION_WEIGHT)

    comm_checks = reward_info.get("communicate_checks") or []
    if comm_checks:
        met = sum(1 for cc in comm_checks if cc.get("met"))
        scores.append(met / len(comm_checks))
        weights.append(COMMUNICATE_WEIGHT)

    env_assertions = reward_info.get("env_assertions") or []
    if env_assertions:
        met = sum(1 for ea in env_assertions if ea.get("met"))
        scores.append(met / len(env_assertions))
        weights.append(ASSERTION_WEIGHT)

    if not scores:
        return 0.0

    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def tau2_reward_post_process(
    args, samples: Union[List["Sample"], List[List["Sample"]]]
) -> tuple:
    """Custom reward post-processing for tau2 with shaping."""
    raw_rewards = []
    shaped_rewards = []

    for sample in samples:
        task_reward = sample.get_reward_value(args)
        raw_rewards.append(task_reward)

        if sample.metadata is None:
            sample.metadata = {}

        reward_info_data = sample.metadata.get("reward_info", {})
        if isinstance(reward_info_data, str):
            try:
                reward_info = json.loads(reward_info_data)
            except json.JSONDecodeError:
                reward_info = {}
        else:
            reward_info = reward_info_data

        partial_score = compute_partial_score(reward_info)
        shaped = task_reward + ALPHA * partial_score
        shaped_rewards.append(shaped)

        sample.metadata["raw_reward"] = task_reward
        sample.metadata["partial_score"] = partial_score

        logger.debug(f"Shaping: task={task_reward:.2f}, partial={partial_score:.2f}, shaped={shaped:.2f}")

    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(shaped_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, shaped_rewards
