"""Reward shaping for tau2-bench using environment signals.

This module provides a custom_reward_post_process function for slime that adds
partial credit based on tau2's intermediate evaluation signals (action_checks,
communicate_checks, env_assertions).

Usage:
    --custom-reward-post-process-path tau2_reward_shaping.tau2_reward_post_process

Formula (v2 - dynamic weight normalization):
    R_train = task_reward + alpha * partial_score

    Only components with actual checks are included. Weights are renormalized
    over present components to avoid inflated scores from missing check types.

    Base weights: action=0.7, communicate=0.1, assertion=0.2
    If no checks present, partial_score = 0.0 (no free shaping)
"""

import json
import logging
from typing import TYPE_CHECKING, List, Union

import torch

# Lazy import to avoid pulling in full slime dependency chain at module load
# The Sample type is only needed for type hints
if TYPE_CHECKING:
    from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Shaping hyperparameters (v2)
ALPHA = 0.5  # Increased from 0.25 - stronger shaping signal
ACTION_WEIGHT = 0.7  # Tool-calling is the core skill (increased from 0.5)
COMMUNICATE_WEIGHT = 0.1  # Smaller role (decreased from 0.3)
ASSERTION_WEIGHT = 0.2  # Environment state correctness


def compute_partial_score(reward_info: dict) -> float:
    """Compute partial credit from tau2 reward_info with dynamic weight normalization.

    Only includes components that have actual checks. Renormalizes weights over
    present components to provide a more discriminative signal.

    Args:
        reward_info: Dict containing action_checks, communicate_checks, env_assertions

    Returns:
        Weighted partial score in [0, 1], or 0.0 if no checks present
    """
    scores = []
    weights = []

    # Action progress: fraction of required actions matched
    action_checks = reward_info.get("action_checks") or []
    if action_checks:
        matched = sum(1 for ac in action_checks if ac.get("action_match"))
        action_progress = matched / len(action_checks)
        scores.append(action_progress)
        weights.append(ACTION_WEIGHT)

    # Communication progress: fraction of required info communicated
    comm_checks = reward_info.get("communicate_checks") or []
    if comm_checks:
        met = sum(1 for cc in comm_checks if cc.get("met"))
        comm_progress = met / len(comm_checks)
        scores.append(comm_progress)
        weights.append(COMMUNICATE_WEIGHT)

    # Assertion progress: fraction of env assertions met
    env_assertions = reward_info.get("env_assertions") or []
    if env_assertions:
        met = sum(1 for ea in env_assertions if ea.get("met"))
        assertion_progress = met / len(env_assertions)
        scores.append(assertion_progress)
        weights.append(ASSERTION_WEIGHT)

    # No checks = no shaping (return 0.0, not inflated 1.0)
    if not scores:
        return 0.0

    # Renormalize weights over present components
    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


def tau2_reward_post_process(
    args, samples: Union[List["Sample"], List[List["Sample"]]]
) -> tuple:
    """Custom reward post-processing for tau2 with shaping.

    This function is called by slime's RolloutManager._post_process_rewards
    via the --custom-reward-post-process-path CLI flag.

    Args:
        args: Training arguments (has advantage_estimator, rewards_normalization, etc.)
        samples: List of Sample objects from rollout

    Returns:
        (raw_rewards, shaped_rewards) tuple where:
        - raw_rewards: original binary tau2 rewards (for logging/eval)
        - shaped_rewards: training rewards with partial credit (possibly normalized)
    """
    raw_rewards = []
    shaped_rewards = []

    for sample in samples:
        task_reward = sample.get_reward_value(args)
        raw_rewards.append(task_reward)

        # Guard: ensure metadata exists
        if sample.metadata is None:
            sample.metadata = {}

        # Extract reward_info - may be JSON string or dict
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

        # Store in metadata for logging
        sample.metadata["raw_reward"] = task_reward
        sample.metadata["partial_score"] = partial_score

        logger.debug(
            f"Shaping: task={task_reward:.2f}, partial={partial_score:.2f}, shaped={shaped:.2f}"
        )

    # Apply GRPO normalization to shaped rewards (mirrors slime's default behavior)
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
