#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on tau2-bench telecom tasks.

This script uses the same AgentGymEnv as training but runs evaluation
on the test split to measure pass@1.

Usage:
    # On the training server, after SFT or RL:
    python evaluate_tau2.py \
        --checkpoint /ephemeral/checkpoints/Qwen3-4B-sft-telecom-v2/iter_XXXXX/ \
        --output-path eval_results.json \
        --num-tasks 25  # or omit to run all test tasks
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standalone HTTP client for evaluation (doesn't require slime's init_http_client)
_eval_http_client: httpx.AsyncClient = None


async def _ensure_http_client():
    """Lazily initialize the HTTP client."""
    global _eval_http_client
    if _eval_http_client is None:
        _eval_http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10),
            timeout=httpx.Timeout(300.0),
        )
    return _eval_http_client


async def post_standalone(url: str, payload: dict, max_retries: int = 60):
    """Standalone async POST function for evaluation."""
    client = await _ensure_http_client()
    retry_count = 0
    while retry_count < max_retries:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            retry_count += 1
            logger.info(f"POST error: {e}, retrying... (attempt {retry_count}/{max_retries})")
            await asyncio.sleep(1.0)
    raise RuntimeError(f"Failed to POST to {url} after {max_retries} retries")

# Partial score computation (inline to avoid slime import dependencies)
# This mirrors the logic in tau2_reward_shaping.py
ACTION_WEIGHT = 0.5
COMMUNICATE_WEIGHT = 0.3
ASSERTION_WEIGHT = 0.2


def compute_partial_score(reward_info: dict) -> float:
    """Compute partial credit from tau2 reward_info.

    Empty check lists are treated as neutral (1.0) - no penalty for tasks
    that don't have certain check types.
    """
    action_checks = reward_info.get("action_checks") or []
    if action_checks:
        matched = sum(1 for ac in action_checks if ac.get("action_match"))
        action_progress = matched / len(action_checks)
    else:
        action_progress = 1.0

    comm_checks = reward_info.get("communicate_checks") or []
    if comm_checks:
        met = sum(1 for cc in comm_checks if cc.get("met"))
        comm_progress = met / len(comm_checks)
    else:
        comm_progress = 1.0

    env_assertions = reward_info.get("env_assertions") or []
    if env_assertions:
        met = sum(1 for ea in env_assertions if ea.get("met"))
        assertion_progress = met / len(env_assertions)
    else:
        assertion_progress = 1.0

    return (
        ACTION_WEIGHT * action_progress
        + COMMUNICATE_WEIGHT * comm_progress
        + ASSERTION_WEIGHT * assertion_progress
    )


@dataclass
class EvalResult:
    task_id: str
    task_index: int
    reward: float
    success: bool
    steps: int
    status: str
    error: Optional[str] = None
    reward_info: Optional[str] = None
    partial_score: Optional[float] = None


async def evaluate_single_task(
    env_class,
    task_id: str,
    task_index: int,
    sglang_url: str,
    sampling_params: dict,
    tokenizer,
    openai_adapter_factory,
    domain: str = "telecom",
    user_llm: str = "gemini/gemini-2.5-flash-lite",
    max_steps: int = 30,
) -> EvalResult:
    """Evaluate a single task using the gym environment."""
    try:
        env = env_class(
            domain=domain,
            task_id=task_id,
            max_steps=max_steps,
            user_llm=user_llm,
            user_llm_args={"temperature": 0.7},
        )

        observation, info = env.reset()
        tools_info = [t if isinstance(t, dict) else t.openai_schema for t in info.get("tools", [])]
        policy = info.get("policy", "")

        openai_adapter = openai_adapter_factory(tools_info=tools_info, parser_type="qwen25")

        messages = [{"role": "system", "content": policy}]
        if observation:
            lines = observation.strip().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                if ": " in line:
                    role, content = line.split(": ", 1)
                    role = role.strip().lower()
                    if role == "assistant":
                        messages.append({"role": "assistant", "content": content})
                    elif role == "user":
                        messages.append({"role": "user", "content": content})
                    else:
                        messages.append({"role": "tool", "name": role, "content": content})

        TOOL_INSTRUCTION = (
            " At each turn, you are allowed to call one or no function to assist "
            "with task execution using <tools></tools> XML tags.\n"
            "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
            "Each tool call leads to a message returned by the system.\n"
            "NEVER confirm execution to the user without seeing confirmation "
            "from the tool system.\n"
        )

        step_count = 0
        terminated = False
        total_reward = 0.0
        final_reward_info = None

        while not terminated and step_count < max_steps:
            text_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, tools=tools_info
            )
            text_input = text_input.replace(
                "You may call one or more functions to assist with the user query.",
                TOOL_INSTRUCTION,
            )

            payload = {"text": text_input, "sampling_params": sampling_params}
            output = await post_standalone(sglang_url, payload)

            if output["meta_info"]["finish_reason"]["type"] == "abort":
                return EvalResult(
                    task_id=task_id,
                    task_index=task_index,
                    reward=0.0,
                    success=False,
                    steps=step_count,
                    status="aborted",
                    error="Generation aborted",
                )

            response = output["text"]
            if response.endswith("<|im_end|>"):
                response = response[:-10]

            openai_result = openai_adapter.parse_response_to_openai_format(response)
            if not openai_result["success"]:
                return EvalResult(
                    task_id=task_id,
                    task_index=task_index,
                    reward=0.0,
                    success=False,
                    steps=step_count,
                    status="parse_error",
                    error=openai_result.get("error", "Unknown parse error"),
                )

            parsed = openai_result["parsed_result"]
            messages.append({"role": "assistant", "content": response})

            calls = parsed.get("calls", [])
            normal_text = parsed.get("normal_text", "")

            if calls:
                tool_call = calls[0]
                action = json.dumps({
                    "name": tool_call["name"],
                    "arguments": json.loads(tool_call["parameters"]),
                })
            else:
                action = normal_text

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward = reward
            final_reward_info = info.get("reward_info")
            step_count += 1

            if terminated:
                break

            if observation:
                obs_messages = [{"role": "system", "content": policy}]
                lines = observation.strip().split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    if ": " in line:
                        role, content = line.split(": ", 1)
                        role = role.strip().lower()
                        if role == "assistant":
                            obs_messages.append({"role": "assistant", "content": content})
                        elif role == "user":
                            obs_messages.append({"role": "user", "content": content})
                        else:
                            obs_messages.append({"role": "tool", "name": role, "content": content})

                for msg in obs_messages[1:]:
                    if msg not in messages:
                        messages.append(msg)

        success = total_reward > 0.5
        status = "completed" if terminated else "truncated"

        return EvalResult(
            task_id=task_id,
            task_index=task_index,
            reward=total_reward,
            success=success,
            steps=step_count,
            status=status,
            reward_info=final_reward_info,
        )

    except Exception as e:
        logger.error(f"Error evaluating task {task_id}: {e}")
        return EvalResult(
            task_id=task_id,
            task_index=task_index,
            reward=0.0,
            success=False,
            steps=0,
            status="error",
            error=str(e),
        )


async def run_evaluation(
    checkpoint_path: str,
    output_path: str,
    num_tasks: Optional[int] = None,
    sglang_port: int = 30000,
    domain: str = "telecom",
    task_split: str = "test",
):
    """Run evaluation on tau2-bench tasks."""
    from tau2.gym.gym_agent import AgentGymEnv
    from tau2.registry import registry
    from transformers import AutoTokenizer
    from openai_tool_adapter import create_openai_adapter

    logger.info(f"Loading tokenizer from checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    tasks_loader = registry.get_tasks_loader(domain)
    tasks = tasks_loader(task_split)
    task_ids = [task.id for task in tasks]

    if num_tasks:
        task_ids = task_ids[:num_tasks]

    logger.info(f"Evaluating {len(task_ids)} {task_split} tasks")

    sglang_url = f"http://127.0.0.1:{sglang_port}/generate"
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": 1024,
        "top_p": 1.0,
    }

    results: List[EvalResult] = []

    for i, task_id in enumerate(task_ids):
        logger.info(f"[{i+1}/{len(task_ids)}] Evaluating task: {task_id}")

        result = await evaluate_single_task(
            env_class=AgentGymEnv,
            task_id=task_id,
            task_index=i,
            sglang_url=sglang_url,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            openai_adapter_factory=create_openai_adapter,
            domain=domain,
        )

        results.append(result)
        logger.info(f"  Result: success={result.success}, reward={result.reward:.3f}, steps={result.steps}")

    successes = sum(1 for r in results if r.success)
    total = len(results)
    pass_at_1 = successes / total if total > 0 else 0.0

    # Compute partial scores from reward_info
    partial_scores = []
    for r in results:
        if r.reward_info:
            ri = json.loads(r.reward_info) if isinstance(r.reward_info, str) else r.reward_info
            ps = compute_partial_score(ri)
            r.partial_score = ps
            partial_scores.append(ps)
        else:
            partial_scores.append(0.0)

    avg_partial = sum(partial_scores) / len(partial_scores) if partial_scores else 0.0

    summary = {
        "checkpoint": checkpoint_path,
        "domain": domain,
        "task_split": task_split,
        "total_tasks": total,
        "successes": successes,
        "pass_at_1": pass_at_1,
        "avg_reward": sum(r.reward for r in results) / total if total > 0 else 0.0,
        "avg_steps": sum(r.steps for r in results) / total if total > 0 else 0.0,
        "avg_partial_score": avg_partial,
        "results": [
            {
                "task_id": r.task_id,
                "task_index": r.task_index,
                "reward": r.reward,
                "success": r.success,
                "steps": r.steps,
                "status": r.status,
                "error": r.error,
                "partial_score": r.partial_score,
                "reward_info": r.reward_info,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*50}")
    logger.info(f"EVALUATION SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Tasks: {total} ({task_split} split)")
    logger.info(f"Pass@1: {pass_at_1:.1%} ({successes}/{total})")
    logger.info(f"Avg Reward: {summary['avg_reward']:.3f}")
    logger.info(f"Avg Partial Score: {avg_partial:.3f}")
    logger.info(f"Avg Steps: {summary['avg_steps']:.1f}")
    logger.info(f"Results saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate tau2-bench checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="eval_results.json",
        help="Output path for evaluation results",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--sglang-port",
        type=int,
        default=30000,
        help="SGLang server port",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="telecom",
        help="tau2 domain",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        help="Task split to evaluate (train/test)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_evaluation(
            checkpoint_path=args.checkpoint,
            output_path=args.output_path,
            num_tasks=args.num_tasks,
            sglang_port=args.sglang_port,
            domain=args.domain,
            task_split=args.task_split,
        )
    )


if __name__ == "__main__":
    main()
