"""Preprocess tau2 telecom tasks to JSONL format for slime training."""

import argparse
import json
import os

from tau2.registry import registry

ALL_DATA_MAPPINGS = {"telecom": ["train", "test"]}


def main():
    parser = argparse.ArgumentParser(description="Preprocess tau2 telecom tasks")
    parser.add_argument("--local_dir", required=True, help="Output directory")
    args = parser.parse_args()

    local_dir = args.local_dir
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)

    for domain, splits in ALL_DATA_MAPPINGS.items():
        get_tasks = registry.get_tasks_loader(domain)

        for split in splits:
            tasks = get_tasks(split)
            output_path = os.path.join(local_dir, f"{domain}_{split}_tasks.jsonl")
            with open(output_path, "w") as f:
                for i, task in enumerate(tasks):
                    row = {"index": i, "metadata": task.model_dump()}
                    f.write(json.dumps(row) + "\n")
            print(f"Saved {len(tasks)} {domain} ({split}) tasks to {output_path}")


if __name__ == "__main__":
    main()
