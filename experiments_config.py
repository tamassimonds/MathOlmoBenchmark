#!/usr/bin/env python3
"""
Experiment configuration for OLMo benchmarking
Defines all model variants and checkpoints to be tested
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    model_name: str
    revision: Optional[str] = None
    checkpoint_url: Optional[str] = None
    batch_size: int = 8
    max_questions: int = 500
    description: str = ""


# OLMo 1B experiments
OLMO_1B_EXPERIMENTS = [
    # Base checkpoints
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step0-tokens0B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step0-unsharded/",
        description="OLMo 1B - Step 0",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step10000-tokens21B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step10000-unsharded/",
        description="OLMo 1B - Step 10k",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step100000-tokens210B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step100000-unsharded/",
        description="OLMo 1B - Step 100k",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step500000-tokens1050B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step500000-unsharded/",
        description="OLMo 1B - Step 500k",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step1000000-tokens2100B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step1000000-unsharded/",
        description="OLMo 1B - Step 1M",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step1500000-tokens3150B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step1500000-unsharded/",
        description="OLMo 1B - Step 1.5M",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage1-step1907359-tokens4005B",
        checkpoint_url="https://olmo-checkpoints.org/ai2-llm/peteish1/step1907359-unsharded/",
        description="OLMo 1B - Final Step 1.9M",
        batch_size=32
    ),
    # Stage 2
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B",
        revision="stage2-step23852-tokens50B",
        description="OLMo 1B - Stage 2 Step 23852",
        batch_size=32
    ),
    # Fine-tuned variants
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B-SFT",
        description="OLMo 1B - SFT",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B-DPO",
        description="OLMo 1B - DPO",
        batch_size=32
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-0425-1B-Instruct",
        description="OLMo 1B - Instruct",
        batch_size=32
    ),
]

# OLMo 7B experiments
OLMO_7B_EXPERIMENTS = [
    # Stage 1 checkpoints
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B",
        revision="stage1-step0-tokens0B",
        description="OLMo 7B - Step 0",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B",
        revision="stage1-step10000-tokens21B",
        description="OLMo 7B - Step 10k",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B",
        revision="stage1-step100000-tokens210B",
        description="OLMo 7B - Step 100k",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B",
        revision="stage1-step500000-tokens1050B",
        description="OLMo 7B - Step 500k",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B",
        revision="stage1-step750000-tokens1575B",
        description="OLMo 7B - Step 750k",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B",
        revision="stage1-step928000-tokens1949B",
        description="OLMo 7B - Final Step 928k",
        batch_size=16
    ),
    # Fine-tuned variants
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B-SFT",
        description="OLMo 7B - SFT",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B-DPO",
        description="OLMo 7B - DPO",
        batch_size=16
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-7B-Instruct",
        description="OLMo 7B - Instruct",
        batch_size=16
    ),
]

# OLMo 13B experiments
OLMO_13B_EXPERIMENTS = [
    # Stage 1 checkpoints
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B",
        revision="stage1-step0-tokens0B",
        description="OLMo 13B - Step 0",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B",
        revision="stage1-step10000-tokens21B",
        description="OLMo 13B - Step 10k",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B",
        revision="stage1-step100000-tokens210B",
        description="OLMo 13B - Step 100k",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B",
        revision="stage1-step300000-tokens630B",
        description="OLMo 13B - Step 300k",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B",
        revision="stage1-step450000-tokens945B",
        description="OLMo 13B - Step 450k",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B",
        revision="stage1-step596057-tokens1252B",
        description="OLMo 13B - Final Step 596k",
        batch_size=8
    ),
    # Fine-tuned variants
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B-SFT",
        description="OLMo 13B - SFT",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B-DPO",
        description="OLMo 13B - DPO",
        batch_size=8
    ),
    ExperimentConfig(
        model_name="allenai/OLMo-2-1124-13B-Instruct",
        description="OLMo 13B - Instruct",
        batch_size=8
    ),
]

# All experiments combined
ALL_EXPERIMENTS = OLMO_1B_EXPERIMENTS + OLMO_7B_EXPERIMENTS + OLMO_13B_EXPERIMENTS

# Experiment groups for easier management
EXPERIMENT_GROUPS = {
    "olmo_1b": OLMO_1B_EXPERIMENTS,
    "olmo_7b": OLMO_7B_EXPERIMENTS,
    "olmo_13b": OLMO_13B_EXPERIMENTS,
    "all": ALL_EXPERIMENTS,
    "base_models": [exp for exp in ALL_EXPERIMENTS if not any(variant in exp.model_name for variant in ["-SFT", "-DPO", "-Instruct"])],
    "sft_models": [exp for exp in ALL_EXPERIMENTS if "-SFT" in exp.model_name],
    "dpo_models": [exp for exp in ALL_EXPERIMENTS if "-DPO" in exp.model_name],
    "instruct_models": [exp for exp in ALL_EXPERIMENTS if "-Instruct" in exp.model_name],
}


def get_experiments(group_name: str = "all") -> List[ExperimentConfig]:
    """Get experiments by group name"""
    if group_name not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown experiment group: {group_name}. Available: {list(EXPERIMENT_GROUPS.keys())}")
    return EXPERIMENT_GROUPS[group_name]


def print_experiment_summary():
    """Print a summary of all experiments"""
    print("=== OLMo Experiment Configuration ===")
    print(f"Total experiments: {len(ALL_EXPERIMENTS)}")
    print()
    
    for group_name, experiments in EXPERIMENT_GROUPS.items():
        if group_name == "all":
            continue
        print(f"{group_name.upper()}: {len(experiments)} experiments")
        for exp in experiments:
            print(f"  - {exp.description}")
        print()


if __name__ == "__main__":
    print_experiment_summary()