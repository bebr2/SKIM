from __future__ import annotations

import os
import random

import numpy as np
import torch

from config.settings import load_settings
from engine.trainer_skill_qa import SkillQATrainer
from engine.trainer_stage1_reconstruction import Stage1ReconstructionTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_params(model: torch.nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    print(f"Trainable params: {trainable}/{total} ({pct:.2f}%)")


def main() -> None:
    settings = load_settings(env_file=os.getenv("ENV_FILE", ".env"))
    os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
    set_seed(settings.seed)

    if settings.log_wandb:
        os.environ.setdefault("WANDB_PROJECT", settings.wandb_project)
        if settings.wandb_entity:
            os.environ["WANDB_ENTITY"] = settings.wandb_entity
        os.environ.setdefault("WANDB_NAME", settings.run_name)
        os.environ.setdefault("WANDB_DIR", settings.wandb_dir)
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)

    os.makedirs(settings.output_dir, exist_ok=True)
    print(f"SKIM stage: {settings.stage}")
    print(f"k_strategy: {settings.k_strategy}; k_values: {settings.k_values}")

    if settings.stage == "stage1_reconstruction":
        trainer = Stage1ReconstructionTrainer(settings)
    elif settings.stage in {"stage2_warmup", "stage3_alignment"}:
        trainer = SkillQATrainer(settings)
    else:
        raise ValueError(f"Unsupported SKIM stage: {settings.stage}")

    print_trainable_params(trainer.model)
    trainer.run()


if __name__ == "__main__":
    main()
