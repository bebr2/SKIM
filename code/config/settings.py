from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

from dotenv import load_dotenv


STAGE = Literal["stage1_reconstruction", "stage2_warmup", "stage3_alignment"]
K_STRATEGY = Literal["curriculum", "random_k", "random", "multi_k_per_case"]
INTERVAL_STRATEGY = Literal["no", "steps", "epoch"]
SAVE_STRATEGY = Literal["steps", "epoch"]
COMPRESSOR_CHAT_SHELL_MODE = Literal["current", "legacy"]
LLM_TRAIN_MODE = Literal["false", "true", "lora"]
SKILL_QA_LOSS_MODE = Literal["sft", "simpo"]


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _get_list_int(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _get_list_str(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _get_train_llm_mode(name: str, default: str, allow_lora: bool = False) -> LLM_TRAIN_MODE:
    value = os.getenv(name, default)
    norm = value.strip().lower()
    if norm in {"1", "true", "yes", "on"}:
        return "true"
    if norm in {"0", "false", "no", "off"}:
        return "false"
    if norm == "lora":
        if not allow_lora:
            raise ValueError(f"{name} does not support 'lora'")
        return "lora"
    if allow_lora:
        raise ValueError(f"{name} must be one of: false, true, lora")
    raise ValueError(f"{name} must be one of: false, true")


def _get_compressor_chat_shell_mode(name: str, default: str) -> COMPRESSOR_CHAT_SHELL_MODE:
    value = os.getenv(name, default).strip().lower()
    if value not in {"current", "legacy"}:
        raise ValueError(f"{name} must be one of: current, legacy")
    return cast(COMPRESSOR_CHAT_SHELL_MODE, value)


def _get_skill_qa_loss_mode(name: str, default: str) -> SKILL_QA_LOSS_MODE:
    value = os.getenv(name, default).strip().lower()
    if value not in {"sft", "simpo"}:
        raise ValueError(f"{name} must be one of: sft, simpo")
    return cast(SKILL_QA_LOSS_MODE, value)


def _get_stage() -> STAGE:
    raw = os.getenv("SKIM_STAGE", os.getenv("STAGE", "stage1_reconstruction")).strip().lower()
    aliases = {
        "stage1": "stage1_reconstruction",
        "stage2": "stage2_warmup",
        "stage3": "stage3_alignment",
    }
    value = aliases.get(raw, raw)
    valid = {"stage1_reconstruction", "stage2_warmup", "stage3_alignment"}
    if value not in valid:
        raise ValueError(f"SKIM_STAGE must be one of: {sorted(valid)}")
    return cast(STAGE, value)


@dataclass
class Trainability:
    train_compressor: bool
    train_q: bool
    train_projector: bool
    train_llm: bool


@dataclass
class SkillQADataSource:
    path: str
    name: str
    size: int
    enabled: bool


@dataclass
class SkillQALoraSettings:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]
    bias: str
    task_type: str
    modules_to_save: list[str]
    use_rslora: bool


def _to_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "yes", "on"}:
            return True
        if norm in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot parse bool value from: {value!r}")


def _parse_skill_qa_data_sources(
    raw_config: str,
    fallback_path: str,
    split_name: str,
) -> list[SkillQADataSource]:
    config = raw_config.strip()
    if config:
        try:
            payload = json.loads(config)
        except json.JSONDecodeError as exc:
            raise ValueError(f"SKILL_QA_{split_name.upper()}_DATA_CONFIG is not valid JSON: {exc}") from exc

        if not isinstance(payload, list):
            raise ValueError(f"SKILL_QA_{split_name.upper()}_DATA_CONFIG must be a JSON array")

        sources: list[SkillQADataSource] = []
        for idx, item in enumerate(payload):
            if isinstance(item, str):
                path = item.strip()
                if path:
                    sources.append(SkillQADataSource(path=path, name=f"{split_name}_{idx}", size=0, enabled=True))
                continue

            if not isinstance(item, dict):
                raise ValueError(f"SKILL_QA_{split_name.upper()}_DATA_CONFIG[{idx}] must be string or object")

            path = str(item.get("path", "")).strip()
            if not path:
                raise ValueError(f"SKILL_QA_{split_name.upper()}_DATA_CONFIG[{idx}].path is required")

            size = int(item.get("size", 0) or 0)
            if size < 0:
                raise ValueError(f"SKILL_QA_{split_name.upper()}_DATA_CONFIG[{idx}].size must be >= 0")

            sources.append(
                SkillQADataSource(
                    path=path,
                    name=str(item.get("name") or f"{split_name}_{idx}").strip(),
                    size=size,
                    enabled=_to_bool(item.get("enabled", True), default=True),
                )
            )

        enabled_sources = [x for x in sources if x.enabled]
        if not enabled_sources:
            raise ValueError(f"SKILL_QA_{split_name.upper()}_DATA_CONFIG has no enabled source")
        return enabled_sources

    fb_path = fallback_path.strip()
    if not fb_path:
        return []
    return [SkillQADataSource(path=fb_path, name=f"{split_name}_default", size=0, enabled=True)]


@dataclass
class Settings:
    seed: int
    output_dir: str
    run_name: str
    log_wandb: bool
    wandb_project: str
    wandb_entity: str
    wandb_dir: str

    stage: STAGE

    stage1_checkpoint: str
    stage1_train_path: str
    stage1_val_path: str
    stage1_wiki_dataset: str
    stage1_wiki_size: int
    stage1_curriculum_epochs: int

    skill_qa_train_path: str
    skill_qa_val_path: str
    skill_qa_train_data_config: str
    skill_qa_val_data_config: str
    skill_qa_train_sources: list[SkillQADataSource]
    skill_qa_val_sources: list[SkillQADataSource]
    skill_qa_dpo_rejected_data_config: str
    skill_qa_dpo_rejected_sources: list[SkillQADataSource]
    skill_qa_loss: SKILL_QA_LOSS_MODE
    skill_qa_simpo_lambda: float
    skill_qa_simpo_beta: float
    skill_qa_simpo_gamma: float
    skill_qa_llm_train_mode: LLM_TRAIN_MODE
    skill_qa_lora: SkillQALoraSettings

    compressor_model: str
    llm_model: str
    projector_layers: int
    projector_hidden: int
    k_values: list[int]
    k_strategy: K_STRATEGY
    curriculum_warmup_ratio: float
    ctx_budget: int
    k_min: int
    identifier_limit: int
    compressor_chat_shell_mode: COMPRESSOR_CHAT_SHELL_MODE

    stage1_trainability: Trainability
    stage2_trainability: Trainability
    stage3_trainability: Trainability

    epochs: int
    lr: float
    lr_warmup_ratio: float
    weight_decay: float
    batch_size: int
    grad_acc: int
    max_length: int
    bf16: bool
    gradient_checkpointing: bool
    logging_steps: int
    eval_strategy: INTERVAL_STRATEGY
    eval_steps: int
    save_strategy: SAVE_STRATEGY
    save_steps: int
    save_total_limit: int

    use_deepspeed: bool
    deepspeed_config: str


def load_settings(env_file: str = ".env") -> Settings:
    load_dotenv(env_file)
    stage = _get_stage()

    default_skill_qa_llm = "lora" if stage == "stage3_alignment" else "false"
    skill_qa_llm_train_mode = _get_train_llm_mode(
        "SKILL_QA_TRAIN_LLM",
        default_skill_qa_llm,
        allow_lora=True,
    )

    skill_qa_train_path = os.getenv("SKILL_QA_TRAIN_PATH", "./data/skill_qa_train.jsonl")
    skill_qa_val_path = os.getenv("SKILL_QA_VAL_PATH", "./data/skill_qa_val.jsonl")
    skill_qa_train_data_config = os.getenv("SKILL_QA_TRAIN_DATA_CONFIG", "")
    skill_qa_val_data_config = os.getenv("SKILL_QA_VAL_DATA_CONFIG", "")
    skill_qa_dpo_rejected_data_config = os.getenv("SKILL_QA_DPO_REJECTED_DATA_CONFIG", "")
    skill_qa_train_sources = _parse_skill_qa_data_sources(skill_qa_train_data_config, skill_qa_train_path, "train")
    skill_qa_val_sources = _parse_skill_qa_data_sources(skill_qa_val_data_config, skill_qa_val_path, "val")
    skill_qa_dpo_rejected_sources = _parse_skill_qa_data_sources(
        skill_qa_dpo_rejected_data_config,
        "",
        "dpo_rejected",
    )

    return Settings(
        seed=_get_int("SEED", 42),
        output_dir=os.getenv("OUTPUT_DIR", "./outputs"),
        run_name=os.getenv("RUN_NAME", "skim"),
        log_wandb=_get_bool("LOG_WANDB", True),
        wandb_project=os.getenv("WANDB_PROJECT", "skim"),
        wandb_entity=os.getenv("WANDB_ENTITY", ""),
        wandb_dir=os.getenv("WANDB_DIR", "./wandb"),
        stage=stage,
        stage1_checkpoint=os.getenv("STAGE1_CHECKPOINT", "").strip(),
        stage1_train_path=os.getenv("STAGE1_TRAIN_PATH", "./data/stage1_train.sample.jsonl"),
        stage1_val_path=os.getenv("STAGE1_VAL_PATH", "./data/stage1_val.sample.jsonl"),
        stage1_wiki_dataset=os.getenv("STAGE1_WIKI_DATASET", "").strip(),
        stage1_wiki_size=_get_int("STAGE1_WIKI_SIZE", 0),
        stage1_curriculum_epochs=_get_int("STAGE1_CURRICULUM_EPOCHS", 0),
        skill_qa_train_path=skill_qa_train_path,
        skill_qa_val_path=skill_qa_val_path,
        skill_qa_train_data_config=skill_qa_train_data_config,
        skill_qa_val_data_config=skill_qa_val_data_config,
        skill_qa_train_sources=skill_qa_train_sources,
        skill_qa_val_sources=skill_qa_val_sources,
        skill_qa_dpo_rejected_data_config=skill_qa_dpo_rejected_data_config,
        skill_qa_dpo_rejected_sources=skill_qa_dpo_rejected_sources,
        skill_qa_loss=_get_skill_qa_loss_mode("SKILL_QA_LOSS", "sft"),
        skill_qa_simpo_lambda=_get_float("SKILL_QA_SIMPO_LAMBDA", 0.2),
        skill_qa_simpo_beta=_get_float("SKILL_QA_SIMPO_BETA", 1.0),
        skill_qa_simpo_gamma=_get_float("SKILL_QA_SIMPO_GAMMA", 0.0),
        skill_qa_llm_train_mode=skill_qa_llm_train_mode,
        skill_qa_lora=SkillQALoraSettings(
            r=_get_int("SKILL_QA_LORA_R", 16),
            alpha=_get_int("SKILL_QA_LORA_ALPHA", 32),
            dropout=_get_float("SKILL_QA_LORA_DROPOUT", 0.05),
            target_modules=_get_list_str(
                "SKILL_QA_LORA_TARGET_MODULES",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
            ),
            bias=os.getenv("SKILL_QA_LORA_BIAS", "none"),
            task_type=os.getenv("SKILL_QA_LORA_TASK_TYPE", "CAUSAL_LM"),
            modules_to_save=_get_list_str("SKILL_QA_LORA_MODULES_TO_SAVE", ""),
            use_rslora=_get_bool("SKILL_QA_LORA_USE_RSLORA", False),
        ),
        compressor_model=os.getenv("COMPRESSOR_MODEL", "Qwen/Qwen3-8B"),
        llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen3-8B"),
        projector_layers=_get_int("PROJECTOR_LAYERS", 3),
        projector_hidden=_get_int("PROJECTOR_HIDDEN", 4096),
        k_values=sorted(_get_list_int("K_VALUES", "256,512")),
        k_strategy=cast(K_STRATEGY, os.getenv("K_STRATEGY", "multi_k_per_case")),
        curriculum_warmup_ratio=_get_float("CURRICULUM_WARMUP_RATIO", 0.3),
        ctx_budget=_get_int("CTX_BUDGET", 512),
        k_min=_get_int("K_MIN", 256),
        identifier_limit=_get_int("IDENTIFIER_LIMIT", 26),
        compressor_chat_shell_mode=_get_compressor_chat_shell_mode("COMPRESSOR_CHAT_SHELL_MODE", "current"),
        stage1_trainability=Trainability(
            train_compressor=_get_bool("STAGE1_TRAIN_COMPRESSOR", True),
            train_q=_get_bool("STAGE1_TRAIN_Q", True),
            train_projector=_get_bool("STAGE1_TRAIN_PROJECTOR", True),
            train_llm=_get_bool("STAGE1_TRAIN_LLM", False),
        ),
        stage2_trainability=Trainability(
            train_compressor=_get_bool("STAGE2_TRAIN_COMPRESSOR", True),
            train_q=_get_bool("STAGE2_TRAIN_Q", True),
            train_projector=_get_bool("STAGE2_TRAIN_PROJECTOR", True),
            train_llm=_get_bool("STAGE2_TRAIN_LLM", False),
        ),
        stage3_trainability=Trainability(
            train_compressor=_get_bool("STAGE3_TRAIN_COMPRESSOR", True),
            train_q=_get_bool("STAGE3_TRAIN_Q", True),
            train_projector=_get_bool("STAGE3_TRAIN_PROJECTOR", True),
            train_llm=(skill_qa_llm_train_mode != "false"),
        ),
        epochs=_get_int("EPOCHS", 2),
        lr=_get_float("LR", 1e-5),
        lr_warmup_ratio=_get_float("LR_WARMUP_RATIO", 0.1),
        weight_decay=_get_float("WEIGHT_DECAY", 0.01),
        batch_size=_get_int("BATCH_SIZE", 1),
        grad_acc=_get_int("GRAD_ACC", 8),
        max_length=_get_int("MAX_LENGTH", 4096),
        bf16=_get_bool("BF16", True),
        gradient_checkpointing=_get_bool("GRADIENT_CHECKPOINTING", True),
        logging_steps=_get_int("LOGGING_STEPS", 10),
        eval_strategy=cast(INTERVAL_STRATEGY, os.getenv("EVAL_STRATEGY", "epoch")),
        eval_steps=_get_int("EVAL_STEPS", 100),
        save_strategy=cast(SAVE_STRATEGY, os.getenv("SAVE_STRATEGY", "epoch")),
        save_steps=_get_int("SAVE_STEPS", 100),
        save_total_limit=_get_int("SAVE_TOTAL_LIMIT", 3),
        use_deepspeed=_get_bool("USE_DEEPSPEED", True),
        deepspeed_config=os.getenv("DEEPSPEED_CONFIG", "./code/config/deepspeed_zero2.json"),
    )


def get_trainability(settings: Settings) -> Trainability:
    if settings.stage == "stage1_reconstruction":
        return settings.stage1_trainability
    if settings.stage == "stage2_warmup":
        return settings.stage2_trainability
    return settings.stage3_trainability
