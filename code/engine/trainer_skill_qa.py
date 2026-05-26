from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

try:
    from safetensors.torch import load_file as load_safetensors_file
except Exception:  # pragma: no cover
    load_safetensors_file = None

from config.settings import Settings, get_trainability
from data.collators import SkillQACollator
from data.skill_qa import SkillQADataset
from models.skim_model import SKIMModel
from utils.k_schedule import pick_ks_for_case
from utils.losses import sequence_log_probs_from_logits, simpo_loss
from utils.prompts import (
    build_chat_prompt_embeds_from_system_and_user_segments,
    build_chat_template_system_user_shell_ids,
)


class SkillQATrainer:
    _SKILL_TAG_PATTERN = re.compile(r"<skill>\s*(.*?)\s*</skill>", re.IGNORECASE | re.DOTALL)

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.trainability = get_trainability(self.settings)
        self.model = SKIMModel(
            compressor_model=settings.compressor_model,
            llm_model=settings.llm_model,
            max_q=max(settings.k_values),
            projector_layers=settings.projector_layers,
            projector_hidden=settings.projector_hidden,
            compressor_use_chat_shell=(settings.compressor_chat_shell_mode == "current"),
        )
        self._load_prior_stage_weights()
        self._configure_lora_if_needed()
        self._configure_trainability()

    def _parse_user_segments(self, user_text: str) -> list[tuple[str, str]]:
        segments: list[tuple[str, str]] = []
        cursor = 0

        for match in self._SKILL_TAG_PATTERN.finditer(user_text):
            if match.start() > cursor:
                segments.append(("text", user_text[cursor : match.start()]))

            skill_spec = match.group(1).strip()
            if skill_spec:
                segments.append(("skill", skill_spec))
            else:
                segments.append(("text", match.group(0)))
            cursor = match.end()

        if cursor < len(user_text):
            segments.append(("text", user_text[cursor:]))

        if not segments:
            segments.append(("text", user_text))
        return segments

    @staticmethod
    def _ensure_skill_map(raw_skill_map: Any) -> dict[str, Any]:
        if not isinstance(raw_skill_map, dict):
            return {}
        out: dict[str, Any] = {}
        for key, value in raw_skill_map.items():
            out[str(key)] = value
        return out

    @staticmethod
    def _value_to_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def _resolve_skill_text_from_sample(self, skill_spec: str, sample_meta: dict[str, Any]) -> str:
        clean_spec = skill_spec.strip()
        if not clean_spec:
            raise ValueError("Encountered empty <skill>...</skill> tag")

        skill_map = self._ensure_skill_map(sample_meta.get("skill_map"))
        if clean_spec in skill_map:
            return self._value_to_text(skill_map[clean_spec])

        if "." in clean_spec:
            skill_id, field = clean_spec.split(".", 1)
            if skill_id in skill_map:
                value = skill_map[skill_id]
                if isinstance(value, dict):
                    if field in value:
                        return self._value_to_text(value[field])
                    if "content" in value:
                        return self._value_to_text(value["content"])
                if field == "content":
                    return self._value_to_text(value)

        source_name = str(sample_meta.get("source_name", "unknown"))
        record_id = str(sample_meta.get("record_id", "unknown"))
        round_index = sample_meta.get("round_index", "-")
        available_keys = sorted(skill_map.keys())
        raise KeyError(
            (
                "Cannot resolve skill tag in sample metadata: "
                f"spec='{clean_spec}', source='{source_name}', record_id='{record_id}', "
                f"round_index='{round_index}', skill_map_keys={available_keys[:10]}"
            )
        )

    def _materialize_user_text(self, user_text: str, sample_meta: dict[str, Any]) -> str:
        chunks: list[str] = []
        for kind, value in self._parse_user_segments(user_text):
            if kind == "text":
                chunks.append(value)
                continue
            chunks.append(self._resolve_skill_text_from_sample(value, sample_meta))
        return "".join(chunks)

    def _split_multi_skill_slot(
        self,
        user_text: str,
        sample_meta: dict[str, Any],
    ) -> tuple[str, list[str], str]:
        segments = self._parse_user_segments(user_text)
        skill_positions = [idx for idx, (kind, _) in enumerate(segments) if kind == "skill"]
        skill_texts: list[str] = []

        if not skill_positions:
            source_name = str(sample_meta.get("source_name", "unknown"))
            record_id = str(sample_meta.get("record_id", "unknown"))
            round_index = sample_meta.get("round_index", "-")
            raise ValueError(
                (
                    "Each skill_qa sample must contain at least one <skill> tag, "
                    f"source='{source_name}', record_id='{record_id}', round_index='{round_index}'"
                )
            )

        for pos_idx, seg_idx in enumerate(skill_positions):
            _, skill_spec = segments[seg_idx]
            block_text = self._resolve_skill_text_from_sample(skill_spec, sample_meta)

            if pos_idx < len(skill_positions) - 1:
                next_skill_idx = skill_positions[pos_idx + 1]
                bridge_text = "".join(
                    value
                    for inner_idx, (kind, value) in enumerate(segments)
                    if kind == "text" and seg_idx < inner_idx < next_skill_idx
                )
                if not bridge_text:
                    bridge_text = "\n\n"
                block_text += bridge_text

            skill_texts.append(block_text)

        first_skill_idx = skill_positions[0]
        last_skill_idx = skill_positions[-1]
        prefix_text = "".join(
            value
            for idx, (kind, value) in enumerate(segments)
            if kind == "text" and idx < first_skill_idx
        )
        suffix_text = "".join(
            value
            for idx, (kind, value) in enumerate(segments)
            if kind == "text" and idx > last_skill_idx
        )
        return prefix_text, skill_texts, suffix_text

    def _load_prior_stage_weights(self) -> None:
        prior_names = (
            ["stage2_warmup", "stage1_reconstruction"]
            if self.settings.stage == "stage3_alignment"
            else ["stage1_reconstruction"]
        )
        checkpoint_dir = None
        checkpoint_name = ""
        for name in prior_names:
            root = os.path.join(self.settings.output_dir, name)
            checkpoint_dir = self._resolve_latest_checkpoint_dir(root)
            if checkpoint_dir is None and os.path.isdir(root):
                checkpoint_dir = root
            if checkpoint_dir is not None:
                checkpoint_name = name
                break


        is_rank0 = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_rank0 = torch.distributed.get_rank() == 0
        
        if checkpoint_dir is None:
            if is_rank0:
                print(f"[{self.settings.stage}] No prior checkpoint found, training from scratch", flush=True)
            return
        
        

        if os.path.exists(checkpoint_dir):
            modular_compressor = os.path.join(checkpoint_dir, "compressor", "model.safetensors")
            modular_qp = os.path.join(checkpoint_dir, "q_projector", "model.safetensors")
            if os.path.exists(modular_compressor) and os.path.exists(modular_qp):
                if is_rank0:
                    print(f"[{self.settings.stage}] Loading {checkpoint_name} checkpoint from {checkpoint_dir}", flush=True)
                self.model.load_modular_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    load_llm=self.trainability.train_llm,
                    strict=False,
                )
                return

            bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            safe_path = os.path.join(checkpoint_dir, "model.safetensors")

            if os.path.exists(bin_path):
                if is_rank0:
                    print(f"[{self.settings.stage}] Loading {checkpoint_name} bin checkpoint from {checkpoint_dir}", flush=True)
                state = torch.load(bin_path, map_location="cpu")
                self.model.load_state_dict(state, strict=False)
                return

            if os.path.exists(safe_path) and load_safetensors_file is not None:
                if is_rank0:
                    print(f"[{self.settings.stage}] Loading {checkpoint_name} safetensors checkpoint from {checkpoint_dir}", flush=True)
                state = load_safetensors_file(safe_path)
                self.model.load_state_dict(state, strict=False)

    @staticmethod
    def _resolve_latest_checkpoint_dir(root_dir: str) -> str | None:
        if not os.path.isdir(root_dir):
            return None

        candidates: list[tuple[int, str]] = []
        for name in os.listdir(root_dir):
            full = os.path.join(root_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith("checkpoint-"):
                continue
            try:
                step = int(name.split("-")[-1])
            except ValueError:
                continue
            candidates.append((step, full))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    def _configure_trainability(self) -> None:
        t = self.trainability
        if self.settings.skill_qa_llm_train_mode == "lora":
            self.model.set_llm_lora_trainable()
        else:
            self.model.freeze_llm(not t.train_llm)
        self.model.freeze_compressor(not t.train_compressor)
        self.model.freeze_q(not t.train_q)
        self.model.freeze_projector(not t.train_projector)

        if self.settings.gradient_checkpointing:
            # Prevent repetitive runtime warnings and keep behavior explicit.
            if hasattr(self.model.llm, "config"):
                self.model.llm.config.use_cache = False
            generation_config = getattr(self.model.llm, "generation_config", None)
            if generation_config is not None:
                generation_config.use_cache = False

    def _configure_lora_if_needed(self) -> None:
        if self.settings.skill_qa_llm_train_mode != "lora":
            return

        lora = self.settings.skill_qa_lora
        self.model.enable_llm_lora(
            r=lora.r,
            alpha=lora.alpha,
            dropout=lora.dropout,
            target_modules=lora.target_modules,
            bias=lora.bias,
            task_type=lora.task_type,
            modules_to_save=lora.modules_to_save,
            use_rslora=lora.use_rslora,
        )

        is_rank0 = True
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_rank0 = torch.distributed.get_rank() == 0
        if is_rank0:
            print(
                (
                    "[skill_qa] LoRA enabled for LLM: "
                    f"r={lora.r}, alpha={lora.alpha}, dropout={lora.dropout}, "
                    f"target_modules={lora.target_modules}, bias={lora.bias}, "
                    f"task_type={lora.task_type}, use_rslora={lora.use_rslora}"
                ),
                flush=True,
            )

    @staticmethod
    def _p75(values: list[int]) -> int:
        if not values:
            return 0
        arr = sorted(values)
        idx = max(0, min(len(arr) - 1, math.ceil(0.75 * len(arr)) - 1))
        return int(arr[idx])

    def _format_token_stats(self, token_lens: list[int]) -> tuple[int, float, int, int]:
        token_sum = int(sum(token_lens))
        n = len(token_lens)
        token_avg = (token_sum / n) if n > 0 else 0.0
        token_max = max(token_lens) if token_lens else 0
        token_p75 = self._p75(token_lens)
        return token_sum, token_avg, token_max, token_p75

    def _log_dataset_token_stats(self, dataset: SkillQADataset, split: str) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return

        tokenizer = self.model.compressor.tokenizer

        input_lens: list[int] = []
        output_lens: list[int] = []
        per_source_input: dict[str, list[int]] = {}
        per_source_output: dict[str, list[int]] = {}

        for sample in dataset.samples:
            sample_meta = sample.metadata if isinstance(sample.metadata, dict) else {}
            source_name = str(sample_meta.get("source_name", "unknown"))
            user_text = self._materialize_user_text(sample.user_content, sample_meta)
            input_text = (
                f"{sample.system_content}\n\n{user_text}"
                if sample.system_content.strip()
                else user_text
            )
            output_text = sample.assistant_content

            in_len = len(tokenizer.encode(input_text, add_special_tokens=False))
            out_len = len(tokenizer.encode(output_text, add_special_tokens=False))
            input_lens.append(in_len)
            output_lens.append(out_len)
            per_source_input.setdefault(source_name, []).append(in_len)
            per_source_output.setdefault(source_name, []).append(out_len)

        for source_stat in dataset.source_stats:
            print(
                (
                    f"[skill_qa][{split}][source_data] name={source_stat.name}; "
                    f"requested_size={source_stat.requested_size}; "
                    f"loaded={source_stat.loaded_samples}; selected={source_stat.selected_samples}; "
                    f"path={source_stat.path}"
                ),
                flush=True,
            )

        n = len(dataset)
        in_sum, in_avg, in_max, in_p75 = self._format_token_stats(input_lens)
        out_sum, out_avg, out_max, out_p75 = self._format_token_stats(output_lens)

        print(
            f"[skill_qa][{split}] samples={n}; "
            f"input_tokens(sum/avg/max/p75)={in_sum}/{in_avg:.2f}/{in_max}/{in_p75}; "
            f"output_tokens(sum/avg/max/p75)={out_sum}/{out_avg:.2f}/{out_max}/{out_p75}",
            flush=True,
        )

        for source_name in sorted(per_source_input.keys()):
            src_in_sum, src_in_avg, src_in_max, src_in_p75 = self._format_token_stats(per_source_input[source_name])
            src_out_sum, src_out_avg, src_out_max, src_out_p75 = self._format_token_stats(
                per_source_output.get(source_name, [])
            )
            print(
                (
                    f"[skill_qa][{split}][source={source_name}] samples={len(per_source_input[source_name])}; "
                    f"input_tokens(sum/avg/max/p75)={src_in_sum}/{src_in_avg:.2f}/{src_in_max}/{src_in_p75}; "
                    f"output_tokens(sum/avg/max/p75)={src_out_sum}/{src_out_avg:.2f}/{src_out_max}/{src_out_p75}"
                ),
                flush=True,
            )

        paired_count = sum(1 for sample in dataset.samples if bool(sample.has_rejected_pair))
        if split == "train" or paired_count > 0:
            paired_ratio = paired_count / max(1, len(dataset.samples))
            print(
                f"[skill_qa][{split}] paired_rejected={paired_count}/{len(dataset.samples)} ({paired_ratio:.4f})",
                flush=True,
            )

    def build_dataloader(self, train: bool) -> DataLoader:
        sources = self.settings.skill_qa_train_sources if train else self.settings.skill_qa_val_sources
        fallback_path = self.settings.skill_qa_train_path if train else self.settings.skill_qa_val_path
        rejected_sources = self.settings.skill_qa_dpo_rejected_sources if train else None
        dataset = SkillQADataset(
            path_or_sources=sources if sources else fallback_path,
            seed=self.settings.seed + (0 if train else 10_000),
            rejected_path_or_sources=rejected_sources,
        )
        split = "train" if train else "val"
        self._log_dataset_token_stats(dataset, split=split)
        collator = SkillQACollator()
        return DataLoader(
            dataset,
            batch_size=self.settings.batch_size,
            shuffle=train,
            collate_fn=collator,
        )

    def run(self) -> None:
        train_loader = self.build_dataloader(train=True)
        eval_loader = self.build_dataloader(train=False)
        # Token statistics run before Trainer builds its dataloader, so keep worker=0.
        worker_count = 0

        args = TrainingArguments(
            output_dir=os.path.join(self.settings.output_dir, self.settings.stage),
            num_train_epochs=self.settings.epochs,
            learning_rate=self.settings.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=self.settings.lr_warmup_ratio,
            weight_decay=self.settings.weight_decay,
            per_device_train_batch_size=self.settings.batch_size,
            per_device_eval_batch_size=self.settings.batch_size,
            gradient_accumulation_steps=self.settings.grad_acc,
            bf16=self.settings.bf16,
            gradient_checkpointing=self.settings.gradient_checkpointing,
            run_name=self.settings.run_name,
            logging_strategy="steps",
            logging_steps=self.settings.logging_steps,
            eval_strategy=self.settings.eval_strategy,
            eval_steps=self.settings.eval_steps if self.settings.eval_strategy == "steps" else None,
            save_strategy="epoch",
            save_safetensors=True,
            save_only_model=True,
            report_to=["wandb"] if self.settings.log_wandb else [],
            deepspeed=self.settings.deepspeed_config if self.settings.use_deepspeed else None,
            dataloader_num_workers=worker_count,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=(worker_count > 0),
        )

        model = self.model
        trainer_owner = self

        class _WrappedTrainer(Trainer):
            def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
                del _internal_call
                target_dir = output_dir or self.args.output_dir

                if self.is_deepspeed_enabled:
                    state_dict = self.accelerator.get_state_dict(self.deepspeed)
                else:
                    state_dict = self.accelerator.get_state_dict(self.model)

                if not self.args.should_save:
                    return

                real_model = self.model.module if hasattr(self.model, "module") else self.model
                real_model.save_modular_checkpoint(
                    checkpoint_dir=target_dir,
                    include_llm=bool(self.args.train_llm),
                    state_dict=state_dict,
                    metadata={
                        "compressor_model": self.args.compressor_model,
                        "llm_model": self.args.llm_model,
                        "projector_layers": self.args.projector_layers,
                        "projector_hidden": self.args.projector_hidden,
                        "k_values": self.args.k_values,
                        "identifier_limit": self.args.identifier_limit,
                        "llm_train_mode": self.args.llm_train_mode,
                        "skill_qa_lora": self.args.skill_qa_lora,
                        "skill_qa_loss": self.args.skill_qa_loss,
                        "skill_qa_simpo_lambda": self.args.skill_qa_simpo_lambda,
                        "skill_qa_simpo_beta": self.args.skill_qa_simpo_beta,
                        "skill_qa_simpo_gamma": self.args.skill_qa_simpo_gamma,
                    },
                )

            def prediction_step(
                self,
                model,
                inputs: dict[str, Any],
                prediction_loss_only: bool,
                ignore_keys: list[str] | None = None,
            ):
                del prediction_loss_only, ignore_keys
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        loss, _ = self.compute_loss(model, inputs, return_outputs=True)

                    if loss is not None:
                        loss = loss.mean().detach()

                return (loss, None, None)

            def compute_loss(self, model, inputs: dict[str, Any], return_outputs: bool = False, **kwargs):
                del kwargs
                real_model = model.module if hasattr(model, "module") else model
                llm_device = real_model.llm.device
                llm_tokenizer = real_model.llm_tokenizer
                emb_layer = real_model.llm.get_input_embeddings()

                is_rank0 = True
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    is_rank0 = torch.distributed.get_rank() == 0

                debug_step0 = int(self.state.global_step) == 0 and is_rank0
                t0 = time.perf_counter()
                if debug_step0 and not hasattr(self, "_skill_qa_debug_entered"):
                    print("[skill_qa][debug] entered compute_loss(step0)", flush=True)
                    self._skill_qa_debug_entered = True

                system_contents = inputs.get("system_contents", [""] * len(inputs["user_contents"]))
                user_contents = inputs["user_contents"]
                assistant_contents = inputs["assistant_contents"]
                rejected_raw = inputs.get("rejected_assistant_contents", [""] * len(user_contents))
                if isinstance(rejected_raw, list):
                    rejected_assistant_contents = [str(x) for x in rejected_raw]
                else:
                    rejected_assistant_contents = [""] * len(user_contents)
                if len(rejected_assistant_contents) < len(user_contents):
                    rejected_assistant_contents.extend([""] * (len(user_contents) - len(rejected_assistant_contents)))

                has_rejected_raw = inputs.get("has_rejected_pair", [False] * len(user_contents))
                has_rejected_pair: list[bool] = []
                if isinstance(has_rejected_raw, list):
                    for idx in range(len(user_contents)):
                        has_rejected_pair.append(bool(has_rejected_raw[idx]) if idx < len(has_rejected_raw) else False)
                else:
                    has_rejected_pair = [False] * len(user_contents)
                if len(has_rejected_pair) < len(user_contents):
                    has_rejected_pair.extend([False] * (len(user_contents) - len(has_rejected_pair)))

                skill_qa_loss_mode = str(getattr(self.args, "skill_qa_loss", "sft")).strip().lower()
                simpo_lambda = float(getattr(self.args, "skill_qa_simpo_lambda", 0.0))
                simpo_beta = float(getattr(self.args, "skill_qa_simpo_beta", 1.0))
                simpo_gamma = float(getattr(self.args, "skill_qa_simpo_gamma", 0.0))
                paired_count = int(sum(1 for x in has_rejected_pair if x))
                paired_ratio = float(paired_count / max(1, len(user_contents)))
                enable_simpo = (
                    skill_qa_loss_mode == "simpo"
                    and simpo_lambda > 0.0
                    and bool(real_model.training)
                )

                metadata_raw = inputs.get("metadata", [{} for _ in user_contents])
                metadata_list: list[dict[str, Any]] = []
                if isinstance(metadata_raw, list):
                    for item in metadata_raw:
                        metadata_list.append(item if isinstance(item, dict) else {})
                else:
                    metadata_list = [{} for _ in user_contents]

                step = int(self.state.global_step)
                total_steps = int(self.state.max_steps if self.state.max_steps > 0 else 1000)
                selected_ks = pick_ks_for_case(
                    k_values=self.args.k_values,
                    strategy=self.args.k_strategy,
                    step=step,
                    total_steps=total_steps,
                    warmup_ratio=self.args.curriculum_warmup_ratio,
                )
                selected_ks = sorted({int(k) for k in selected_ks})
                if not selected_ks:
                    raise ValueError("selected_ks is empty")

                # DeepSpeed ZeRO-3 can fail with duplicate ds_id assertions when trainable
                # LLM parameters are used by multiple forward graphs in one step (multi-k).
                # Keep one-k-per-step in this mode for stability.
                if self.is_deepspeed_enabled and bool(self.args.train_llm) and len(selected_ks) > 1:
                    if not hasattr(self, "_skill_qa_warned_single_k_llm_train"):
                        is_rank0_warn = True
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            is_rank0_warn = torch.distributed.get_rank() == 0
                        if is_rank0_warn:
                            print(
                                (
                                    "[skill_qa][warn] train_llm=True with DeepSpeed detected; "
                                    "forcing single-k per step to avoid ZeRO-3 duplicate-ds_id assertion. "
                                    f"selected_ks(before)={selected_ks}"
                                ),
                                flush=True,
                            )
                        self._skill_qa_warned_single_k_llm_train = True
                    selected_ks = [selected_ks[-1]]

                if not hasattr(self, "_skill_qa_system_user_shell_ids"):
                    self._skill_qa_system_user_shell_ids = build_chat_template_system_user_shell_ids(llm_tokenizer)
                system_user_shell_ids = self._skill_qa_system_user_shell_ids

                sample_specs: list[tuple[str, str, list[int], str]] = []
                skill_text_to_index: dict[str, int] = {}
                skill_texts: list[str] = []

                for i, user_text in enumerate(user_contents):
                    sample_meta = metadata_list[i] if i < len(metadata_list) else {}
                    prefix_text, sample_skill_texts, suffix_text = trainer_owner._split_multi_skill_slot(
                        str(user_text),
                        sample_meta,
                    )
                    sample_skill_indices: list[int] = []
                    for skill_text in sample_skill_texts:
                        if skill_text not in skill_text_to_index:
                            skill_text_to_index[skill_text] = len(skill_texts)
                            skill_texts.append(skill_text)
                        sample_skill_indices.append(skill_text_to_index[skill_text])

                    system_text = str(system_contents[i]) if i < len(system_contents) else ""
                    sample_specs.append(
                        (
                            system_text,
                            prefix_text,
                            sample_skill_indices,
                            suffix_text,
                        )
                    )

                if debug_step0 and not hasattr(self, "_skill_qa_debug_after_parse"):
                    dt = time.perf_counter() - t0
                    print(
                        f"[skill_qa][debug] parsed batch(step0): bsz={len(user_contents)}, "
                        f"unique_skills={len(skill_texts)}, elapsed={dt:.2f}s",
                        flush=True,
                    )
                    self._skill_qa_debug_after_parse = True

                max_k = selected_ks[-1]
                skill_soft_all: torch.Tensor | None = None
                if skill_texts:
                    skill_enc = real_model.compressor.tokenizer(
                        skill_texts,
                        truncation=True,
                        max_length=int(self.args.max_length),
                        padding=True,
                        return_tensors="pt",
                    ).to(llm_device)
                    skill_z = real_model.compressor(
                        skill_enc["input_ids"],
                        skill_enc["attention_mask"],
                        k=max_k,
                    )
                    skill_soft_all = real_model.projector(skill_z)

                if debug_step0 and not hasattr(self, "_skill_qa_debug_after_compressor"):
                    dt = time.perf_counter() - t0
                    print(
                        f"[skill_qa][debug] compressor/projector done(step0): "
                        f"max_k={max_k}, elapsed={dt:.2f}s",
                        flush=True,
                    )
                    self._skill_qa_debug_after_compressor = True

                eos_text = llm_tokenizer.eos_token or ""
                target_texts = [text + eos_text for text in assistant_contents]
                target_enc = llm_tokenizer(
                    target_texts,
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    max_length=int(self.args.max_length),
                    return_tensors="pt",
                ).to(llm_device)
                target_embeds = emb_layer(target_enc["input_ids"])
                target_labels = target_enc["input_ids"].clone()
                if llm_tokenizer.pad_token_id is not None:
                    target_labels[target_labels == llm_tokenizer.pad_token_id] = -100

                rejected_target_enc = None
                rejected_target_embeds = None
                rejected_target_labels = None
                if enable_simpo:
                    rejected_target_texts = [text + eos_text for text in rejected_assistant_contents]
                    rejected_target_enc = llm_tokenizer(
                        rejected_target_texts,
                        add_special_tokens=False,
                        padding=True,
                        truncation=True,
                        max_length=int(self.args.max_length),
                        return_tensors="pt",
                    ).to(llm_device)
                    rejected_target_embeds = emb_layer(rejected_target_enc["input_ids"])
                    rejected_target_labels = rejected_target_enc["input_ids"].clone()
                    if llm_tokenizer.pad_token_id is not None:
                        rejected_target_labels[rejected_target_labels == llm_tokenizer.pad_token_id] = -100

                llm_max_positions = int(
                    getattr(real_model.llm.config, "max_position_embeddings", int(self.args.max_length))
                )
                # Use configured MAX_LENGTH as the hard runtime budget to avoid first-step stragglers.
                max_total_positions = min(int(self.args.max_length), llm_max_positions)
                # Reserve at least one position for assistant target token.
                max_context_len = max(1, max_total_positions - 1)

                losses: list[torch.Tensor] = []
                sft_components: list[torch.Tensor] = []
                simpo_components: list[torch.Tensor] = []
                last_output = None
                for k in selected_ks:
                    per_sample_embeds: list[torch.Tensor] = []
                    per_sample_masks: list[torch.Tensor] = []
                    per_sample_labels: list[torch.Tensor] = []
                    per_sample_rejected_embeds: list[torch.Tensor] = []
                    per_sample_rejected_masks: list[torch.Tensor] = []
                    per_sample_rejected_labels: list[torch.Tensor] = []

                    for i, (system_text, prefix_text, skill_indices, suffix_text) in enumerate(sample_specs):
                        if skill_soft_all is None:
                            raise ValueError("skill_soft_all is unexpectedly None")
                        if not skill_indices:
                            raise ValueError("Encountered empty skill index list in skill_qa sample")

                        # Keep embedding call pattern stable across ranks under ZeRO-3.
                        system_for_template = system_text if system_text.strip() else "<empty_system>"
                        prefix_segment = prefix_text if prefix_text else " "
                        suffix_segment = suffix_text if suffix_text else " "
                        soft_segments = [skill_soft_all[idx, :k, :] for idx in skill_indices]
                        merged_skill_soft = soft_segments[0] if len(soft_segments) == 1 else torch.cat(soft_segments, dim=0)
                        user_segments: list[str | torch.Tensor] = [
                            prefix_segment,
                            merged_skill_soft,
                            suffix_segment,
                        ]

                        context_emb = build_chat_prompt_embeds_from_system_and_user_segments(
                            llm=real_model.llm,
                            tokenizer=llm_tokenizer,
                            system_text=system_for_template,
                            user_segments=user_segments,
                            template_shell_ids=system_user_shell_ids,
                            force_system_template=True,
                        )

                        if context_emb.size(1) > max_context_len:
                            context_emb = context_emb[:, :max_context_len, :]

                        target_emb_i = target_embeds[i : i + 1]
                        target_mask_i = target_enc["attention_mask"][i : i + 1]
                        target_labels_i = target_labels[i : i + 1]
                        # Keep per-sample sequence length bounded for ZeRO-3 stability.
                        target_len_allowed = max(1, max_total_positions - int(context_emb.size(1)))
                        target_emb_i = target_emb_i[:, :target_len_allowed, :]
                        target_mask_i = target_mask_i[:, :target_len_allowed]
                        target_labels_i = target_labels_i[:, :target_len_allowed]

                        input_embeds = torch.cat([context_emb, target_emb_i], dim=1)
                        context_mask = torch.ones(
                            (1, context_emb.size(1)),
                            device=llm_device,
                            dtype=torch.long,
                        )
                        attention_mask = torch.cat([context_mask, target_mask_i], dim=1)

                        ignore = torch.full(
                            (1, context_emb.size(1)),
                            -100,
                            device=llm_device,
                            dtype=torch.long,
                        )
                        labels = torch.cat([ignore, target_labels_i], dim=1)

                        per_sample_embeds.append(input_embeds.squeeze(0))
                        per_sample_masks.append(attention_mask.squeeze(0))
                        per_sample_labels.append(labels.squeeze(0))

                        if enable_simpo:
                            if (
                                rejected_target_embeds is None
                                or rejected_target_labels is None
                                or rejected_target_enc is None
                            ):
                                raise ValueError("rejected target buffers are unexpectedly None in SimPO mode")

                            rejected_emb_i = rejected_target_embeds[i : i + 1]
                            rejected_mask_i = rejected_target_enc["attention_mask"][i : i + 1]
                            rejected_labels_i = rejected_target_labels[i : i + 1]
                            rejected_len_allowed = max(1, max_total_positions - int(context_emb.size(1)))

                            rejected_emb_i = rejected_emb_i[:, :rejected_len_allowed, :]
                            rejected_mask_i = rejected_mask_i[:, :rejected_len_allowed]
                            rejected_labels_i = rejected_labels_i[:, :rejected_len_allowed]

                            rejected_input_embeds = torch.cat([context_emb, rejected_emb_i], dim=1)
                            rejected_attention_mask = torch.cat([context_mask, rejected_mask_i], dim=1)
                            rejected_full_labels = torch.cat([ignore, rejected_labels_i], dim=1)

                            per_sample_rejected_embeds.append(rejected_input_embeds.squeeze(0))
                            per_sample_rejected_masks.append(rejected_attention_mask.squeeze(0))
                            per_sample_rejected_labels.append(rejected_full_labels.squeeze(0))

                    if not per_sample_embeds:
                        losses.append(torch.zeros((), device=llm_device))
                        sft_components.append(torch.zeros((), device=llm_device))
                        simpo_components.append(torch.zeros((), device=llm_device))
                        continue

                    max_seq_len = max(x.size(0) for x in per_sample_embeds)
                    bsz = len(per_sample_embeds)
                    hidden_dim = per_sample_embeds[0].size(1)
                    dtype = per_sample_embeds[0].dtype

                    if (
                        int(self.state.global_step) == 0
                        and not hasattr(self, "_skill_qa_first_step_logged")
                    ):
                        is_rank0 = True
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            is_rank0 = torch.distributed.get_rank() == 0
                        if is_rank0:
                            print(
                                f"[skill_qa][step0] k={k}, bsz={bsz}, "
                                f"max_context_len={max_context_len}, max_seq_len={max_seq_len}, "
                                f"max_total_positions={max_total_positions}, llm_max_positions={llm_max_positions}"
                            )
                        self._skill_qa_first_step_logged = True

                    if debug_step0 and not hasattr(self, "_skill_qa_debug_before_first_k_forward"):
                        print(
                            f"[skill_qa][debug] before first k forward(step0): "
                            f"k={k}, max_seq_len={max_seq_len}, max_total_positions={max_total_positions}",
                            flush=True,
                        )
                        self._skill_qa_debug_before_first_k_forward = True

                    batch_input_embeds = torch.zeros(
                        (bsz, max_seq_len, hidden_dim),
                        device=llm_device,
                        dtype=dtype,
                    )
                    batch_attention_mask = torch.zeros(
                        (bsz, max_seq_len),
                        device=llm_device,
                        dtype=torch.long,
                    )
                    batch_labels = torch.full(
                        (bsz, max_seq_len),
                        -100,
                        device=llm_device,
                        dtype=torch.long,
                    )

                    for i in range(bsz):
                        seq_len = per_sample_embeds[i].size(0)
                        batch_input_embeds[i, :seq_len, :] = per_sample_embeds[i]
                        batch_attention_mask[i, :seq_len] = per_sample_masks[i]
                        batch_labels[i, :seq_len] = per_sample_labels[i]

                    out = None
                    l_sft_k = torch.zeros((), device=llm_device)
                    l_simpo_k = torch.zeros((), device=llm_device)

                    if enable_simpo:
                        max_rejected_seq_len = max(x.size(0) for x in per_sample_rejected_embeds)
                        batch_rejected_input_embeds = torch.zeros(
                            (bsz, max_rejected_seq_len, hidden_dim),
                            device=llm_device,
                            dtype=dtype,
                        )
                        batch_rejected_attention_mask = torch.zeros(
                            (bsz, max_rejected_seq_len),
                            device=llm_device,
                            dtype=torch.long,
                        )
                        batch_rejected_labels = torch.full(
                            (bsz, max_rejected_seq_len),
                            -100,
                            device=llm_device,
                            dtype=torch.long,
                        )

                        for i in range(bsz):
                            seq_len = per_sample_rejected_embeds[i].size(0)
                            batch_rejected_input_embeds[i, :seq_len, :] = per_sample_rejected_embeds[i]
                            batch_rejected_attention_mask[i, :seq_len] = per_sample_rejected_masks[i]
                            batch_rejected_labels[i, :seq_len] = per_sample_rejected_labels[i]

                        combined_seq_len = max(max_seq_len, max_rejected_seq_len)
                        combined_input_embeds = torch.zeros(
                            (bsz * 2, combined_seq_len, hidden_dim),
                            device=llm_device,
                            dtype=dtype,
                        )
                        combined_attention_mask = torch.zeros(
                            (bsz * 2, combined_seq_len),
                            device=llm_device,
                            dtype=torch.long,
                        )
                        combined_labels = torch.full(
                            (bsz * 2, combined_seq_len),
                            -100,
                            device=llm_device,
                            dtype=torch.long,
                        )

                        combined_input_embeds[:bsz, :max_seq_len, :] = batch_input_embeds
                        combined_attention_mask[:bsz, :max_seq_len] = batch_attention_mask
                        combined_labels[:bsz, :max_seq_len] = batch_labels

                        combined_input_embeds[bsz:, :max_rejected_seq_len, :] = batch_rejected_input_embeds
                        combined_attention_mask[bsz:, :max_rejected_seq_len] = batch_rejected_attention_mask
                        combined_labels[bsz:, :max_rejected_seq_len] = batch_rejected_labels

                        combined_out = real_model.llm_forward_with_embeds(
                            input_embeds=combined_input_embeds,
                            attention_mask=combined_attention_mask,
                            labels=combined_labels,
                        )

                        chosen_logits = combined_out.logits[:bsz]
                        chosen_labels = combined_labels[:bsz]
                        rejected_logits = combined_out.logits[bsz:]
                        rejected_labels = combined_labels[bsz:]

                        chosen_logp_sum, chosen_token_count = sequence_log_probs_from_logits(
                            logits=chosen_logits,
                            labels=chosen_labels,
                        )
                        rejected_logp_sum, rejected_token_count = sequence_log_probs_from_logits(
                            logits=rejected_logits,
                            labels=rejected_labels,
                        )
                        chosen_token_denom = chosen_token_count.to(dtype=chosen_logp_sum.dtype).sum().clamp_min(1.0)
                        l_sft_k = -(chosen_logp_sum.sum() / chosen_token_denom)

                        pair_mask_tensor = torch.tensor(has_rejected_pair, device=llm_device, dtype=torch.bool)
                        valid_pair_mask = (
                            pair_mask_tensor
                            & (chosen_token_count > 0)
                            & (rejected_token_count > 0)
                        )
                        l_simpo_k, _, _ = simpo_loss(
                            chosen_logp_sum=chosen_logp_sum,
                            rejected_logp_sum=rejected_logp_sum,
                            chosen_token_count=chosen_token_count,
                            rejected_token_count=rejected_token_count,
                            beta=simpo_beta,
                            gamma=simpo_gamma,
                            pair_mask=valid_pair_mask,
                        )
                        out = combined_out
                    else:
                        out = real_model.llm_forward_with_embeds(
                            input_embeds=batch_input_embeds,
                            attention_mask=batch_attention_mask,
                            labels=batch_labels,
                        )
                        l_sft_k = out.loss

                    if debug_step0 and not hasattr(self, "_skill_qa_debug_first_k_done"):
                        dt = time.perf_counter() - t0
                        print(
                            f"[skill_qa][debug] first k forward done(step0): "
                            f"k={k}, max_seq_len={max_seq_len}, elapsed={dt:.2f}s",
                            flush=True,
                        )
                        self._skill_qa_debug_first_k_done = True

                    loss_k = l_sft_k + simpo_lambda * l_simpo_k if enable_simpo else l_sft_k
                    losses.append(loss_k)
                    sft_components.append(l_sft_k.detach())
                    simpo_components.append(l_simpo_k.detach())
                    last_output = out

                loss = torch.stack(losses).mean()
                l_sft = torch.stack(sft_components).mean() if sft_components else torch.zeros((), device=llm_device)
                l_simpo = torch.stack(simpo_components).mean() if simpo_components else torch.zeros((), device=llm_device)
                out_dict = {
                    "loss": loss,
                    "l_sft": l_sft.detach(),
                    "l_simpo": l_simpo.detach(),
                    "paired_ratio": torch.tensor(paired_ratio, device=llm_device, dtype=loss.dtype).detach(),
                }
                return (loss, out_dict) if return_outputs else loss

        args.k_values = self.settings.k_values  # type: ignore[attr-defined]
        args.k_strategy = self.settings.k_strategy  # type: ignore[attr-defined]
        args.curriculum_warmup_ratio = self.settings.curriculum_warmup_ratio  # type: ignore[attr-defined]
        args.train_llm = self.trainability.train_llm  # type: ignore[attr-defined]
        args.compressor_model = self.settings.compressor_model  # type: ignore[attr-defined]
        args.llm_model = self.settings.llm_model  # type: ignore[attr-defined]
        args.projector_layers = self.settings.projector_layers  # type: ignore[attr-defined]
        args.projector_hidden = self.settings.projector_hidden  # type: ignore[attr-defined]
        args.identifier_limit = self.settings.identifier_limit  # type: ignore[attr-defined]
        args.max_length = self.settings.max_length  # type: ignore[attr-defined]
        args.llm_train_mode = self.settings.skill_qa_llm_train_mode  # type: ignore[attr-defined]
        args.skill_qa_loss = self.settings.skill_qa_loss  # type: ignore[attr-defined]
        args.skill_qa_simpo_lambda = self.settings.skill_qa_simpo_lambda  # type: ignore[attr-defined]
        args.skill_qa_simpo_beta = self.settings.skill_qa_simpo_beta  # type: ignore[attr-defined]
        args.skill_qa_simpo_gamma = self.settings.skill_qa_simpo_gamma  # type: ignore[attr-defined]
        args.skill_qa_lora = {
            "r": self.settings.skill_qa_lora.r,
            "alpha": self.settings.skill_qa_lora.alpha,
            "dropout": self.settings.skill_qa_lora.dropout,
            "target_modules": self.settings.skill_qa_lora.target_modules,
            "bias": self.settings.skill_qa_lora.bias,
            "task_type": self.settings.skill_qa_lora.task_type,
            "modules_to_save": self.settings.skill_qa_lora.modules_to_save,
            "use_rslora": self.settings.skill_qa_lora.use_rslora,
        }  # type: ignore[attr-defined]

        trainer = _WrappedTrainer(
            model=model,
            args=args,
            train_dataset=train_loader.dataset,
            eval_dataset=eval_loader.dataset if self.settings.eval_strategy != "no" else None,
            data_collator=train_loader.collate_fn,
        )
        trainer.train()
