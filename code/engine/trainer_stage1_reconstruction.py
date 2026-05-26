from __future__ import annotations

import os
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from utils import losses
from config.settings import Settings, get_trainability
from data.collators import ReconstructionCollator
from data.reconstruction import ReconstructionDataset, ReconstructionSample
from models.skim_model import SKIMModel
from utils.k_schedule import pick_ks_for_case
from utils.prompts import (
    build_chat_template_user_shell_ids,
    stage1_prompt,
)


class _Stage1EpochSwitchDataset(Dataset):
    def __init__(
        self,
        base_samples: list[ReconstructionSample],
        wiki_epoch_samples: list[list[ReconstructionSample]],
        wiki_token_caps: list[int],
    ) -> None:
        self._base_samples = list(base_samples)
        self._wiki_epoch_samples = [list(x) for x in wiki_epoch_samples]
        self._wiki_token_caps = list(wiki_token_caps)
        self._active_samples: list[ReconstructionSample] = self._base_samples
        self._active_desc = "base"

    def set_epoch(self, epoch_idx: int) -> None:
        if 0 <= epoch_idx < len(self._wiki_epoch_samples):
            cap = self._wiki_token_caps[epoch_idx]
            self._active_samples = self._wiki_epoch_samples[epoch_idx]
            self._active_desc = f"wiki(token_count<={cap})"
        else:
            self._active_samples = self._base_samples
            self._active_desc = "base"

        print(
            f"[stage1] epoch={epoch_idx + 1}: train_source={self._active_desc}, samples={len(self._active_samples)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self._active_samples)

    def __getitem__(self, idx: int) -> ReconstructionSample:
        return self._active_samples[idx]


class _Stage1CurriculumCallback(TrainerCallback):
    def __init__(self, dataset: _Stage1EpochSwitchDataset) -> None:
        self._dataset = dataset

    def on_train_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        self._dataset.set_epoch(0)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        del args, kwargs
        epoch_idx = int(state.epoch) if state.epoch is not None else 0
        self._dataset.set_epoch(epoch_idx)
        return control


class Stage1ReconstructionTrainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = SKIMModel(
            compressor_model=settings.compressor_model,
            llm_model=settings.llm_model,
            max_q=max(settings.k_values),
            projector_layers=settings.projector_layers,
            projector_hidden=settings.projector_hidden,
            compressor_use_chat_shell=(settings.compressor_chat_shell_mode == "current"),
        )
        self.trainability = get_trainability(self.settings)
        self._stage1_data_curriculum_callback: TrainerCallback | None = None
        self._configure_trainability()

        if settings.stage1_checkpoint:
            checkpoint_dir = settings.stage1_checkpoint
            if os.path.isdir(checkpoint_dir):
                print(f"[stage1] Loading checkpoint from {checkpoint_dir}", flush=True)
                llm_path = os.path.join(checkpoint_dir, "llm", "model.safetensors")
                load_llm = os.path.exists(llm_path)
                if load_llm:
                    print(f"[stage1] Found LLM checkpoint, will load it", flush=True)
                else:
                    print(f"[stage1] No LLM checkpoint found, using pretrained {settings.llm_model}", flush=True)
                self.model.load_modular_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    load_llm=load_llm,
                    load_compressor=True,
                    strict=False,
                )
            else:
                print(f"[stage1] WARNING: STAGE1_CHECKPOINT={checkpoint_dir} is not a valid directory", flush=True)

    @staticmethod
    def _stage1_wiki_token_cap_for_epoch(epoch_idx: int) -> int:
        if epoch_idx <= 0:
            return 200
        if epoch_idx == 1:
            return 800
        return 1200 + (epoch_idx - 2) * 400

    @staticmethod
    def _extract_wiki_document_text(row: dict[str, Any]) -> str | None:
        text = row.get("text")
        if not isinstance(text, str):
            text = row.get("document")
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None
        return text

    def _build_wiki_curriculum_epoch_samples(
        self,
        wiki_path: str,
        wiki_size: int,
        curriculum_epochs: int,
    ) -> tuple[list[list[ReconstructionSample]], list[int]]:
        token_caps = [self._stage1_wiki_token_cap_for_epoch(i) for i in range(curriculum_epochs)]
        buckets: dict[int, list[ReconstructionSample]] = {cap: [] for cap in token_caps}

        hf_train_split = load_dataset(path=wiki_path, split="train")
        for row in hf_train_split:
            if not isinstance(row, dict):
                continue

            text = self._extract_wiki_document_text(row)
            if text is None:
                continue

            token_count_raw = row.get("token_count")
            try:
                token_count = int(token_count_raw)
            except (TypeError, ValueError):
                continue
            if token_count <= 0:
                continue

            sample = ReconstructionSample(
                document=text,
                metadata={
                    "source": "hf_train",
                    "dataset": wiki_path,
                    "token_count": token_count,
                },
            )

            for cap in token_caps:
                bucket = buckets[cap]
                if token_count <= cap and len(bucket) < wiki_size:
                    bucket.append(sample)

            if all(len(buckets[cap]) >= wiki_size for cap in token_caps):
                break

        epoch_samples: list[list[ReconstructionSample]] = []
        for cap in token_caps:
            collected = list(buckets[cap])
            if not collected:
                raise ValueError(
                    f"No valid rows from HF dataset '{wiki_path}' with token_count <= {cap}."
                )
            if len(collected) < wiki_size:
                print(
                    (
                        f"[stage1] token_count<={cap} collected={len(collected)} < wiki_size={wiki_size}; "
                        "will repeat samples to satisfy STAGE1_WIKI_SIZE"
                    ),
                    flush=True,
                )
                cursor = 0
                while len(collected) < wiki_size:
                    collected.append(collected[cursor % len(collected)])
                    cursor += 1
            epoch_samples.append(collected[:wiki_size])

        return epoch_samples, token_caps

    def _configure_trainability(self) -> None:
        t = self.trainability
        self.model.freeze_llm(not t.train_llm)
        self.model.freeze_compressor(not t.train_compressor)
        self.model.freeze_q(not t.train_q)
        self.model.freeze_projector(not t.train_projector)

    def _build_stage1_train_dataset(self) -> Dataset:
        self._stage1_data_curriculum_callback = None
        base_dataset = ReconstructionDataset(self.settings.stage1_train_path)
        wiki_path = self.settings.stage1_wiki_dataset.strip()
        wiki_size = int(self.settings.stage1_wiki_size)
        curriculum_epochs = max(0, int(self.settings.stage1_curriculum_epochs))

        if curriculum_epochs > 0 and wiki_path and wiki_size > 0:
            wiki_epoch_samples, token_caps = self._build_wiki_curriculum_epoch_samples(
                wiki_path=wiki_path,
                wiki_size=wiki_size,
                curriculum_epochs=curriculum_epochs,
            )
            switch_dataset = _Stage1EpochSwitchDataset(
                base_samples=list(base_dataset.samples),
                wiki_epoch_samples=wiki_epoch_samples,
                wiki_token_caps=token_caps,
            )
            self._stage1_data_curriculum_callback = _Stage1CurriculumCallback(switch_dataset)
            print(
                (
                    f"[stage1] enabled wiki curriculum: first {curriculum_epochs} epoch(s) use "
                    f"token_count caps={token_caps}, each with wiki_size={wiki_size}; "
                    f"then switch to base train path={self.settings.stage1_train_path}"
                ),
                flush=True,
            )
            return switch_dataset

        if not wiki_path:
            return base_dataset

        if curriculum_epochs > 0 and wiki_size <= 0:
            print(
                (
                    "[stage1] STAGE1_CURRICULUM_EPOCHS > 0 but wiki path/size is not valid; "
                    "falling back to regular stage1 train dataset"
                ),
                flush=True,
            )
            return base_dataset

        if wiki_size <= 0:
            raise ValueError(
                "STAGE1_WIKI_SIZE must be > 0 when STAGE1_WIKI_DATASET is set"
            )

        hf_train_split = load_dataset(path=wiki_path, split="train")
        samples: list[ReconstructionSample] = []
        for row in hf_train_split:
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                continue
            if len(text) < 500:
                continue

            samples.append(
                ReconstructionSample(
                    document=text,
                    metadata={
                        "source": "hf_train",
                        "dataset": wiki_path,
                    },
                )
            )
            if len(samples) >= wiki_size:
                break

        if not samples:
            raise ValueError(
                f"No valid rows from HF dataset '{wiki_path}'. Ensure train split has a 'text' field with len >= 500."
            )
        if len(samples) < wiki_size:
            print(
                f"[stage1] requested wiki size={wiki_size}, but only collected {len(samples)} rows with text len >= 500",
                flush=True,
            )
        merged_samples = list(base_dataset.samples)
        merged_samples.extend(samples)
        print(
            (
                f"[stage1] merged base train samples={len(base_dataset)} "
                f"with wiki samples={len(samples)}; total_before_dataset_cap={len(merged_samples)}"
            ),
            flush=True,
        )
        return ReconstructionDataset.from_samples(merged_samples)

    def build_dataloader(self, train: bool) -> DataLoader:
        if train:
            dataset = self._build_stage1_train_dataset()
        else:
            dataset = ReconstructionDataset(self.settings.stage1_val_path)
        collator = ReconstructionCollator(
            tokenizer=self.model.compressor.tokenizer,
            max_length=self.settings.max_length,
        )
        return DataLoader(
            dataset,
            batch_size=self.settings.batch_size,
            shuffle=train,
            collate_fn=collator,
        )

    def run(self) -> None:
        train_loader = self.build_dataloader(train=True)
        eval_loader = self.build_dataloader(train=False)
        worker_count = min(4, os.cpu_count() or 1)

        args = TrainingArguments(
            output_dir=os.path.join(self.settings.output_dir, "stage1_reconstruction"),
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
                    },
                )

            def prediction_step(
                self, 
                model, 
                inputs: dict[str, Any], 
                prediction_loss_only: bool, 
                ignore_keys: list[str] | None = None
            ):
                """Force evaluation to use compute_loss, because batches do not expose labels."""
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    
                    if loss is not None:
                        loss = loss.mean().detach()

                return (loss, None, None)
            def compute_loss(self, model, inputs: dict[str, Any], return_outputs: bool = False, **kwargs):
                real_model = model.module if hasattr(model, "module") else model
                
                doc_ids = inputs["doc_input_ids"].to(real_model.llm.device)
                doc_mask = inputs["doc_attention_mask"].to(real_model.llm.device)
                prefix = stage1_prompt()
                tokenizer = real_model.llm_tokenizer
                llm_device = real_model.llm.device

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

                if not hasattr(self, "_stage1_shell_ids"):
                    self._stage1_shell_ids = build_chat_template_user_shell_ids(tokenizer)
                shell_prefix_ids, shell_suffix_ids = self._stage1_shell_ids
                emb_layer = real_model.llm.get_input_embeddings()

                shell_prefix_embeds = None
                if shell_prefix_ids:
                    shell_prefix_t = torch.tensor(shell_prefix_ids, device=llm_device, dtype=torch.long).unsqueeze(0)
                    shell_prefix_embeds = emb_layer(shell_prefix_t)

                shell_suffix_embeds = None
                if shell_suffix_ids:
                    shell_suffix_t = torch.tensor(shell_suffix_ids, device=llm_device, dtype=torch.long).unsqueeze(0)
                    shell_suffix_embeds = emb_layer(shell_suffix_t)

                if not hasattr(self, "_stage1_prompt_ids"):
                    self._stage1_prompt_ids = tokenizer.encode(f" {prefix}", add_special_tokens=False)
                prompt_ids = self._stage1_prompt_ids
                prompt_embeds = None
                if prompt_ids:
                    prompt_t = torch.tensor(prompt_ids, device=llm_device, dtype=torch.long).unsqueeze(0)
                    prompt_embeds = emb_layer(prompt_t)

                target_text = real_model.compressor.tokenizer.batch_decode(doc_ids, skip_special_tokens=True)

                target_text = [text + tokenizer.eos_token for text in target_text]

                llm_max_positions = int(
                    getattr(real_model.llm.config, "max_position_embeddings", int(self.args.max_length))
                )
                # Keep first-step runtime stable across ranks under long inputs.
                max_total_positions = min(int(self.args.max_length), llm_max_positions)
                max_context_len = max(1, max_total_positions - 1)

                t = tokenizer(
                    target_text,
                    add_special_tokens=False,
                    padding=True,
                    truncation=True,
                    max_length=max_total_positions,
                    return_tensors="pt",
                ).to(llm_device)
                target_embeds = emb_layer(t["input_ids"])
                target_labels = t["input_ids"].clone()

                if tokenizer.pad_token_id is not None:
                    target_labels[target_labels == tokenizer.pad_token_id] = -100

                max_k = selected_ks[-1]
                soft_all = real_model.projector(real_model.compressor(doc_ids, doc_mask, k=max_k))

                bsz = soft_all.size(0)

                losses = []
                for k in selected_ks:
                    soft = soft_all[:, :k, :]

                    context_parts = []
                    if shell_prefix_embeds is not None:
                        context_parts.append(shell_prefix_embeds.expand(bsz, -1, -1))
                    context_parts.append(soft)
                    if prompt_embeds is not None:
                        context_parts.append(prompt_embeds.expand(bsz, -1, -1))
                    if shell_suffix_embeds is not None:
                        context_parts.append(shell_suffix_embeds.expand(bsz, -1, -1))

                    context_embeds = torch.cat(context_parts, dim=1)
                    if context_embeds.size(1) > max_context_len:
                        context_embeds = context_embeds[:, :max_context_len, :]

                    target_len_allowed = max(1, max_total_positions - int(context_embeds.size(1)))
                    target_embeds_k = target_embeds[:, :target_len_allowed, :]
                    target_mask_k = t["attention_mask"][:, :target_len_allowed]
                    target_labels_k = target_labels[:, :target_len_allowed]

                    input_embeds = torch.cat([context_embeds, target_embeds_k], dim=1)

                    context_mask = torch.ones(
                        (bsz, context_embeds.size(1)),
                        device=llm_device,
                        dtype=torch.long,
                    )
                    mask = torch.cat([context_mask, target_mask_k], dim=1)

                    ignore = torch.full(
                        (bsz, context_embeds.size(1)),
                        -100,
                        device=llm_device,
                    )
                    labels = torch.cat([ignore, target_labels_k], dim=1)

                    out = real_model.llm_forward_with_embeds(
                        input_embeds=input_embeds,
                        attention_mask=mask,
                        labels=labels,
                    )
                    losses.append(out.loss)

                loss = torch.stack(losses).mean()
                return (loss, out) if return_outputs else loss

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

        trainer = _WrappedTrainer(
            model=model,
            args=args,
            train_dataset=train_loader.dataset,
            eval_dataset=eval_loader.dataset if self.settings.eval_strategy != "no" else None,
            data_collator=train_loader.collate_fn,
        )
        if self._stage1_data_curriculum_callback is not None:
            trainer.add_callback(self._stage1_data_curriculum_callback)
        trainer.train()
