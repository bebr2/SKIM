from __future__ import annotations

import json
import os

import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None

from models.compressor import SKIMCompressor, _get_hidden_size
from models.projector import MLPProjector


class SKIMModel(nn.Module):
    def __init__(
        self,
        compressor_model: str,
        llm_model: str,
        max_q: int,
        projector_layers: int,
        projector_hidden: int,
        compressor_use_chat_shell: bool = True,
        llm_device_map: str | dict | None = None,
        llm_max_memory: dict | None = None,
        torch_dtype: torch.dtype | None = None,
        skip_compressor: bool = False,
    ) -> None:
        super().__init__()
        self._skip_compressor = skip_compressor

        if skip_compressor:
            self.compressor = None
            self._compressor_hidden_size = 4096
        else:
            self.compressor = SKIMCompressor(
                compressor_model,
                max_q=max_q,
                use_chat_shell=compressor_use_chat_shell,
                torch_dtype=torch_dtype,
            )
            self._compressor_hidden_size = self.compressor.hidden_size

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        if llm_device_map is not None:
            if llm_max_memory is not None:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_model, device_map=llm_device_map, max_memory=llm_max_memory, torch_dtype=torch_dtype
                )
            else:
                self.llm = AutoModelForCausalLM.from_pretrained(
                    llm_model, device_map=llm_device_map, torch_dtype=torch_dtype
                )
            self._llm_device_map = llm_device_map
            self._llm_max_memory = llm_max_memory
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch_dtype)
            self._llm_device_map = None
            self._llm_max_memory = None

        self.projector = MLPProjector(
            in_dim=self._compressor_hidden_size,
            out_dim=_get_hidden_size(self.llm.config),
            hidden_dim=projector_hidden,
            layers=projector_layers,
            torch_dtype=torch_dtype,
        )
        self._llm_lora_enabled = False

    def get_llm_first_device(self) -> torch.device:
        """Return the first LLM device for projector placement."""
        if self._llm_device_map is not None:
            if hasattr(self.llm, "hf_device_map"):
                first_device = list(self.llm.hf_device_map.values())[0]
                if isinstance(first_device, torch.device):
                    return first_device
                return torch.device(first_device)
        return self.llm.device if hasattr(self.llm, "device") else torch.device("cpu")

    def enable_llm_lora(
        self,
        r: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        modules_to_save: list[str] | None = None,
        use_rslora: bool = False,
    ) -> None:
        if get_peft_model is None or LoraConfig is None:
            raise ImportError("LoRA mode requires `peft`. Please install peft>=0.12.0.")
        if not target_modules:
            raise ValueError("target_modules must not be empty when LoRA is enabled")
        if TaskType is None:
            raise RuntimeError("peft.TaskType is unavailable")

        task_type_key = str(task_type).upper()
        if not hasattr(TaskType, task_type_key):
            raise ValueError(f"Unsupported LoRA task type: {task_type}")

        config = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            target_modules=list(target_modules),
            bias=str(bias),
            task_type=getattr(TaskType, task_type_key),
            modules_to_save=list(modules_to_save) if modules_to_save else None,
            use_rslora=bool(use_rslora),
        )
        self.llm = get_peft_model(self.llm, config)
        self._llm_lora_enabled = True

    def set_llm_lora_trainable(self) -> None:
        if not self._llm_lora_enabled:
            raise RuntimeError("LLM LoRA is not enabled")
        for name, param in self.llm.named_parameters():
            param.requires_grad = ("lora_" in name) or ("modules_to_save" in name)

    def freeze_llm(self, freeze: bool = True) -> None:
        for p in self.llm.parameters():
            p.requires_grad = not freeze

    def freeze_compressor(self, freeze: bool = True) -> None:
        for p in self.compressor.backbone.parameters():
            p.requires_grad = not freeze

    def freeze_q(self, freeze: bool = True) -> None:
        self.compressor.learnable_q.requires_grad = not freeze

    def freeze_projector(self, freeze: bool = True) -> None:
        for p in self.projector.parameters():
            p.requires_grad = not freeze

    def encode_docs_soft_tokens(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        k: int,
        doc_micro_batch: int = 8,
    ) -> torch.Tensor:
        # doc_input_ids: [B, N, L]
        bsz, n_docs, _ = doc_input_ids.shape
        chunks = []
        for start in range(0, n_docs, doc_micro_batch):
            end = min(start + doc_micro_batch, n_docs)
            ids = doc_input_ids[:, start:end, :].reshape(-1, doc_input_ids.size(-1))
            mask = doc_attention_mask[:, start:end, :].reshape(-1, doc_attention_mask.size(-1))
            z = self.compressor(ids, mask, k=k)
            s = self.projector(z)
            s = s.reshape(bsz, end - start, k, -1)
            chunks.append(s)
        return torch.cat(chunks, dim=1)

    def llm_forward_with_embeds(
        self,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        return self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Forward gradient checkpointing to both model backbones."""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(**kwargs)
        if hasattr(self.compressor.backbone, "gradient_checkpointing_enable"):
            self.compressor.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on both model backbones."""
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()
        if hasattr(self.compressor.backbone, "gradient_checkpointing_disable"):
            self.compressor.backbone.gradient_checkpointing_disable()

    def save_modular_checkpoint(
        self,
        checkpoint_dir: str,
        include_llm: bool,
        metadata: dict | None = None,
        state_dict: dict | None = None,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.state_dict()

        compressor_dir = os.path.join(checkpoint_dir, "compressor")
        llm_dir = os.path.join(checkpoint_dir, "llm")
        q_projector_dir = os.path.join(checkpoint_dir, "q_projector")
        os.makedirs(compressor_dir, exist_ok=True)
        os.makedirs(q_projector_dir, exist_ok=True)

        compressor_state = {
            k.replace("compressor.backbone.", ""): v.cpu()
            for k, v in state_dict.items() if k.startswith("compressor.backbone.")
        }
        
        shared_ptrs = set()
        for k, v in list(compressor_state.items()):
            if v.data_ptr() in shared_ptrs:
                compressor_state[k] = v.clone()
            else:
                shared_ptrs.add(v.data_ptr())

        save_safetensors_file(compressor_state, os.path.join(compressor_dir, "model.safetensors"))

        q_projector_state = {
            "learnable_q": state_dict["compressor.learnable_q"].cpu(),
        }
        q_projector_state.update({
            k: v.cpu()
            for k, v in state_dict.items() if k.startswith("projector.")
        })
        save_safetensors_file(
            q_projector_state,
            os.path.join(q_projector_dir, "model.safetensors"),
        )

        if include_llm:
            os.makedirs(llm_dir, exist_ok=True)
            llm_state = {
                k.replace("llm.", ""): v.cpu()
                for k, v in state_dict.items() if k.startswith("llm.")
            }
            
            shared_ptrs_llm = set()
            for k, v in list(llm_state.items()):
                if v.data_ptr() in shared_ptrs_llm:
                    llm_state[k] = v.clone()
                else:
                    shared_ptrs_llm.add(v.data_ptr())
                    
            save_safetensors_file(llm_state, os.path.join(llm_dir, "model.safetensors"))
        save_payload = {
            "include_llm": include_llm,
            "max_q": self.compressor.max_q if self.compressor is not None else self._compressor_hidden_size,
            "projector_layers": len(self.projector.net),
            "projector_hidden": getattr(self.projector.net[0], "out_features", None),
            "compressor_use_chat_shell": bool(self.compressor.use_chat_shell) if self.compressor is not None else True,
        }
        if metadata is not None:
            save_payload.update(metadata)
        with open(os.path.join(checkpoint_dir, "skim_checkpoint.json"), "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=2)

    def load_modular_checkpoint(
        self,
        checkpoint_dir: str,
        load_llm: bool = True,
        load_compressor: bool = True,
        strict: bool = False,
    ) -> None:
        meta_path = os.path.join(checkpoint_dir, "skim_checkpoint.json")
        meta: dict = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        llm_train_mode = str(meta.get("llm_train_mode", "")).strip().lower()
        if load_llm and llm_train_mode == "lora" and not self._llm_lora_enabled:
            lora_meta = meta.get("skill_qa_lora") if isinstance(meta.get("skill_qa_lora"), dict) else {}
            self.enable_llm_lora(
                r=int(lora_meta.get("r", 16)),
                alpha=int(lora_meta.get("alpha", 32)),
                dropout=float(lora_meta.get("dropout", 0.05)),
                target_modules=[
                    str(x).strip()
                    for x in lora_meta.get(
                        "target_modules",
                        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    )
                    if str(x).strip()
                ],
                bias=str(lora_meta.get("bias", "none")),
                task_type=str(lora_meta.get("task_type", "CAUSAL_LM")),
                modules_to_save=[
                    str(x).strip()
                    for x in lora_meta.get("modules_to_save", [])
                    if str(x).strip()
                ],
                use_rslora=bool(lora_meta.get("use_rslora", False)),
            )

        compressor_path = os.path.join(checkpoint_dir, "compressor", "model.safetensors")
        llm_path = os.path.join(checkpoint_dir, "llm", "model.safetensors")
        q_projector_path = os.path.join(checkpoint_dir, "q_projector", "model.safetensors")

        if load_compressor and self.compressor is not None:
            if os.path.exists(compressor_path):
                compressor_state = load_safetensors_file(compressor_path)
                self.compressor.backbone.load_state_dict(compressor_state, strict=strict)
                print(f"[SKIMModel] Loaded compressor checkpoint from {compressor_path}")
            else:
                print(f"[SKIMModel] WARNING: compressor checkpoint not found at {compressor_path}, using random weights")

            if os.path.exists(q_projector_path):
                q_projector_state = load_safetensors_file(q_projector_path)
                if "learnable_q" in q_projector_state:
                    saved_q = q_projector_state["learnable_q"]
                    with torch.no_grad():
                        rows = min(saved_q.size(0), self.compressor.learnable_q.size(0))
                        cols = min(saved_q.size(1), self.compressor.learnable_q.size(1))
                        self.compressor.learnable_q[:rows, :cols].copy_(
                            saved_q[:rows, :cols].to(
                                device=self.compressor.learnable_q.device,
                                dtype=self.compressor.learnable_q.dtype,
                            )
                        )

                projector_state = {
                    k.replace("projector.", "", 1): v
                    for k, v in q_projector_state.items()
                    if k.startswith("projector.")
                }
                if projector_state:
                    self.projector.load_state_dict(projector_state, strict=strict)
                    print(f"[SKIMModel] Loaded projector checkpoint from {q_projector_path}")
                else:
                    print(f"[SKIMModel] WARNING: projector checkpoint not found at {q_projector_path}, using random weights")
        elif load_compressor and self.compressor is None:
            print(f"[SKIMModel] WARNING: load_compressor=True but compressor is None (skip_compressor was set)")

        if load_llm and os.path.exists(llm_path):
            llm_state = load_safetensors_file(llm_path)
            self.llm.load_state_dict(llm_state, strict=strict)
            print(f"[SKIMModel] Loaded LLM checkpoint from {llm_path}")
