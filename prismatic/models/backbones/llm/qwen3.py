# qwen3.py
"""
qwen3.py

Class definition for all LLMs derived from QwenForCausalLM (Qwen3 series).
"""

from typing import Optional, Sequence, Type

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting.base_prompter import PromptBuilder
from prismatic.models.backbones.llm.prompting.qwen3_prompter import Qwen3PromptBuilder
from peft import LoraConfig

# Registry =>> Support Qwen-3 Models (from HF Transformers)
QWEN3_MODELS = {
    "qwen3-0_6b-pure": {
        "llm_family": "qwen3", 
        "llm_cls": AutoModelForCausalLM, 
        "hf_hub_path": "Qwen/Qwen3-0.6B"
    },
    "qwen3-0_6b-extra": {
        "llm_family": "qwen3", 
        "llm_cls": AutoModelForCausalLM, 
        "hf_hub_path": "Qwen/Qwen3-0.6B"
    },
    # Add additional Qwen3 variants here if needed:
    # "qwen3-3b-pure": {...},
    # "qwen3-3b-extra": {...},
}


class Qwen3LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
        num_extra_tokens: int = 0,
        enable_peft: bool = False,
        lora_peft_config: LoraConfig = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            enable_peft=enable_peft,
            lora_config=lora_peft_config,
            **QWEN3_MODELS[llm_backbone_id],
        )

        # Optionally add extra tokens (e.g., for new task-specific markers)
        if num_extra_tokens > 0:
            added = self.tokenizer.add_tokens([f"<|extra_{i}|>" for i in range(num_extra_tokens)])
            assert added == num_extra_tokens, f"Added {added} of {num_extra_tokens} extra tokens to tokenizer!"
            print(f"Added {num_extra_tokens} extra tokens.")

        # Ensure pad token is set and embeddings resized to multiple of 64
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Qwen3PromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[torch.nn.Module]:
        return Qwen3DecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[torch.nn.Module]:
        # You may adjust which modules to fine-tune in the final layer
        return (
            self.llm.model.embed_tokens,
            self.llm.model.layers[-1],
            self.llm.lm_head,
        )
