"""Qwen model family capability adapter."""

from __future__ import annotations

from typing import Any

from app.inference.adapters.base import ModelAdapter
from app.inference.capabilities import CapabilityStatus, ModelCapabilities


class QwenAdapter(ModelAdapter):
    family_name = "qwen"

    def matches_config(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = [str(a).lower() for a in config.get("architectures", [])]
        return "qwen" in model_type or any("qwen" in a for a in architectures)

    def get_capabilities(self, config: dict[str, Any]) -> ModelCapabilities:
        param_count = self._extract_param_count(config)
        vram_est = self.estimate_vram(config)
        max_ctx = config.get("max_position_embeddings") or config.get("seq_length") or 32768

        _has_moe = "num_experts" in config or "num_local_experts" in config
        moe_status = CapabilityStatus(
            supported=_has_moe,
            confidence="high",
            reason=(
                "Router logits captured via forward hooks on the gate module "
                "(num_experts / num_experts_per_tok present in the Qwen2-MoE config)."
                if _has_moe
                else "Dense model; no expert routing."
            ),
        )

        return ModelCapabilities(
            supports_attention=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Native softmax attention weights exposed via output_attentions=True.",
            ),
            supports_hidden_states=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Intermediate residual stream hidden states accessible via output_hidden_states=True.",
            ),
            supports_logit_lens=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="RMSNorm and lm_head projection supported for logit lens decomposition.",
            ),
            supports_head_ablation=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Instrumented PyTorch forward hooks support per-head zeroing.",
            ),
            supports_layer_ablation=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="PyTorch layer forward pass bypass enabled.",
            ),
            supports_activation_patch=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Residual stream state injection supported.",
            ),
            supports_moe_routing=moe_status,
            max_context_length=max_ctx,
            parameter_count=param_count,
            architecture="Qwen2ForCausalLM",
            vram_estimate=vram_est,
        )
