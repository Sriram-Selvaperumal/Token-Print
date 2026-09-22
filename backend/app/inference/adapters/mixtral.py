"""Mixtral model family capability adapter (Issue #294 -- MoE routing).

Verified config facts for mistralai/Mixtral-8x7B-Instruct-v0.1:
  - model_type: "mixtral"  (disjoint from plain Mistral "mistral")
  - num_local_experts: 8
  - num_experts_per_tok: 2
"""

from __future__ import annotations

from typing import Any

from app.inference.adapters.base import ModelAdapter
from app.inference.capabilities import CapabilityStatus, ModelCapabilities


class MixtralAdapter(ModelAdapter):
    family_name = "mixtral"

    def matches_config(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = [str(a).lower() for a in config.get("architectures", [])]
        # model_type == "mixtral" is confirmed disjoint from plain Mistral
        # ("mistral") in HF configs, preventing accidental over-matching.
        return model_type == "mixtral" or any("mixtral" in a for a in architectures)

    def get_capabilities(self, config: dict[str, Any]) -> ModelCapabilities:
        param_count = self._extract_param_count(config)
        vram_est = self.estimate_vram(config)
        max_ctx = config.get("max_position_embeddings") or 32768

        return ModelCapabilities(
            supports_attention=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Native sliding window attention weights exposed via output_attentions=True.",
            ),
            supports_hidden_states=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Intermediate residual stream hidden states accessible.",
            ),
            supports_logit_lens=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="RMSNorm + lm_head projection supported.",
            ),
            supports_head_ablation=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Attention head zeroing supported.",
            ),
            supports_layer_ablation=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Layer bypass supported.",
            ),
            supports_activation_patch=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Residual stream state injection supported.",
            ),
            supports_moe_routing=CapabilityStatus(
                supported=True,
                confidence="high",
                reason=(
                    "Router logits captured via forward hooks on the gate module "
                    "(num_local_experts / num_experts_per_tok confirmed in Mixtral HF config)."
                ),
            ),
            max_context_length=max_ctx,
            parameter_count=param_count,
            architecture="MixtralForCausalLM",
            # Mixtral is sparsely activated (top-2 of 8 experts per token), so
            # active compute FLOPs are ~25% of an equivalent-parameter dense model.
            # However, ALL expert weights reside in VRAM simultaneously, so the
            # weight-size estimate from base.estimate_vram() (params x 2 bytes x
            # 1.2 overhead) is correct for *memory planning* -- it accurately
            # captures resident VRAM even though it overstates active FLOPs.
            vram_estimate=vram_est,
        )
