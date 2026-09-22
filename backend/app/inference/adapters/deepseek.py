"""DeepSeek-MoE model family capability adapter (Issue #294 -- MoE routing).

DeepSeek-V2/V3 use a distinct MoE architecture with config keys that differ
from Mixtral.  Verified from the real deepseek-ai/DeepSeek-V2-Lite HF repo:

  model_type: "deepseek_v2"
  n_routed_experts:   64  (total routed experts per MoE layer)
  num_experts_per_tok: 6  (active experts per token)
  n_shared_experts:    2  (always-on shared experts, not routed)
  moe_layer_freq:      1  (every layer is an MoE layer for V2-Lite)

The existing _detect_moe_blocks() generic fallback already handles DeepSeek
correctly: it scans for an "experts" ModuleList + "gate"/"router" Linear and
reads num_experts_per_tok from config.  It will find n_routed_experts experts
(the shared experts are separate and not in the same ModuleList as routed ones)
and top-k = num_experts_per_tok.  No extra config-key parsing is needed here.
We document this explicitly so future contributors do not add redundant logic.
"""

from __future__ import annotations

from typing import Any

from app.inference.adapters.base import ModelAdapter
from app.inference.capabilities import CapabilityStatus, ModelCapabilities


class DeepseekAdapter(ModelAdapter):
    family_name = "deepseek"

    def matches_config(self, config: dict[str, Any]) -> bool:
        model_type = str(config.get("model_type", "")).lower()
        architectures = [str(a).lower() for a in config.get("architectures", [])]
        return "deepseek" in model_type or any("deepseek" in a for a in architectures)

    def get_capabilities(self, config: dict[str, Any]) -> ModelCapabilities:
        param_count = self._extract_param_count(config)
        vram_est = self.estimate_vram(config)
        max_ctx = config.get("max_position_embeddings") or 163840

        # DeepSeek-V2/V3 are MoE: n_routed_experts is the canonical key.
        # The _detect_moe_blocks() generic detector already handles this model
        # correctly via the "experts" ModuleList + "gate" Linear scan.
        # We report MoE routing as supported=True with high confidence.
        has_moe = (
            "n_routed_experts" in config
            or "num_local_experts" in config
        )
        moe_status = CapabilityStatus(
            supported=has_moe,
            confidence="high" if has_moe else "medium",
            reason=(
                "Router logits captured via forward hooks on the gate module "
                "(n_routed_experts / num_experts_per_tok; _detect_moe_blocks() generic "
                "scan confirmed to match DeepSeek-V2 module layout)."
                if has_moe
                else "No MoE config keys detected; treating as dense DeepSeek variant."
            ),
        )

        return ModelCapabilities(
            supports_attention=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Multi-head latent attention weights exposed via output_attentions=True.",
            ),
            supports_hidden_states=CapabilityStatus(
                supported=True,
                confidence="high",
                reason="Residual stream hidden states accessible via output_hidden_states=True.",
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
            supports_moe_routing=moe_status,
            max_context_length=max_ctx,
            parameter_count=param_count,
            architecture="DeepseekV2ForCausalLM",
            vram_estimate=vram_est,
        )
