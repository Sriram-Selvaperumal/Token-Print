"use client";

import React, { useMemo, useState } from "react";
import { Billboard, Text } from "@react-three/drei";
import { useStore } from "@/lib/store";
import type { MoELayerRouting } from "@/lib/types";

interface ExpertLanes3DProps {
  layerIndex: number;
  position?: [number, number, number];
}

interface HoveredState {
  idx: number;
  weight: number;
  xPos: number;
}

/**
 * 3D Mixture-of-Experts (MoE) routing visualization lanes.
 *
 * Placed beneath the layer MLP stage, this component renders the gate router's
 * expert selection lanes in real-time as tokens are generated (or analyzed).
 * Active top-k experts light up with a warm amber emissive glow proportional
 * to their routing weight, while inactive experts remain in dim slate.
 *
 * If the model is dense (no expert_routing or moe_routing data), this component
 * returns null to ensure zero runtime and visual overhead.
 */
export const ExpertLanes3D = React.memo(function ExpertLanes3D({
  layerIndex,
  position = [0, 0, 0],
}: ExpertLanes3DProps) {
  const genFrames = useStore((s) => s.genFrames);
  const playIndex = useStore((s) => s.playIndex);
  const analyzeData = useStore((s) => s.data);

  const [hovered, setHovered] = useState<HoveredState | null>(null);

  // Fast check: does ANY frame or analysis payload have MoE routing?
  const hasMoEData = useMemo(() => {
    if (analyzeData?.moe_routing?.per_layer?.length) return true;
    return genFrames.some((f) => Boolean(f.expert_routing?.per_layer?.length));
  }, [analyzeData, genFrames]);

  // Extract the routing entry for this layer for the currently displayed token.
  const layerRouting: MoELayerRouting | null = useMemo(() => {
    if (!hasMoEData) return null;

    // 1. Check generation playback frame first
    if (genFrames.length > 0) {
      const idx = playIndex >= 0 && playIndex < genFrames.length ? playIndex : genFrames.length - 1;
      const frame = genFrames[idx];
      const match = frame?.expert_routing?.per_layer?.find((r) => r.layer === layerIndex);
      if (match) return match;
    }

    // 2. Fall back to static analyze moe_routing if available
    const staticMatch = analyzeData?.moe_routing?.per_layer?.find((r) => r.layer === layerIndex);
    return staticMatch ?? null;
  }, [hasMoEData, genFrames, playIndex, analyzeData, layerIndex]);

  // Resolve active expert weights for the latest decoded position in this step.
  const currentRoutingEntry =
    layerRouting?.routing && layerRouting.routing.length > 0
      ? layerRouting.routing[layerRouting.routing.length - 1]
      : null;

  const weightsMap = useMemo(() => {
    const map = new Map<number, number>();
    if (currentRoutingEntry?.experts) {
      for (const e of currentRoutingEntry.experts) {
        map.set(e.idx, e.weight);
      }
    }
    return map;
  }, [currentRoutingEntry]);

  // If no MoE data exists for this model or layer, unmount cleanly.
  if (!layerRouting || !layerRouting.n_experts) {
    return null;
  }

  const { n_experts, used } = layerRouting;

  // Dynamic layout calculations based on expert count (e.g. 8 for Mixtral, 64 for DeepSeek).
  const maxSpan = 9.0;
  const spacing = Math.min(0.9, maxSpan / Math.max(n_experts, 1));
  const barWidth = Math.max(0.12, Math.min(0.5, spacing * 0.75));
  const baseHeight = 0.65;
  const barDepth = Math.max(0.15, Math.min(0.35, spacing * 0.6));
  const totalWidth = (n_experts - 1) * spacing;

  return (
    <group position={position}>
      {/* ── Header Title & Active Count ── */}
      <Billboard position={[0, 0.85, 0]}>
        <Text
          fontSize={0.22}
          color="#f97316"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.015}
          outlineColor="#090d16"
        >
          {`MoE Router · ${used}/${n_experts} Active`}
        </Text>
      </Billboard>

      {/* ── Subtitle indicating token step / routing ── */}
      {currentRoutingEntry && (
        <Billboard position={[0, 0.62, 0]}>
          <Text
            fontSize={0.14}
            color="#94a3b8"
            anchorX="center"
            anchorY="middle"
          >
            {`Pos ${currentRoutingEntry.token} · Top-${used} Experts`}
          </Text>
        </Billboard>
      )}

      {/* ── Base Track / Guide Rail ── */}
      <mesh position={[0, -0.38, 0]}>
        <boxGeometry args={[totalWidth + barWidth + 0.4, 0.05, barDepth + 0.2]} />
        <meshStandardMaterial
          color="#0f172a"
          roughness={0.8}
          metalness={0.2}
          transparent
          opacity={0.7}
        />
      </mesh>

      {/* ── Expert Lane Pillars ── */}
      {Array.from({ length: n_experts }, (_, idx) => {
        const x = -totalWidth / 2 + idx * spacing;
        const weight = weightsMap.get(idx) ?? 0;
        const isActive = weight > 0;

        // Active experts scale slightly taller with weight to show physical dominance
        const heightScale = isActive ? 1.0 + weight * 0.6 : 0.8;
        const currentHeight = baseHeight * heightScale;
        const y = -0.35 + currentHeight / 2;

        const isHovered = hovered?.idx === idx;

        // Color palette: warm amber/orange (#f97316 / #fb923c) for active, slate (#1e293b) for inactive
        const barColor = isActive ? "#f97316" : "#1e293b";
        const emissiveColor = isActive ? "#ea580c" : "#020617";
        const emissiveIntensity = isActive ? 0.9 + weight * 1.6 : isHovered ? 0.3 : 0.05;
        const opacity = isActive ? 0.95 : isHovered ? 0.7 : 0.35;

        return (
          <group key={idx} position={[x, y, 0]}>
            <mesh
              onPointerOver={(e) => {
                e.stopPropagation();
                setHovered({ idx, weight, xPos: x });
              }}
              onPointerOut={() => setHovered(null)}
            >
              <boxGeometry args={[barWidth, currentHeight, barDepth]} />
              <meshStandardMaterial
                color={barColor}
                emissive={emissiveColor}
                emissiveIntensity={emissiveIntensity}
                roughness={isActive ? 0.2 : 0.7}
                metalness={isActive ? 0.4 : 0.1}
                transparent
                opacity={opacity}
              />
            </mesh>

            {/* Expert Number Tag under the lane (shown for smaller expert sets or active ones) */}
            {(n_experts <= 16 || isActive || isHovered) && (
              <Billboard position={[0, -currentHeight / 2 - 0.14, 0]}>
                <Text
                  fontSize={0.11}
                  color={isActive ? "#fb923c" : "#64748b"}
                  anchorX="center"
                  anchorY="top"
                >
                  {`E${idx}`}
                </Text>
              </Billboard>
            )}

            {/* Weight Percentage Tag on top of active lanes */}
            {isActive && (
              <Billboard position={[0, currentHeight / 2 + 0.14, 0]}>
                <Text
                  fontSize={0.12}
                  color="#fed7aa"
                  anchorX="center"
                  anchorY="bottom"
                  outlineWidth={0.01}
                  outlineColor="#431407"
                >
                  {`${(weight * 100).toFixed(0)}%`}
                </Text>
              </Billboard>
            )}
          </group>
        );
      })}

      {/* ── Interactive Hover Tooltip ── */}
      {hovered !== null && (
        <Billboard position={[hovered.xPos, 0.45, barDepth / 2 + 0.4]}>
          <group>
            {/* Tooltip Background Badge */}
            <mesh position={[0, 0, -0.01]}>
              <planeGeometry args={[1.6, 0.42]} />
              <meshBasicMaterial color="#0b0f19" transparent opacity={0.92} />
            </mesh>
            <Text
              fontSize={0.13}
              color="#ffffff"
              anchorX="center"
              anchorY="middle"
              outlineWidth={0.012}
              outlineColor="#000000"
            >
              {`Expert ${hovered.idx} · ${hovered.weight > 0 ? (hovered.weight * 100).toFixed(1) + "%" : "Inactive (0%)"
                }`}
            </Text>
          </group>
        </Billboard>
      )}
    </group>
  );
});
