/**
 * Scene Graph visualization component.
 * Renders nodes (objects) and edges (spatial relations) from a graph snapshot.
 * Uses Canvas 2D for performance; nodes are colored by confidence level.
 */

import { useRef, useEffect } from "react";
import type { GraphSnapshot } from "@/lib/api";

interface Props {
  snapshot: GraphSnapshot | null;
}

// Confidence → color mapping
function confidenceColor(c: number): string {
  if (c >= 0.7) return "#22c55e"; // green
  if (c >= 0.4) return "#eab308"; // yellow
  return "#ef4444";               // red
}

function deltaColor(uid: string, delta?: GraphSnapshot["delta_G"]): string | null {
  if (!delta) return null;
  if (delta.new_nodes.includes(uid)) return "#3b82f6";    // blue pulse for new
  if (delta.confirmed.includes(uid)) return "#22c55e";    // green for confirmed
  if (delta.declining.includes(uid)) return "#f97316";    // orange for declining
  if (delta.disappeared.includes(uid)) return "#6b7280";  // gray for disappeared
  return null;
}

export default function SceneGraph({ snapshot }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !snapshot) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const nodes = snapshot.nodes;
    if (nodes.length === 0) {
      ctx.fillStyle = "#9ca3af";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No nodes in graph", W / 2, H / 2);
      return;
    }

    // Compute bounding box of all node centers for auto-scaling
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const n of nodes) {
      minX = Math.min(minX, n.center[0]);
      maxX = Math.max(maxX, n.center[0]);
      minY = Math.min(minY, n.center[1]);
      maxY = Math.max(maxY, n.center[1]);
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const pad = 60;

    const toScreen = (cx: number, cy: number): [number, number] => [
      pad + ((cx - minX) / rangeX) * (W - 2 * pad),
      pad + ((cy - minY) / rangeY) * (H - 2 * pad),
    ];

    // Build uid → screen pos map
    const posMap: Record<string, [number, number]> = {};
    for (const n of nodes) {
      posMap[n.uid] = toScreen(n.center[0], n.center[1]);
    }

    // Draw hyperedge regions (dashed rectangles)
    for (const he of (snapshot.hyperedges || [])) {
      const memberPositions = he.members
        .map((m) => posMap[m])
        .filter(Boolean);
      if (memberPositions.length < 2) continue;

      let hMinX = Infinity, hMaxX = -Infinity, hMinY = Infinity, hMaxY = -Infinity;
      for (const [px, py] of memberPositions) {
        hMinX = Math.min(hMinX, px);
        hMaxX = Math.max(hMaxX, px);
        hMinY = Math.min(hMinY, py);
        hMaxY = Math.max(hMaxY, py);
      }
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(hMinX - 15, hMinY - 15, hMaxX - hMinX + 30, hMaxY - hMinY + 30);
      ctx.setLineDash([]);
      // Label
      ctx.fillStyle = "rgba(99, 102, 241, 0.6)";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(he.name, (hMinX + hMaxX) / 2, hMinY - 20);
    }

    // Draw edges
    for (const edge of (snapshot.edges || [])) {
      const p1 = posMap[edge.src];
      const p2 = posMap[edge.dst];
      if (!p1 || !p2) continue;
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.strokeStyle = `rgba(156, 163, 175, ${Math.min(edge.weight, 1)})`;
      ctx.lineWidth = Math.max(0.5, edge.weight * 2);
      ctx.stroke();
      // Draw relation label at midpoint
      if (edge.relation) {
        const mx = (p1[0] + p2[0]) / 2;
        const my = (p1[1] + p2[1]) / 2;
        ctx.fillStyle = "rgba(107, 114, 128, 0.7)";
        ctx.font = "9px sans-serif";
        ctx.textAlign = "center";
        const label = edge.relation.length > 18 ? edge.relation.slice(0, 16) + ".." : edge.relation;
        ctx.fillText(label, mx, my - 3);
      }
    }

    // Draw nodes
    for (const n of nodes) {
      const [sx, sy] = posMap[n.uid];
      const radius = Math.max(6, Math.min(18, 4 + n.n_obs * 1.5));

      // Delta highlight ring
      const dc = deltaColor(n.uid, snapshot.delta_G);
      if (dc) {
        ctx.beginPath();
        ctx.arc(sx, sy, radius + 4, 0, Math.PI * 2);
        ctx.strokeStyle = dc;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // FOV indicator
      if (n.is_in_fov) {
        ctx.beginPath();
        ctx.arc(sx, sy, radius + 7, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(59, 130, 246, 0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(sx, sy, radius, 0, Math.PI * 2);
      ctx.fillStyle = confidenceColor(n.confidence);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label
      ctx.fillStyle = "#1f2937";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      const label = n.caption.length > 15 ? n.caption.slice(0, 13) + ".." : n.caption;
      ctx.fillText(label, sx, sy + radius + 14);

      // Confidence text
      ctx.fillStyle = "#6b7280";
      ctx.font = "9px sans-serif";
      ctx.fillText(n.confidence.toFixed(2), sx, sy + radius + 25);
    }

    // Draw agent pose
    const ap = snapshot.agent_pose;
    if (ap) {
      const [ax, ay] = toScreen(ap.x, ap.y);
      const size = 10;
      ctx.beginPath();
      ctx.moveTo(ax + size * Math.cos(ap.heading), ay + size * Math.sin(ap.heading));
      ctx.lineTo(
        ax + (size * 0.6) * Math.cos(ap.heading + (Math.PI * 4) / 3),
        ay + (size * 0.6) * Math.sin(ap.heading + (Math.PI * 4) / 3)
      );
      ctx.lineTo(
        ax + (size * 0.6) * Math.cos(ap.heading - (Math.PI * 4) / 3),
        ay + (size * 0.6) * Math.sin(ap.heading - (Math.PI * 4) / 3)
      );
      ctx.closePath();
      ctx.fillStyle = "#2563eb";
      ctx.fill();
    }

    // Step label
    ctx.fillStyle = "#9ca3af";
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`Step ${snapshot.step}  |  ${nodes.length} nodes  |  ${(snapshot.edges || []).length} edges`, 8, H - 8);

  }, [snapshot]);

  return (
    <canvas
      ref={canvasRef}
      width={500}
      height={400}
      className="w-full h-full"
      style={{ imageRendering: "auto" }}
    />
  );
}
