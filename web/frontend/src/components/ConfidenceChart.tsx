/**
 * Node confidence over time chart.
 * Fetches snapshots progressively and plots confidence trends for top nodes.
 */

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { api, SnapshotListItem, GraphSnapshot } from "@/lib/api";

interface Props {
  taskId: string;
  snapshots: SnapshotListItem[];
  currentStep?: number;
}

const COLORS = [
  "#ef4444", "#3b82f6", "#22c55e", "#eab308", "#a855f7",
  "#ec4899", "#14b8a6", "#f97316", "#6366f1", "#84cc16",
];

export default function ConfidenceChart({ taskId, snapshots, currentStep }: Props) {
  const [chartData, setChartData] = useState<any[]>([]);
  const [nodeNames, setNodeNames] = useState<string[]>([]);

  useEffect(() => {
    if (!taskId || snapshots.length === 0) return;

    // Sample up to 30 snapshots evenly to avoid flooding the API
    const step = Math.max(1, Math.floor(snapshots.length / 30));
    const sampled = snapshots.filter((_, i) => i % step === 0);

    const loadData = async () => {
      const allNodes = new Set<string>();
      const rows: any[] = [];

      for (const s of sampled) {
        try {
          const snap = await api.getSnapshot(taskId, s.step);
          const row: Record<string, any> = { step: snap.step };
          for (const n of snap.nodes) {
            row[n.caption] = n.confidence;
            allNodes.add(n.caption);
          }
          rows.push(row);
        } catch {
          // Skip failed snapshot loads
        }
      }

      // Pick top 8 most frequent nodes
      const freq: Record<string, number> = {};
      for (const r of rows) {
        for (const key of Object.keys(r)) {
          if (key !== "step") freq[key] = (freq[key] || 0) + 1;
        }
      }
      const topNames = Object.entries(freq)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([name]) => name);

      setNodeNames(topNames);
      setChartData(rows);
    };

    loadData();
  }, [taskId, snapshots]);

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400 text-sm">
        No confidence data available
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="step"
          tick={{ fontSize: 11 }}
          label={{ value: "Step", position: "insideBottom", offset: -2, fontSize: 11 }}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 11 }}
          label={{ value: "Confidence", angle: -90, position: "insideLeft", fontSize: 11 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 8 }}
          formatter={(value: number) => value?.toFixed(3)}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {currentStep !== undefined && (
          <ReferenceLine x={currentStep} stroke="#6366f1" strokeDasharray="3 3" label="" />
        )}
        {nodeNames.map((name, i) => (
          <Line
            key={name}
            type="monotone"
            dataKey={name}
            stroke={COLORS[i % COLORS.length]}
            dot={false}
            strokeWidth={1.5}
            connectNulls
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
