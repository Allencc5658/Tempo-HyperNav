import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { GitCompare } from "lucide-react";
import { api, Task } from "@/lib/api";

interface CompareRow {
  task_id: string;
  display_name: string;
  success: number;
  spl: number;
  total_steps: number;
  disable_temporal: boolean;
  disable_hypergraph: boolean;
  disable_peek: boolean;
}

export default function Compare() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<CompareRow[]>([]);

  useEffect(() => {
    api.listTasks("done").then(setTasks);
  }, []);

  const toggle = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const doCompare = async () => {
    if (selected.size < 2) return;
    const data = await api.compareTasks(Array.from(selected));
    setResults(data);
  };

  // Prepare chart data
  const chartData = results.map((r) => ({
    name: r.display_name.slice(0, 20),
    SR: r.success * 100,
    SPL: r.spl * 100,
    Steps: r.total_steps,
  }));

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Ablation Comparison</h2>

      {/* Task selector */}
      <div className="bg-white border rounded-xl p-4 mb-6">
        <p className="text-sm text-gray-500 mb-3">Select 2+ completed tasks to compare:</p>
        <div className="space-y-2 max-h-60 overflow-y-auto custom-scroll">
          {tasks.map((t) => (
            <label
              key={t.id}
              className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-50 cursor-pointer"
            >
              <input
                type="checkbox"
                checked={selected.has(t.id)}
                onChange={() => toggle(t.id)}
                className="rounded"
              />
              <span className="text-sm text-gray-700 flex-1">{t.display_name}</span>
              <span className="text-xs text-gray-400">{t.id}</span>
            </label>
          ))}
          {tasks.length === 0 && (
            <p className="text-gray-400 text-sm py-4 text-center">No completed tasks to compare.</p>
          )}
        </div>
        <button
          onClick={doCompare}
          disabled={selected.size < 2}
          className="mt-4 flex items-center gap-2 px-4 py-2 bg-brand-600 text-white rounded-lg text-sm
                     hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          <GitCompare size={14} />
          Compare Selected ({selected.size})
        </button>
      </div>

      {/* Results table */}
      {results.length > 0 && (
        <>
          <div className="bg-white border rounded-xl overflow-hidden mb-6">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-gray-600">
                <tr>
                  <th className="text-left px-4 py-3">Task</th>
                  <th className="text-center px-4 py-3">SR</th>
                  <th className="text-center px-4 py-3">SPL</th>
                  <th className="text-center px-4 py-3">Steps</th>
                  <th className="text-center px-4 py-3">Temporal</th>
                  <th className="text-center px-4 py-3">Hypergraph</th>
                  <th className="text-center px-4 py-3">Peek</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r) => (
                  <tr key={r.task_id} className="border-t hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium">{r.display_name}</td>
                    <td className="text-center px-4 py-3">{(r.success * 100).toFixed(1)}%</td>
                    <td className="text-center px-4 py-3">{(r.spl * 100).toFixed(1)}%</td>
                    <td className="text-center px-4 py-3">{r.total_steps}</td>
                    <td className="text-center px-4 py-3">
                      <Badge enabled={!r.disable_temporal} />
                    </td>
                    <td className="text-center px-4 py-3">
                      <Badge enabled={!r.disable_hypergraph} />
                    </td>
                    <td className="text-center px-4 py-3">
                      <Badge enabled={!r.disable_peek} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Chart */}
          <div className="bg-white border rounded-xl p-4">
            <h3 className="text-sm font-medium text-gray-600 mb-4">Performance Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="SR" fill="#22c55e" />
                <Bar dataKey="SPL" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}

function Badge({ enabled }: { enabled: boolean }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
        enabled ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"
      }`}
    >
      {enabled ? "ON" : "OFF"}
    </span>
  );
}
