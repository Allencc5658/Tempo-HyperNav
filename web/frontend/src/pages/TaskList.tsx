import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { RefreshCw, Play, Eye, Trash2, CheckCircle, XCircle, Clock, Loader2, Inbox } from "lucide-react";
import { api, Task } from "@/lib/api";

const statusConfig: Record<string, { icon: typeof Clock; color: string; label: string }> = {
  queued: { icon: Clock, color: "text-yellow-600 bg-yellow-50 border-yellow-200", label: "Queued" },
  running: { icon: Loader2, color: "text-blue-600 bg-blue-50 border-blue-200", label: "Running" },
  done: { icon: CheckCircle, color: "text-green-600 bg-green-50 border-green-200", label: "Done" },
  failed: { icon: XCircle, color: "text-red-500 bg-red-50 border-red-200", label: "Failed" },
};

export default function TaskList() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchTasks = async () => {
    try {
      const data = await api.listTasks();
      setTasks(data);
    } catch (e) {
      console.error("Failed to load tasks:", e);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (task: Task) => {
    if (!window.confirm(`Delete task "${task.display_name}"? This will permanently remove all data.`)) return;
    try {
      await api.deleteTask(task.id);
      setTasks((prev) => prev.filter((t) => t.id !== task.id));
    } catch (e) {
      console.error("Failed to delete task:", e);
    }
  };

  useEffect(() => {
    fetchTasks();
    const interval = setInterval(fetchTasks, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">Navigation Tasks</h2>
          <p className="text-sm text-gray-400 mt-0.5">{tasks.length} task{tasks.length !== 1 ? "s" : ""} total</p>
        </div>
        <div className="flex gap-2">
          <Link
            to="/submit"
            className="flex items-center gap-2 px-4 py-2 text-sm bg-brand-600 text-white rounded-lg hover:bg-brand-700 transition shadow-sm"
          >
            + New Task
          </Link>
          <button
            onClick={fetchTasks}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-white border rounded-lg hover:bg-gray-50 transition"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* Loading skeletons */}
      {loading && tasks.length === 0 && (
        <div className="space-y-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="bg-white border rounded-xl p-4 animate-pulse flex items-center gap-4">
              <div className="w-20 h-6 bg-gray-100 rounded-full" />
              <div className="flex-1">
                <div className="h-4 w-48 bg-gray-100 rounded mb-1.5" />
                <div className="h-3 w-32 bg-gray-100 rounded" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty state */}
      {tasks.length === 0 && !loading && (
        <div className="text-center py-20">
          <Inbox size={48} className="mx-auto text-gray-300 mb-4" />
          <p className="text-gray-400 text-lg mb-2">No tasks yet</p>
          <Link to="/submit" className="text-brand-600 hover:underline text-sm">
            Submit your first navigation task →
          </Link>
        </div>
      )}

      {/* Task list */}
      <div className="space-y-3">
        {tasks.map((task) => {
          const st = statusConfig[task.status] || statusConfig.queued;
          const StatusIcon = st.icon;
          return (
            <div
              key={task.id}
              className="bg-white border rounded-xl p-4 flex items-center gap-4 hover:shadow-sm transition group"
            >
              {/* Status badge */}
              <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border ${st.color}`}>
                <StatusIcon size={12} className={task.status === "running" ? "animate-spin" : ""} />
                {st.label}
              </div>

              {/* Task info */}
              <div className="flex-1 min-w-0">
                <p className="font-medium text-gray-800 truncate text-sm">{task.display_name}</p>
                <p className="text-[11px] text-gray-400 mt-0.5 font-mono">
                  {task.id.slice(0, 13)} &middot; {task.goal_type} &middot;{" "}
                  {new Date(task.created_at).toLocaleString()}
                </p>
              </div>

              {/* Metrics (if done) */}
              {task.status === "done" && (
                <div className="hidden sm:flex gap-3 text-xs">
                  <span className={`px-2 py-1 rounded-md font-medium ${
                    (task.success ?? 0) > 0 ? "bg-green-50 text-green-600" : "bg-red-50 text-red-500"
                  }`}>
                    SR {((task.success ?? 0) * 100).toFixed(0)}%
                  </span>
                  <span className="px-2 py-1 rounded-md bg-blue-50 text-blue-600 font-medium">
                    SPL {(task.spl ?? 0).toFixed(3)}
                  </span>
                  <span className="px-2 py-1 rounded-md bg-gray-50 text-gray-500">
                    {task.total_steps} steps
                  </span>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-1 opacity-60 group-hover:opacity-100 transition">
                {task.status === "running" && (
                  <Link
                    to={`/monitor/${task.id}`}
                    className="p-2 rounded-lg hover:bg-blue-50 text-blue-500 transition"
                    title="Monitor"
                  >
                    <Play size={15} />
                  </Link>
                )}
                {task.status === "done" && (
                  <Link
                    to={`/results/${task.id}`}
                    className="p-2 rounded-lg hover:bg-green-50 text-green-600 transition"
                    title="View Results"
                  >
                    <Eye size={15} />
                  </Link>
                )}
                {(task.status === "queued" || task.status === "running") && (
                  <button
                    onClick={() => api.cancelTask(task.id).then(fetchTasks)}
                    className="p-2 rounded-lg hover:bg-red-50 text-red-400 transition"
                    title="Cancel"
                  >
                    <XCircle size={15} />
                  </button>
                )}
                {(task.status === "done" || task.status === "failed") && (
                  <button
                    onClick={() => handleDelete(task)}
                    className="p-2 rounded-lg hover:bg-red-50 text-red-400 transition"
                    title="Delete"
                  >
                    <Trash2 size={15} />
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
