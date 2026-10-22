import { useEffect, useState, useRef } from "react";
import { useParams, Link } from "react-router-dom";
import { Wifi, WifiOff, ArrowRight, AlertCircle, Activity } from "lucide-react";
import { api, Task, GraphSnapshot } from "@/lib/api";
import { NavigationMonitor, TaskProgress } from "@/lib/websocket";
import SceneGraph from "@/components/SceneGraph";

export default function Monitor() {
  const { taskId } = useParams<{ taskId: string }>();
  const [task, setTask] = useState<Task | null>(null);
  const [progress, setProgress] = useState<TaskProgress | null>(null);
  const [wsStatus, setWsStatus] = useState<"connected" | "disconnected" | "error">("disconnected");
  const [liveSnapshot, setLiveSnapshot] = useState<GraphSnapshot | null>(null);
  const monitorRef = useRef<NavigationMonitor | null>(null);

  // Fetch task info — poll every 5s to detect completion
  useEffect(() => {
    if (!taskId) return;
    api.getTask(taskId).then(setTask);
    const interval = setInterval(() => {
      api.getTask(taskId).then((t) => {
        setTask(t);
        if (t.status === "done" || t.status === "failed") {
          clearInterval(interval);
        }
      }).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, [taskId]);

  // WebSocket connection
  useEffect(() => {
    if (!taskId) return;
    const handleProgress = (msg: TaskProgress) => {
      setProgress(msg);
      if (msg.graph_snapshot) {
        setLiveSnapshot(msg.graph_snapshot);
      }
    };
    const monitor = new NavigationMonitor(taskId, handleProgress, setWsStatus);
    monitor.connect();
    monitorRef.current = monitor;
    return () => monitor.disconnect();
  }, [taskId]);

  // Stop reconnecting once task is done/failed
  useEffect(() => {
    if (task && (task.status === "done" || task.status === "failed")) {
      monitorRef.current?.disconnect();
    }
  }, [task?.status]);

  if (!taskId) {
    return (
      <div className="p-6 max-w-4xl mx-auto text-center py-24">
        <Activity size={48} className="mx-auto text-gray-300 mb-4" />
        <p className="text-gray-400 text-lg">No task selected</p>
        <Link to="/" className="text-brand-600 hover:underline mt-2 inline-block text-sm">
          Select a running task from the task list →
        </Link>
      </div>
    );
  }

  const progressPct = progress ? Math.round((progress.step / Math.max(progress.max_steps, 1)) * 100) : 0;
  const isFinished = task?.status === "done" || task?.status === "failed";

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">
            Live Monitor {task && <span className="text-gray-400 font-normal">— {task.display_name}</span>}
          </h2>
          <p className="text-xs text-gray-400 mt-0.5 font-mono">ID: {taskId}</p>
        </div>
        <div className={`flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border ${
          wsStatus === "connected"
            ? "border-green-200 bg-green-50 text-green-700"
            : wsStatus === "error"
            ? "border-red-200 bg-red-50 text-red-600"
            : "border-gray-200 bg-gray-50 text-gray-500"
        }`}>
          {wsStatus === "connected" ? (
            <Wifi size={13} />
          ) : (
            <WifiOff size={13} />
          )}
          {wsStatus === "connected" ? "Live" : wsStatus === "error" ? "Error" : "Disconnected"}
        </div>
      </div>

      {/* Task completed banner */}
      {task?.status === "done" && (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-xl mb-5 flex items-center justify-between">
          <span className="text-sm font-medium">Navigation completed successfully!</span>
          <Link to={`/results/${taskId}`} className="flex items-center gap-1 text-sm font-medium hover:underline">
            View Results <ArrowRight size={14} />
          </Link>
        </div>
      )}
      {task?.status === "failed" && (
        <div className="bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-xl mb-5 flex items-center gap-2">
          <AlertCircle size={16} />
          <span className="text-sm font-medium">Navigation failed. Check logs for details.</span>
        </div>
      )}

      {/* Progress bar */}
      <div className="bg-white border rounded-xl p-4 mb-5">
        <div className="flex items-center justify-between text-sm mb-2">
          <span className="text-gray-600 font-medium">
            Step {progress?.step ?? 0} <span className="text-gray-400 font-normal">/ {progress?.max_steps ?? "?"}</span>
          </span>
          <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${
            isFinished ? "bg-gray-100 text-gray-500" : "bg-brand-50 text-brand-600"
          }`}>{progressPct}%</span>
        </div>
        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ease-out ${
              isFinished ? "bg-gray-400" : "bg-gradient-to-r from-brand-500 to-brand-600"
            }`}
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Main panels — 3 columns */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* RGB frame */}
        <div className="bg-white border rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b bg-gray-50 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Agent View
          </div>
          <div className="aspect-video bg-gray-900 flex items-center justify-center relative">
            {progress?.rgb_frame ? (
              <img
                src={`data:image/jpeg;base64,${progress.rgb_frame}`}
                alt="Agent RGB"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="flex flex-col items-center gap-2">
                <div className="w-8 h-8 border-2 border-gray-600 border-t-gray-400 rounded-full animate-spin" />
                <span className="text-gray-500 text-xs">Waiting for frames...</span>
              </div>
            )}
            {progress && (
              <div className="absolute bottom-2 right-2 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded font-mono">
                step {progress.step}
              </div>
            )}
          </div>
        </div>

        {/* Live Scene Graph */}
        <div className="bg-white border rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b bg-gray-50 text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center justify-between">
            <span>Live Scene Graph</span>
            {liveSnapshot && (
              <span className="text-[10px] font-mono text-gray-400 normal-case">
                {liveSnapshot.nodes.length}N / {liveSnapshot.edges.length}E
              </span>
            )}
          </div>
          <div className="h-[300px] lg:h-[280px]">
            {liveSnapshot ? (
              <SceneGraph snapshot={liveSnapshot} />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-400 text-xs">
                Graph will appear after first snapshot...
              </div>
            )}
          </div>
        </div>

        {/* Live stats */}
        <div className="bg-white border rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b bg-gray-50 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Statistics
          </div>
          <div className="p-4 space-y-3">
            <StatItem label="Status" value={progress?.status_text || task?.status || "—"} highlight />
            <div className="h-px bg-gray-100" />
            <StatItem label="Graph Nodes" value={String(progress?.nodes_count ?? 0)} />
            <StatItem label="Graph Edges" value={String(progress?.edges_count ?? 0)} />
            <StatItem
              label="Max Confidence"
              value={progress?.current_confidence?.toFixed(3) ?? "—"}
            />
            <div className="h-px bg-gray-100" />
            <StatItem label="Goal Type" value={task?.goal_type ?? "—"} />
            <StatItem label="Task Status" value={task?.status ?? "—"} />
            {task?.status === "done" && (
              <>
                <div className="h-px bg-gray-100" />
                <StatItem label="Success" value={task.success ? "Yes" : "No"} />
                <StatItem label="SPL" value={task.spl?.toFixed(3) ?? "—"} />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function StatItem({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-xs font-medium ${highlight ? "text-brand-600" : "text-gray-800"}`}>{value}</span>
    </div>
  );
}
