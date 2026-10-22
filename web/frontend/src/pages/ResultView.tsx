import { useEffect, useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import { Play, Pause, SkipForward, SkipBack } from "lucide-react";
import { api, Task, GraphSnapshot, SnapshotListItem } from "@/lib/api";
import SceneGraph from "@/components/SceneGraph";
import ConfidenceChart from "@/components/ConfidenceChart";

export default function ResultView() {
  const { taskId } = useParams<{ taskId: string }>();
  const [task, setTask] = useState<Task | null>(null);
  const [snapshots, setSnapshots] = useState<SnapshotListItem[]>([]);
  const [currentSnapshot, setCurrentSnapshot] = useState<GraphSnapshot | null>(null);
  const [currentIdx, setCurrentIdx] = useState(0);
  const [playing, setPlaying] = useState(false);

  // Load task and snapshot list
  useEffect(() => {
    if (!taskId) return;
    api.getTask(taskId).then(setTask);
    api.listSnapshots(taskId).then(setSnapshots);
  }, [taskId]);

  // Load current snapshot
  useEffect(() => {
    if (!taskId || snapshots.length === 0) return;
    const step = snapshots[currentIdx]?.step;
    if (step !== undefined) {
      api.getSnapshot(taskId, step).then(setCurrentSnapshot);
    }
  }, [taskId, snapshots, currentIdx]);

  // Auto-play
  useEffect(() => {
    if (!playing || snapshots.length === 0) return;
    const timer = setInterval(() => {
      setCurrentIdx((prev) => {
        if (prev >= snapshots.length - 1) {
          setPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 500);
    return () => clearInterval(timer);
  }, [playing, snapshots.length]);

  if (!taskId) {
    return (
      <div className="p-6 text-center py-20 text-gray-400">
        Select a completed task from the <Link to="/" className="text-brand-600 hover:underline">task list</Link>.
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800">
          Results {task && `— ${task.display_name}`}
        </h2>
        {task && (
          <div className="flex gap-6 text-sm mt-2">
            <span className={task.success ? "text-green-600 font-medium" : "text-red-500"}>
              {task.success ? "Success" : "Failed"}
            </span>
            <span className="text-gray-500">SPL: {task.spl?.toFixed(3) ?? "—"}</span>
            <span className="text-gray-500">{task.total_steps} steps</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Video panel */}
        <div className="xl:col-span-1 bg-white border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50 text-sm font-medium text-gray-600">
            Episode Video
          </div>
          <div className="p-2">
            {task?.video_path ? (
              <video
                controls
                className="w-full rounded-lg"
                src={`/static/runs/${taskId}/video.mp4`}
              />
            ) : (
              <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center text-gray-400 text-sm">
                No video available
              </div>
            )}
          </div>
        </div>

        {/* Scene graph panel */}
        <div className="xl:col-span-1 bg-white border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50 text-sm font-medium text-gray-600">
            Scene Graph (Step {currentSnapshot?.step ?? "—"})
          </div>
          <div className="h-[400px]">
            <SceneGraph snapshot={currentSnapshot} />
          </div>
        </div>

        {/* Confidence chart */}
        <div className="xl:col-span-1 bg-white border rounded-xl overflow-hidden">
          <div className="px-4 py-3 border-b bg-gray-50 text-sm font-medium text-gray-600">
            Node Confidence Over Time
          </div>
          <div className="h-[400px] p-2">
            <ConfidenceChart taskId={taskId} snapshots={snapshots} currentStep={currentSnapshot?.step} />
          </div>
        </div>
      </div>

      {/* Timeline control */}
      <div className="mt-6 bg-white border rounded-xl p-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setCurrentIdx(Math.max(0, currentIdx - 1))}
            className="p-2 rounded-lg hover:bg-gray-100 transition"
            disabled={currentIdx <= 0}
          >
            <SkipBack size={16} />
          </button>
          <button
            onClick={() => setPlaying(!playing)}
            className="p-2 rounded-lg hover:bg-gray-100 transition"
          >
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
          <button
            onClick={() => setCurrentIdx(Math.min(snapshots.length - 1, currentIdx + 1))}
            className="p-2 rounded-lg hover:bg-gray-100 transition"
            disabled={currentIdx >= snapshots.length - 1}
          >
            <SkipForward size={16} />
          </button>

          {/* Slider */}
          <input
            type="range"
            min={0}
            max={Math.max(snapshots.length - 1, 0)}
            value={currentIdx}
            onChange={(e) => setCurrentIdx(Number(e.target.value))}
            className="flex-1"
          />
          <span className="text-sm text-gray-500 w-24 text-right">
            {currentIdx + 1} / {snapshots.length}
          </span>
        </div>
      </div>
    </div>
  );
}
