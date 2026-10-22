import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Image, FileText, Send, ToggleLeft, ToggleRight, Loader2, AlertTriangle } from "lucide-react";
import { api, GoalOption } from "@/lib/api";

const difficultyColors: Record<string, string> = {
  easy: "bg-green-100 text-green-700",
  medium: "bg-yellow-100 text-yellow-700",
  hard: "bg-red-100 text-red-600",
};

export default function TaskSubmit() {
  const navigate = useNavigate();
  const [goals, setGoals] = useState<GoalOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedGoalId, setSelectedGoalId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Ablation toggles
  const [disableTemporal, setDisableTemporal] = useState(false);
  const [disableHypergraph, setDisableHypergraph] = useState(false);
  const [disablePeek, setDisablePeek] = useState(false);

  useEffect(() => {
    api.getGoals()
      .then(setGoals)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSubmit = async () => {
    if (!selectedGoalId) return;
    setSubmitting(true);
    setError(null);
    try {
      const task = await api.createTask({
        goal_id: selectedGoalId,
        disable_temporal: disableTemporal,
        disable_hypergraph: disableHypergraph,
        disable_peek: disablePeek,
      });
      navigate(`/monitor/${task.id}`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  };

  const selectedGoal = goals.find((g) => g.id === selectedGoalId);

  return (
    <div className="p-6 max-w-5xl mx-auto">
      <h2 className="text-2xl font-bold text-gray-800 mb-1">Submit Navigation Task</h2>
      <p className="text-sm text-gray-400 mb-6">Select a goal, configure ablation settings, and start navigation.</p>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl mb-5 flex items-center gap-2 text-sm">
          <AlertTriangle size={16} />
          {error}
        </div>
      )}

      {/* Goal selection cards */}
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Select a Goal</h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {loading && [0, 1, 2].map((i) => (
          <div key={i} className="border rounded-xl p-4 animate-pulse">
            <div className="w-full h-32 rounded-lg bg-gray-100 mb-3" />
            <div className="h-4 w-3/4 bg-gray-100 rounded mb-2" />
            <div className="h-3 w-1/2 bg-gray-100 rounded" />
          </div>
        ))}
        {!loading && goals.map((goal) => (
          <button
            key={goal.id}
            onClick={() => setSelectedGoalId(goal.id)}
            className={`text-left border rounded-xl p-4 transition-all hover:shadow-md ${
              selectedGoalId === goal.id
                ? "border-brand-500 ring-2 ring-brand-500/30 bg-brand-50"
                : "border-gray-200 bg-white hover:border-gray-300"
            }`}
          >
            {/* Goal preview card */}
            <div className={`w-full h-32 rounded-lg mb-3 flex flex-col items-center justify-center relative overflow-hidden ${
              goal.goal_type === "ins-image"
                ? "bg-gradient-to-br from-indigo-500/10 via-purple-500/10 to-pink-500/10"
                : "bg-gradient-to-br from-emerald-500/10 via-teal-500/10 to-cyan-500/10"
            }`}>
              {goal.goal_type === "ins-image" ? (
                <>
                  <Image size={24} className="text-indigo-400 mb-1" />
                  <span className="text-[10px] font-medium text-indigo-400/80 tracking-wide uppercase">Instance Image</span>
                </>
              ) : (
                <>
                  <FileText size={24} className="text-teal-500 mb-1" />
                  <span className="text-[10px] font-medium text-teal-500/80 tracking-wide uppercase">Text Description</span>
                </>
              )}
              {goal.description && (
                <p className="text-[9px] text-gray-500 mt-1.5 px-3 text-center line-clamp-2 leading-relaxed">
                  {goal.description}
                </p>
              )}
              <span className="absolute top-2 right-2 text-[10px] px-1.5 py-0.5 rounded bg-white/70 text-gray-500 font-mono">
                ep:{goal.episode_id}
              </span>
            </div>

            <p className="font-medium text-gray-800 text-sm">{goal.display_name}</p>
            <div className="flex items-center gap-1.5 mt-1.5">
              <span className="text-[10px] px-2 py-0.5 bg-gray-100 rounded-full text-gray-500 font-medium">
                {goal.goal_type}
              </span>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                difficultyColors[goal.difficulty] || "bg-gray-100 text-gray-500"
              }`}>
                {goal.difficulty}
              </span>
            </div>
          </button>
        ))}
        {!loading && goals.length === 0 && (
          <p className="text-gray-400 col-span-3 text-center py-12 text-sm">
            No goals available. Add entries to goal_catalog.json.
          </p>
        )}
      </div>

      {/* Ablation toggles */}
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Ablation Settings</h3>
      <div className="bg-white border rounded-xl p-4 mb-8 space-y-1">
        {[
          { label: "Disable Temporal Scene Graph", desc: "Turn off time-decay confidence & ghost pruning", value: disableTemporal, set: setDisableTemporal },
          { label: "Disable Hypergraph Matching", desc: "Fall back to overlap-based matching", value: disableHypergraph, set: setDisableHypergraph },
          { label: "Disable Doorway Peek", desc: "No VLM room-type check for frontier scoring", value: disablePeek, set: setDisablePeek },
        ].map((item) => (
          <button
            key={item.label}
            onClick={() => item.set(!item.value)}
            className="flex items-center justify-between w-full px-3 py-2.5 rounded-lg hover:bg-gray-50 transition"
          >
            <div className="text-left">
              <span className="text-sm text-gray-700 block">{item.label}</span>
              <span className="text-[11px] text-gray-400">{item.desc}</span>
            </div>
            {item.value ? (
              <ToggleRight size={26} className="text-red-500 shrink-0" />
            ) : (
              <ToggleLeft size={26} className="text-gray-300 shrink-0" />
            )}
          </button>
        ))}
      </div>

      {/* Submit */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleSubmit}
          disabled={!selectedGoalId || submitting}
          className="flex items-center gap-2 px-6 py-3 bg-brand-600 text-white rounded-xl font-medium
                     hover:bg-brand-700 disabled:opacity-50 disabled:cursor-not-allowed transition shadow-sm"
        >
          {submitting ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          {submitting ? "Submitting..." : "Start Navigation"}
        </button>
        {selectedGoal && (
          <span className="text-sm text-gray-500">
            Selected: <strong>{selectedGoal.display_name}</strong>
            <span className="text-gray-400 ml-1">(episode {selectedGoal.episode_id})</span>
          </span>
        )}
      </div>
    </div>
  );
}
