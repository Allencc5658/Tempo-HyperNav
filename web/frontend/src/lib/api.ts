/**
 * Backend API client.
 * All requests go through Vite's proxy (see vite.config.ts),
 * so we use relative URLs here.
 */

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

// ── Types ────────────────────────────────────────────

export interface GoalOption {
  id: string;
  display_name: string;
  goal_type: "ins-image" | "text" | "objnav";
  goal_image?: string;
  goal_text?: string;
  episode_id: number;
  scene_id: string;
  preview_image?: string;
  difficulty: string;
  description?: string;
}

export interface Task {
  id: string;
  goal_id: string;
  goal_type: string;
  display_name: string;
  status: "queued" | "running" | "done" | "failed";
  created_at: string;
  started_at?: string;
  finished_at?: string;
  success?: number;
  spl?: number;
  total_steps?: number;
  video_path?: string;
  snapshot_dir?: string;
}

export interface GraphSnapshot {
  step: number;
  agent_pose: { x: number; y: number; heading: number };
  nodes: Array<{
    uid: string;
    caption: string;
    center: [number, number];
    confidence: number;
    n_obs: number;
    c_trend: number;
    is_in_fov: boolean;
  }>;
  edges: Array<{ src: string; dst: string; relation?: string; weight: number; n_simul: number }>;
  hyperedges: Array<{ name: string; members: string[]; confidence: number }>;
  prob_map: Record<string, number>;
  frontiers: Array<{ pos: [number, number]; score: number }>;
  delta_G?: {
    new_nodes: string[];
    disappeared: string[];
    confirmed: string[];
    declining: string[];
  };
}

export interface SnapshotListItem {
  step: number;
  filename: string;
}

// ── API calls ────────────────────────────────────────

export const api = {
  // Goals catalog
  getGoals: () => request<GoalOption[]>("/tasks/catalog"),

  // Tasks CRUD
  createTask: (body: {
    goal_id: string;
    disable_temporal?: boolean;
    disable_hypergraph?: boolean;
    disable_peek?: boolean;
  }) => request<Task>("/tasks/", { method: "POST", body: JSON.stringify(body) }),

  listTasks: (status?: string) =>
    request<Task[]>(`/tasks/${status ? `?status=${status}` : ""}`),

  getTask: (id: string) => request<Task>(`/tasks/${id}`),

  cancelTask: (id: string) =>
    request<{ message: string }>(`/tasks/${id}`, { method: "DELETE" }),

  deleteTask: (id: string) =>
    request<{ message: string }>(`/tasks/${id}/permanent`, { method: "DELETE" }),

  // Results
  getMetrics: (taskId: string) => request<any>(`/results/${taskId}/metrics`),

  listSnapshots: (taskId: string) =>
    request<SnapshotListItem[]>(`/results/${taskId}/snapshots`),

  getSnapshot: (taskId: string, step: number) =>
    request<GraphSnapshot>(`/results/${taskId}/snapshots/${step}`),

  compareTasks: (taskIds: string[]) =>
    request<any[]>("/results/compare", {
      method: "POST",
      body: JSON.stringify({ task_ids: taskIds }),
    }),
};
