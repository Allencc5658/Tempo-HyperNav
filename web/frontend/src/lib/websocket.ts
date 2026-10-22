/**
 * WebSocket client for real-time navigation monitoring.
 */

import type { GraphSnapshot } from "@/lib/api";

export interface TaskProgress {
  task_id: string;
  step: number;
  max_steps: number;
  rgb_frame?: string; // base64
  status_text: string;
  current_confidence?: number;
  nodes_count: number;
  edges_count: number;
  graph_snapshot?: GraphSnapshot;
}

type ProgressCallback = (msg: TaskProgress) => void;
type StatusCallback = (status: "connected" | "disconnected" | "error") => void;

export class NavigationMonitor {
  private ws: WebSocket | null = null;
  private taskId: string;
  private onProgress: ProgressCallback;
  private onStatus: StatusCallback;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private stopped = false;
  private reconnectAttempts = 0;
  private static MAX_RECONNECT_ATTEMPTS = 20;
  private static BASE_RECONNECT_MS = 2000;

  constructor(
    taskId: string,
    onProgress: ProgressCallback,
    onStatus: StatusCallback
  ) {
    this.taskId = taskId;
    this.onProgress = onProgress;
    this.onStatus = onStatus;
  }

  connect() {
    if (this.stopped) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/monitor/${this.taskId}`;

    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.onStatus("connected");
      this.reconnectAttempts = 0; // Reset on successful connection
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Ignore keepalive pings from server
        if (data.type === "ping" || data.ack) return;
        this.onProgress(data as TaskProgress);
      } catch {
        // Ignore non-JSON messages
      }
    };

    this.ws.onclose = () => {
      this.onStatus("disconnected");
      if (this.stopped) return;
      // Exponential backoff: 2s, 4s, 8s, ... capped at 30s
      if (this.reconnectAttempts < NavigationMonitor.MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          NavigationMonitor.BASE_RECONNECT_MS * Math.pow(1.5, this.reconnectAttempts),
          30000
        );
        this.reconnectAttempts++;
        this.reconnectTimer = setTimeout(() => this.connect(), delay);
      }
    };

    this.ws.onerror = () => {
      this.onStatus("error");
    };
  }

  disconnect() {
    this.stopped = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    if (this.ws) {
      this.ws.onclose = null; // Prevent reconnect on intentional close
      this.ws.close();
      this.ws = null;
    }
  }
}
