import { Routes, Route, NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Crosshair,
  Play,
  Eye,
  GitCompare,
} from "lucide-react";

import TaskList from "@/pages/TaskList";
import TaskSubmit from "@/pages/TaskSubmit";
import Monitor from "@/pages/Monitor";
import ResultView from "@/pages/ResultView";
import Compare from "@/pages/Compare";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Tasks" },
  { to: "/submit", icon: Crosshair, label: "New Task" },
  { to: "/monitor", icon: Play, label: "Monitor" },
  { to: "/results", icon: Eye, label: "Results" },
  { to: "/compare", icon: GitCompare, label: "Compare" },
];

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* ── Sidebar ── */}
      <aside className="w-56 shrink-0 bg-brand-900 text-white flex flex-col">
        <div className="px-5 py-6 border-b border-white/10">
          <h1 className="text-lg font-bold tracking-wide">Tempo-HyperNav</h1>
          <p className="text-xs text-blue-200 mt-0.5">Navigation Visualization</p>
        </div>
        <nav className="flex-1 py-4 space-y-1 px-3">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? "bg-white/15 text-white font-medium"
                    : "text-blue-100 hover:bg-white/10"
                }`
              }
            >
              <item.icon size={18} />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-white/10 text-xs text-blue-200">
          Tempo-HyperNav v0.1.0
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="flex-1 overflow-y-auto bg-gray-50">
        <Routes>
          <Route path="/" element={<TaskList />} />
          <Route path="/submit" element={<TaskSubmit />} />
          <Route path="/monitor" element={<Monitor />} />
          <Route path="/monitor/:taskId" element={<Monitor />} />
          <Route path="/results" element={<ResultView />} />
          <Route path="/results/:taskId" element={<ResultView />} />
          <Route path="/compare" element={<Compare />} />
        </Routes>
      </main>
    </div>
  );
}
