import { NavLink, Route, Routes } from "react-router-dom";
import { api, usePolling } from "./api";
import type { Overview } from "./generated/schemas";
import Builder from "./screens/Builder";
import Data from "./screens/Data";
import FindingDetail from "./screens/FindingDetail";
import Findings from "./screens/Findings";
import Matrix from "./screens/Matrix";
import OverviewScreen from "./screens/Overview";
import RunDetail from "./screens/RunDetail";
import Runs from "./screens/Runs";
import Scaling from "./screens/Scaling";

function TopMeta() {
  const { data } = usePolling(api.overview, 15000);
  if (!data) return null;
  const o = data as Overview;
  return (
    <div className="top-meta">
      <div>
        generation <span className="mono">{o.active_generation}</span>
        {o.engine_commit && (
          <>
            {" · engine "}
            <span className="mono">{o.engine_commit.slice(0, 10)}</span>
          </>
        )}
        {o.lab_commit && (
          <>
            {" · lab "}
            <span className="mono">{o.lab_commit.slice(0, 10)}</span>
          </>
        )}
      </div>
    </div>
  );
}

const links = [
  { to: "/", label: "Overview" },
  { to: "/data", label: "Data" },
  { to: "/build", label: "Ablation builder" },
  { to: "/runs", label: "Runs" },
  { to: "/matrix", label: "Switch matrix" },
  { to: "/scaling", label: "Scaling" },
  { to: "/findings", label: "Findings" },
];

export default function App() {
  return (
    <>
      <header className="topbar">
        <span className="brand">
          CVS LAB <small>· research control room</small>
        </span>
        <nav>
          {links.map((link) => (
            <NavLink key={link.to} to={link.to} end={link.to === "/"}>
              {({ isActive }) => <span className={isActive ? "active" : ""}>{link.label}</span>}
            </NavLink>
          ))}
        </nav>
        <TopMeta />
      </header>
      <main>
        <Routes>
          <Route path="/" element={<OverviewScreen />} />
          <Route path="/data" element={<Data />} />
          <Route path="/build" element={<Builder />} />
          <Route path="/runs" element={<Runs />} />
          <Route path="/runs/:runId" element={<RunDetail />} />
          <Route path="/matrix" element={<Matrix />} />
          <Route path="/scaling" element={<Scaling />} />
          <Route path="/findings" element={<Findings />} />
          <Route path="/findings/:findingId" element={<FindingDetail />} />
        </Routes>
      </main>
    </>
  );
}
