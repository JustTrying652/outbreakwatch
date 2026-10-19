import { useState } from "react";
import { api } from "./api/client";
import { useAsync } from "./api/useAsync";
import CaseChart from "./components/CaseChart";
import ResourcePanel from "./components/ResourcePanel";
import SummaryCard from "./components/SummaryCard";

const WINDOWS = [
  { label: "Last 90 days", value: 90 },
  { label: "Last year", value: 365 },
  { label: "All history", value: 1700 },
];

export default function App() {
  const meta = useAsync(() => api.facilities(), []);
  const [facility, setFacility] = useState(null);
  const [condition, setCondition] = useState(null);
  const [days, setDays] = useState(365);

  if (meta.loading) return <p className="muted pad">Connecting to the API…</p>;
  if (meta.error) {
    return (
      <p className="error pad">
        Cannot reach the API at http://127.0.0.1:8000. Is uvicorn running? ({meta.error.message})
      </p>
    );
  }

  const { facilities, conditions, resources } = meta.data;
  const f = facility ?? facilities[0];
  const c = condition ?? conditions[0];

  return (
    <div className="page">
      <header>
        <h1>OutbreakWatch</h1>
        <div className="controls">
          <label>
            Facility{" "}
            <select value={f} onChange={(e) => setFacility(e.target.value)}>
              {facilities.map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
          <label>
            Window{" "}
            <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {WINDOWS.map((w) => (
                <option key={w.value} value={w.value}>
                  {w.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      </header>

      <section>
        <h2>Disease surveillance</h2>
        <div className="tabs">
          {conditions.map((x) => (
            <button key={x} className={x === c ? "tab active" : "tab"} onClick={() => setCondition(x)}>
              {x}
            </button>
          ))}
        </div>
        <CaseChart facility={f} condition={c} days={days} />
      </section>

      <section>
        <h2>Resource readiness</h2>
        <ResourcePanel key={f} facility={f} resources={resources} days={days} />
      </section>

      <section>
        <h2>Situation report</h2>
        <SummaryCard key={f} facility={f} />
      </section>
    </div>
  );
}