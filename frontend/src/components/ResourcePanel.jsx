import { useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid, ReferenceLine,
} from "recharts";
import { api } from "../api/client";
import { useAsync } from "../api/useAsync";

export default function ResourcePanel({ facility, resources, days }) {
  const [resource, setResource] = useState(resources[0]);
  const [asOf, setAsOf] = useState("");

  const history = useAsync(
    () => api.resource(facility, resource, days),
    [facility, resource, days]
  );
  const stockout = useAsync(
    () => api.stockout(facility, resource, 30, asOf || null),
    [facility, resource, asOf]
  );

  const so = stockout.data;
  const d = so?.days_until_stockout;
  const atRisk = d && d.expected !== null;

  return (
    <div>
      <div className="controls">
        <label>
          Resource{" "}
          <select value={resource} onChange={(e) => setResource(e.target.value)}>
            {resources.map((r) => (
              <option key={r}>{r}</option>
            ))}
          </select>
        </label>

        <label>
          Replay as of{" "}
          <input
            type="date"
            value={asOf}
            min="2022-06-01"
            max="2026-08-31"
            onChange={(e) => setAsOf(e.target.value)}
          />
        </label>
        {asOf && <button onClick={() => setAsOf("")}>Back to latest</button>}
      </div>
      <p className="muted">
        Tip: Kisumu Referral Hospital / ORS Sachets, replayed as of 2023-04-18, shows the
        stock-out warning during the April 2023 outbreak.
      </p>

      {history.loading && <p className="muted">Loading stock history…</p>}
      {history.error && <p className="error">{history.error.message}</p>}
      {history.data && (
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={history.data.data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" minTickGap={48} tick={{ fontSize: 12 }} />
            <YAxis tick={{ fontSize: 12 }} />
            <Tooltip />
            <Line dataKey="level" name="Stock level" stroke="#059669" strokeWidth={2} dot={false} />
            {so && (
              <ReferenceLine
                y={so.safety_threshold}
                stroke="#dc2626"
                strokeDasharray="5 4"
                ifOverflow="extendDomain"
                label={{ value: "Safety threshold", fontSize: 12, fill: "#dc2626", position: "insideTopRight" }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      )}

      {stockout.loading && <p className="muted">Predicting stock-out…</p>}
      {stockout.error && <p className="error">{stockout.error.message}</p>}
      {so && (
        <div className={`card ${atRisk ? "risk" : "ok"}`}>
          <strong>{atRisk ? "Stock-out risk" : "No stock-out projected"}</strong>
          <p>{so.summary}</p>
          {atRisk && (
            <p className="muted">
              Best case {d.best_case ?? "—"} days · Expected {d.expected} days · Worst case{" "}
              {d.worst_case ?? "—"} days
            </p>
          )}
        </div>
      )}
    </div>
  );
}