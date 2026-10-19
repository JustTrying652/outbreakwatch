import {
  ResponsiveContainer, ComposedChart, Area, Line,
  XAxis, YAxis, Tooltip, Legend, CartesianGrid,
} from "recharts";
import { api } from "../api/client";
import { useAsync } from "../api/useAsync";

// Merge history, forecast and anomaly flags into one row per date.
function buildRows({ history, fc, flags }) {
  const rows = new Map();

  history.data.forEach((p) => rows.set(p.date, { date: p.date, actual: p.case_count }));

  // start the forecast line at the last actual point so the two lines join
  const last = history.data[history.data.length - 1];
  rows.get(last.date).mean = last.case_count;

  fc.forecast.forEach((p) =>
    rows.set(p.date, { date: p.date, mean: p.mean, band: [p.lower, p.upper] })
  );

  // a date can be flagged by both detectors; "alert" wins over "watch"
  const severity = new Map();
  flags.flags.forEach((f) => {
    if (severity.get(f.date) !== "alert") severity.set(f.date, f.severity);
  });
  severity.forEach((sev, date) => {
    const row = rows.get(date);
    if (row) row[sev] = row.actual;
  });

  return [...rows.values()];
}

function Reliability({ facility, condition }) {
  const { data, error, loading } = useAsync(
    () => api.backtest(facility, condition),
    [facility, condition]
  );

  if (loading) return <p className="muted">Running model back-test…</p>;
  if (error) return null;

  return (
    <p className="muted">
      Model back-test ({data.n_folds} folds, {data.horizon_days}-day horizon): average error{" "}
      {data.mae} cases/day; the 95% interval contained{" "}
      {(data.interval_coverage * 100).toFixed(0)}% of actual values.
    </p>
  );
}

export default function CaseChart({ facility, condition, days }) {
  const { data, error, loading } = useAsync(async () => {
    const [history, fc, flags] = await Promise.all([
      api.surveillance(facility, condition, days),
      api.forecast(facility, condition, 14),
      api.anomalies(facility, condition, days),
    ]);
    return { history, fc, flags };
  }, [facility, condition, days]);

  if (loading) {
    return (
      <p className="muted">
        Loading… (the first request for a series fits the model and can take a few seconds)
      </p>
    );
  }
  if (error) return <p className="error">Could not load chart: {error.message}</p>;

  const rows = buildRows(data);

  return (
    <div>
      <ResponsiveContainer width="100%" height={340}>
        <ComposedChart data={rows} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="date" minTickGap={48} tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip />
          <Legend />
          <Area
            dataKey="band"
            name="95% forecast interval"
            stroke="none"
            fill="#93c5fd"
            fillOpacity={0.45}
          />
          <Line dataKey="actual" name="Actual cases" stroke="#1f2937" strokeWidth={1.5} dot={false} />
          <Line
            dataKey="mean"
            name="Forecast"
            stroke="#2563eb"
            strokeWidth={2}
            strokeDasharray="5 4"
            dot={false}
          />
          <Line
            dataKey="watch"
            name="Watch"
            stroke="#f59e0b"
            strokeWidth={0}
            dot={{ r: 4, fill: "#f59e0b" }}
            activeDot={false}
            legendType="circle"
          />
          <Line
            dataKey="alert"
            name="Alert"
            stroke="#dc2626"
            strokeWidth={0}
            dot={{ r: 4, fill: "#dc2626" }}
            activeDot={false}
            legendType="circle"
          />
        </ComposedChart>
      </ResponsiveContainer>

      <p className="muted">
        {data.flags.count} anomaly flags in this window. Forecast covers the 14 days after{" "}
        {data.history.data[data.history.data.length - 1].date}.
      </p>
      <Reliability facility={facility} condition={condition} />
    </div>
  );
}