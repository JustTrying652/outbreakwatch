import { useState } from "react";
import { api } from "../api/client";

export default function SummaryCard({ facility }) {
  const [state, setState] = useState({ status: "idle" });

  async function generate() {
    setState({ status: "loading" });
    try {
      const data = await api.summary(facility);
      setState({ status: "done", report: data.report });
    } catch (e) {
      setState({ status: "error", message: e.message });
    }
  }

  return (
    <div>
      <button onClick={generate} disabled={state.status === "loading"}>
        {state.status === "loading" ? "Generating…" : "Generate situation report"}
      </button>
      {state.status === "loading" && (
        <p className="muted">The first report fits several models and can take a minute. Later ones are cached.</p>
      )}
      {state.status === "error" && <p className="error">{state.message}</p>}
      {state.status === "done" && <pre className="report">{state.report}</pre>}
    </div>
  );
}