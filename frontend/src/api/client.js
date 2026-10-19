const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function get(path, params = {}) {
  const url = new URL(BASE + path);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== "") {
      url.searchParams.set(key, value);
    }
  });

  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

const enc = encodeURIComponent;

export const api = {
  facilities: () => get("/facilities"),
  surveillance: (facility, condition, days) =>
    get(`/surveillance/${enc(facility)}/${enc(condition)}`, { days }),
  forecast: (facility, condition, steps = 14) =>
    get(`/forecast/${enc(facility)}/${enc(condition)}`, { steps }),
  anomalies: (facility, condition, since_days) =>
    get(`/anomalies/${enc(facility)}/${enc(condition)}`, { since_days }),
  backtest: (facility, condition) =>
    get(`/backtest/${enc(facility)}/${enc(condition)}`),
  resource: (facility, resource, days) =>
    get(`/resources/${enc(facility)}/${enc(resource)}`, { days }),
  stockout: (facility, resource, horizon = 30, as_of = null) =>
    get(`/stockout/${enc(facility)}/${enc(resource)}`, { horizon, as_of }),
  summary: (facility) => get(`/summary/${enc(facility)}`),
};