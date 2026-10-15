export function pct(p: number, digits = 1): string {
  if (p >= 0.9995) return "100%";
  if (p > 0 && p < 0.001) return "<0.1%";
  return `${(p * 100).toFixed(digits)}%`;
}

export function odds(p: number): string {
  // fair decimal odds, for intuition only — this site gives no betting advice
  return p > 0 ? (1 / p).toFixed(p > 0.1 ? 1 : 0) : "—";
}

export function dateLabel(iso: string): string {
  return new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function shortDate(iso: string): string {
  return new Date(`${iso}T12:00:00Z`).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
  });
}

/** Sequential heat colour for a probability (deep blue-violet → cyan → amber → white). */
export function heat(p: number): string {
  const t = Math.max(0, Math.min(1, Math.sqrt(p)));
  const l = 0.18 + 0.78 * t;
  const c = 0.04 + 0.14 * Math.sin(Math.PI * Math.min(1, t * 1.1));
  const h = 280 - 210 * t;
  return `oklch(${l.toFixed(3)} ${c.toFixed(3)} ${h.toFixed(1)})`;
}
