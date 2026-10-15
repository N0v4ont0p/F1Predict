// Team-lineage accent colours (approximate, for identification only — no team marks are used).
const LINEAGE: Record<string, string> = {
  red_bull: "oklch(0.55 0.17 265)",
  mercedes: "oklch(0.8 0.13 180)",
  ferrari: "oklch(0.62 0.23 27)",
  mclaren: "oklch(0.76 0.17 58)",
  aston_martin: "oklch(0.55 0.11 165)",
  alpine: "oklch(0.68 0.16 340)",
  williams: "oklch(0.62 0.17 255)",
  racing_bulls: "oklch(0.72 0.12 250)",
  haas: "oklch(0.82 0.01 265)",
  audi: "oklch(0.7 0.19 35)",
  sauber: "oklch(0.78 0.2 140)",
  cadillac: "oklch(0.86 0.05 90)",
};

export function teamColour(lineage: string): string {
  const hit = LINEAGE[lineage];
  if (hit) return hit;
  let h = 0;
  for (const ch of lineage) h = (h * 31 + ch.charCodeAt(0)) % 360;
  return `oklch(0.7 0.12 ${h})`;
}
