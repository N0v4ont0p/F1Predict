// Pure SVG charts (server-rendered, accessible, no chart library).

export function Reliability({
  bins,
  title,
}: {
  bins: { predicted: number; observed: number; count: number }[];
  title: string;
}) {
  const S = 220;
  const pad = 28;
  const x = (v: number) => pad + v * (S - pad * 1.4);
  const y = (v: number) => S - pad - v * (S - pad * 1.4);
  const maxCount = Math.max(...bins.map((b) => b.count), 1);
  const path = bins.map((b, i) => `${i ? "L" : "M"}${x(b.predicted)},${y(b.observed)}`).join(" ");
  return (
    <figure className="glass p-4">
      <figcaption className="eyebrow mb-2">{title}</figcaption>
      <svg
        viewBox={`0 0 ${S} ${S}`}
        className="w-full"
        role="img"
        aria-label={`Reliability diagram: ${title}`}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={y(0)} y2={y(1)} stroke="white" strokeOpacity="0.05" />
            <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="white" strokeOpacity="0.05" />
            <text
              x={x(t)}
              y={S - 8}
              fontSize="8"
              fill="white"
              fillOpacity="0.4"
              textAnchor="middle"
            >
              {t}
            </text>
          </g>
        ))}
        <line
          x1={x(0)}
          y1={y(0)}
          x2={x(1)}
          y2={y(1)}
          stroke="white"
          strokeOpacity="0.25"
          strokeDasharray="3 3"
        />
        <path d={path} fill="none" stroke="var(--color-signal)" strokeWidth="1.8" />
        {bins.map((b, i) => (
          <circle
            key={i}
            cx={x(b.predicted)}
            cy={y(b.observed)}
            r={2 + 5 * Math.sqrt(b.count / maxCount)}
            fill="var(--color-signal)"
            fillOpacity="0.35"
            stroke="var(--color-signal)"
          >
            <title>{`predicted ${(b.predicted * 100).toFixed(0)}% → observed ${(b.observed * 100).toFixed(0)}% (${b.count})`}</title>
          </circle>
        ))}
      </svg>
    </figure>
  );
}

export function StageCurve({ temps, label }: { temps: number[]; label: string }) {
  const W = 420;
  const H = 160;
  const pad = 24;
  const max = Math.max(...temps, 1);
  const x = (i: number) => pad + (i / Math.max(temps.length - 1, 1)) * (W - pad * 2);
  const y = (v: number) => H - pad - (v / max) * (H - pad * 2);
  const line = temps.map((t, i) => `${i ? "L" : "M"}${x(i)},${y(t)}`).join(" ");
  const area = `${line} L${x(temps.length - 1)},${y(0)} L${x(0)},${y(0)} Z`;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={label}>
      <defs>
        <linearGradient id="sc" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="var(--color-signal)" stopOpacity="0.45" />
          <stop offset="1" stopColor="var(--color-signal)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#sc)" />
      <path d={line} fill="none" stroke="var(--color-signal)" strokeWidth="2" />
      {temps.map((_, i) =>
        i % 3 === 0 ? (
          <text
            key={i}
            x={x(i)}
            y={H - 6}
            fontSize="9"
            fill="white"
            fillOpacity="0.45"
            textAnchor="middle"
          >
            P{i + 1}
          </text>
        ) : null,
      )}
    </svg>
  );
}

export function Sparkband({
  series,
  colour,
  min,
  max,
}: {
  series: { mu: number; sd: number }[];
  colour: string;
  min: number;
  max: number;
}) {
  const W = 160;
  const H = 36;
  if (series.length < 2)
    return <svg viewBox={`0 0 ${W} ${H}`} className="h-9 w-40" aria-hidden="true" />;
  const x = (i: number) => (i / (series.length - 1)) * W;
  const y = (v: number) => H - ((v - min) / Math.max(max - min, 1e-6)) * H;
  const upper = series.map((s, i) => `${i ? "L" : "M"}${x(i)},${y(s.mu + s.sd)}`).join(" ");
  const lower = [...series]
    .map((s, i) => [i, s] as const)
    .reverse()
    .map(([i, s]) => `L${x(i)},${y(s.mu - s.sd)}`)
    .join(" ");
  const mid = series.map((s, i) => `${i ? "L" : "M"}${x(i)},${y(s.mu)}`).join(" ");
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-9 w-40"
      aria-hidden="true"
      preserveAspectRatio="none"
    >
      <path d={`${upper} ${lower} Z`} fill={colour} fillOpacity="0.15" />
      <path d={mid} fill="none" stroke={colour} strokeWidth="1.6" />
    </svg>
  );
}
