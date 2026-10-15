"use client";

import { useEffect, useState } from "react";

export function Countdown({ to }: { to: string }) {
  const target = new Date(to).getTime();
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  if (now === null) return <span className="tabular font-mono">--d --h --m --s</span>;
  const s = Math.max(0, Math.floor((target - now) / 1000));
  const parts = [
    [Math.floor(s / 86400), "d"],
    [Math.floor((s % 86400) / 3600), "h"],
    [Math.floor((s % 3600) / 60), "m"],
    [s % 60, "s"],
  ] as const;
  return (
    <span className="tabular font-mono" aria-live="off">
      {parts.map(([v, u]) => (
        <span key={u} className="mr-2">
          {String(v).padStart(2, "0")}
          <span className="text-[var(--color-carbon-500)]">{u}</span>
        </span>
      ))}
    </span>
  );
}
