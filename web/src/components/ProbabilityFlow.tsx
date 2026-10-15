"use client";

import { useEffect, useRef } from "react";

export interface FlowLane {
  code: string;
  p: number;
  colour: string;
}

/**
 * The hero visual: probability as a river. Each contender is a lane whose width and particle
 * density are its win probability; particles stream towards the flag and the strongest lanes
 * glow. Canvas 2D, DPR-aware, paused off-screen, static when reduced motion is requested.
 */
/** Lane centres (same geometry as the canvas), for lanes wide enough to carry a label. */
function labelPositions(lanes: FlowLane[], height: number) {
  const total = lanes.reduce((s, l) => s + l.p, 0) || 1;
  const usable = height - 24;
  let y = 12;
  const out: { code: string; y: number }[] = [];
  for (const l of lanes) {
    const h = Math.max(3, (l.p / total) * usable);
    if (h >= 14) out.push({ code: l.code, y: y + h / 2 });
    y += h;
  }
  return out;
}

export function ProbabilityFlow({ lanes, height = 320 }: { lanes: FlowLane[]; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const total = lanes.reduce((s, l) => s + l.p, 0) || 1;
    let width = 0;
    let dpr = 1;
    type P = { lane: number; x: number; y: number; v: number; r: number; a: number };
    let particles: P[] = [];
    const bands: { y0: number; y1: number }[] = [];

    // Canvas colour strings can't take an alpha suffix on oklch(), so resolve every lane
    // colour to RGB once by painting it into a 1x1 canvas.
    const probe = document.createElement("canvas").getContext("2d", { willReadFrequently: true });
    const resolveColour = (c: string) => {
      if (!probe) return "rgb(200,200,200)";
      probe.clearRect(0, 0, 1, 1);
      probe.fillStyle = c;
      probe.fillRect(0, 0, 1, 1);
      const [r, g, b] = probe.getImageData(0, 0, 1, 1).data;
      return `rgb(${r},${g},${b})`;
    };
    const colours = lanes.map((l) => resolveColour(l.colour));

    const layout = () => {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = canvas.clientWidth;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      bands.length = 0;
      let y = 12;
      const usable = height - 24;
      for (const l of lanes) {
        const h = Math.max(3, (l.p / total) * usable);
        bands.push({ y0: y, y1: y + h });
        y += h;
      }
      const count = Math.min(1400, Math.floor(width * 1.4));
      particles = Array.from({ length: count }, () => spawn(Math.random() * width));
    };

    const pickLane = () => {
      let r = Math.random() * total;
      for (let i = 0; i < lanes.length; i++) {
        r -= lanes[i]?.p ?? 0;
        if (r <= 0) return i;
      }
      return lanes.length - 1;
    };

    const spawn = (x: number): P => {
      const lane = pickLane();
      const b = bands[lane] ?? { y0: 0, y1: height };
      const share = (lanes[lane]?.p ?? 0) / total;
      return {
        lane,
        x,
        y: b.y0 + Math.random() * (b.y1 - b.y0),
        v: 0.6 + Math.random() * 1.4 + share * 3,
        r: 0.6 + Math.random() * 1.6,
        a: 0.25 + Math.random() * 0.75,
      };
    };

    const draw = (t: number, fresh = false) => {
      // fade the previous frame instead of clearing it: particles leave light trails
      ctx.globalCompositeOperation = "destination-out";
      ctx.fillStyle = fresh || reduced ? "rgba(0,0,0,1)" : "rgba(0,0,0,0.22)";
      ctx.fillRect(0, 0, width, height);
      ctx.globalCompositeOperation = "source-over";
      // lanes: soft glowing bands
      bands.forEach((b, i) => {
        const g = ctx.createLinearGradient(0, 0, width, 0);
        g.addColorStop(0, "rgba(0,0,0,0)");
        g.addColorStop(
          1,
          colours[i]?.replace("rgb(", "rgba(").replace(")", ",0.10)") ?? "transparent",
        );
        ctx.fillStyle = g;
        ctx.fillRect(0, b.y0, width, b.y1 - b.y0);
      });
      ctx.globalCompositeOperation = "lighter";
      for (const p of particles) {
        if (!reduced) {
          p.x += p.v;
          p.y += Math.sin((p.x + t * 0.05) * 0.01 + p.lane) * 0.15;
          if (p.x > width + 4) Object.assign(p, spawn(-4));
        }
        const c = colours[p.lane] ?? "rgb(255,255,255)";
        const fade = Math.min(1, p.x / (width * 0.25));
        ctx.fillStyle = c.replace("rgb(", "rgba(").replace(")", `,${(p.a * fade).toFixed(3)})`);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";
    };

    let raf = 0;
    let visible = true;
    const loop = (t: number) => {
      if (visible) draw(t);
      if (!reduced) raf = requestAnimationFrame(loop);
    };
    layout();
    raf = requestAnimationFrame(loop);
    const ro = new ResizeObserver(layout);
    ro.observe(canvas);
    const io = new IntersectionObserver(([e]) => {
      visible = Boolean(e?.isIntersecting);
    });
    io.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, [lanes, height]);

  return (
    <div className="relative">
      <canvas
        ref={ref}
        style={{ width: "100%", height }}
        className="block"
        role="img"
        aria-label={`Win probability: ${lanes
          .slice(0, 5)
          .map((l) => `${l.code} ${Math.round(l.p * 100)}%`)
          .join(", ")}`}
      />
      {labelPositions(lanes, height).map(({ code, y }) => (
        <span
          key={code}
          className="pointer-events-none absolute right-2 -translate-y-1/2 font-mono text-[0.65rem] tracking-widest text-white/60"
          style={{ top: y }}
        >
          {code}
        </span>
      ))}
    </div>
  );
}
