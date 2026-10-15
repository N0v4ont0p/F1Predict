"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { teamColour } from "@/lib/teams";
import type { CircuitMart, TheatreMart, TheatreRace } from "@/lib/types";

const SPEEDS = [10, 30, 60, 120] as const;
const TRACK_SIZE = 100; // scene units across the circuit's longest side

/** Where every car is at race time `t` (seconds): laps completed + fraction of the next. */
function progressAt(race: TheatreRace, t: number, n: number): (number | null)[] {
  const out: (number | null)[] = [];
  for (let i = 0; i < n; i++) {
    let prev = 0;
    let done: number | null = null;
    for (let lap = 0; lap < race.t.length; lap++) {
      const cur = race.t[lap]?.[i];
      if (cur == null) {
        // retired during this lap: stays where it stopped
        done = lap === 0 ? null : lap;
        break;
      }
      if (t < cur) {
        done = lap + (t - prev) / Math.max(cur - prev, 1e-6);
        break;
      }
      prev = cur;
      done = lap + 1;
    }
    out.push(done);
  }
  return out;
}

function retiredAt(race: TheatreRace, i: number): number | null {
  for (let lap = 0; lap < race.t.length; lap++) if (race.t[lap]?.[i] == null) return lap;
  return null;
}

/** Position traces: running order after every lap of one simulated race. */
function Traces({ race, cars }: { race: TheatreRace; cars: TheatreMart["cars"] }) {
  const laps = race.t.length;
  const n = cars.length;
  const W = 900;
  const H = 260;
  const x = (lap: number) => (lap / Math.max(laps, 1)) * W;
  const y = (pos: number) => ((pos - 0.5) / n) * H;
  const lines = (() => {
    const order: number[][] = [];
    order.push(race.grid.map((g) => g + 1));
    for (let lap = 0; lap < laps; lap++) {
      const row = race.t[lap] ?? [];
      const idx = row
        .map((v, i) => ({ v, i }))
        .sort((a, b) => (a.v ?? Number.POSITIVE_INFINITY) - (b.v ?? Number.POSITIVE_INFINITY));
      const pos = new Array<number>(n).fill(n);
      idx.forEach(({ i }, k) => {
        pos[i] = k + 1;
      });
      order.push(pos);
    }
    return cars.map((_, i) =>
      order
        .map((row, lap) => ({ lap, pos: row[i] ?? n }))
        .filter(({ lap }) => {
          const r = retiredAt(race, i);
          return r === null || lap <= r;
        })
        .map(({ lap, pos }) => `${x(lap).toFixed(1)},${y(pos).toFixed(1)}`)
        .join(" "),
    );
  })();
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-56 w-full"
      role="img"
      aria-label="Running order after every lap"
    >
      {race.sc_laps.map((lap) => (
        <rect
          key={lap}
          x={x(lap - 1)}
          y={0}
          width={W / laps}
          height={H}
          fill="oklch(0.85 0.15 90 / 0.08)"
        />
      ))}
      {lines.map((pts, i) => (
        <polyline
          key={cars[i]?.driver_id}
          points={pts}
          fill="none"
          stroke={teamColour(cars[i]?.lineage ?? "")}
          strokeWidth={1.6}
          strokeOpacity={0.85}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

/**
 * The race theatre: one simulated race at a time, replayed on the real circuit in 3D
 * (three.js, loaded on demand). Cars move with the simulator's lap-by-lap race times; the
 * tower shows the running order; "Roll again" plays another of the sampled races.
 */
export function RaceTheatre({
  theatre,
  circuit,
}: {
  theatre: TheatreMart;
  circuit: CircuitMart | null;
}) {
  const mount = useRef<HTMLDivElement>(null);
  const [raceIdx, setRaceIdx] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(30);
  const [clock, setClock] = useState(0); // race seconds
  const [failed, setFailed] = useState(false);
  const state = useRef({ t: 0, playing: true, speed: 30 as number, race: 0 });
  const race = theatre.races[raceIdx] as TheatreRace;
  const n = theatre.cars.length;
  const end = useMemo(() => {
    const last = race.t[race.t.length - 1] ?? [];
    return Math.max(...last.map((v) => v ?? 0), theatre.laps * theatre.base_lap_s);
  }, [race, theatre]);

  useEffect(() => {
    state.current.playing = playing;
    state.current.speed = speed;
  }, [playing, speed]);
  useEffect(() => {
    state.current.race = raceIdx;
    state.current.t = 0;
    setClock(0);
    setPlaying(true);
  }, [raceIdx]);

  useEffect(() => {
    const el = mount.current;
    if (!el || !circuit) return;
    let disposed = false;
    let frame = 0;
    let cleanup = () => {};
    (async () => {
      const THREE = await import("three");
      const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");
      if (disposed) return;
      let renderer: import("three").WebGLRenderer;
      try {
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      } catch {
        setFailed(true);
        return;
      }
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      el.appendChild(renderer.domElement);
      const scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x0b0d12, 120, 260);
      const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
      camera.position.set(0, 95, 95);
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.maxPolarAngle = Math.PI * 0.47;
      controls.minDistance = 30;
      controls.maxDistance = 220;
      controls.autoRotate = !reduced;
      controls.autoRotateSpeed = 0.25;

      // -- track geometry: normalised, y-up, a little elevation exaggeration
      const pts = circuit.points;
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      const zs = pts.map((p) => p[2]);
      const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
      const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
      const zMin = Math.min(...zs);
      const span =
        Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) || 1;
      const k = TRACK_SIZE / span;
      const path = pts.map(
        (p) => new THREE.Vector3((p[0] - cx) * k, (p[2] - zMin) * k * 1.5, -(p[1] - cy) * k),
      );
      const m = path.length;
      const width = 1.6;
      const positions: number[] = [];
      const colours: number[] = [];
      const sectorCol = [
        new THREE.Color(0x2a3242),
        new THREE.Color(0x26323a),
        new THREE.Color(0x30303c),
      ];
      const starts = circuit.sector_starts;
      const sectorOf = (i: number) => (i >= (starts[1] ?? m) ? 2 : i >= (starts[0] ?? m) ? 1 : 0);
      for (let i = 0; i <= m; i++) {
        const a = path[i % m] as import("three").Vector3;
        const b = path[(i + 1) % m] as import("three").Vector3;
        const tangent = new THREE.Vector3().subVectors(b, a).setY(0).normalize();
        const normal = new THREE.Vector3(-tangent.z, 0, tangent.x).multiplyScalar(width);
        const l = a.clone().add(normal);
        const r = a.clone().sub(normal);
        positions.push(l.x, l.y, l.z, r.x, r.y, r.z);
        const c = sectorCol[sectorOf(i % m)] as import("three").Color;
        colours.push(c.r, c.g, c.b, c.r, c.g, c.b);
      }
      const index: number[] = [];
      for (let i = 0; i < m; i++) {
        const q = i * 2;
        index.push(q, q + 1, q + 2, q + 1, q + 3, q + 2);
      }
      const ribbon = new THREE.BufferGeometry();
      ribbon.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      ribbon.setAttribute("color", new THREE.Float32BufferAttribute(colours, 3));
      ribbon.setIndex(index);
      ribbon.computeVertexNormals();
      scene.add(
        new THREE.Mesh(
          ribbon,
          new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.85,
            side: THREE.DoubleSide,
          }),
        ),
      );
      // glowing racing line and start line
      const line = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(path.map((p) => p.clone().setY(p.y + 0.05))),
        new THREE.LineBasicMaterial({ color: 0x5ee0f0, transparent: true, opacity: 0.55 }),
      );
      scene.add(line);
      const p0 = path[0] as import("three").Vector3;
      const p1 = path[1] as import("three").Vector3;
      const startBar = new THREE.Mesh(
        new THREE.BoxGeometry(width * 2.4, 0.08, 0.35),
        new THREE.MeshBasicMaterial({ color: 0xffffff }),
      );
      startBar.position.copy(p0).setY(p0.y + 0.06);
      startBar.lookAt(p1.x, p0.y + 0.06, p1.z);
      startBar.rotateY(Math.PI / 2);
      scene.add(startBar);
      const grid = new THREE.GridHelper(260, 52, 0x1b2230, 0x141a24);
      grid.position.y = -0.5;
      scene.add(grid);
      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const sun = new THREE.DirectionalLight(0xffffff, 1.1);
      sun.position.set(40, 80, 30);
      scene.add(sun);

      // -- cars
      const carGeo = new THREE.CapsuleGeometry(0.42, 1.1, 4, 8);
      carGeo.rotateZ(Math.PI / 2);
      const cars = new THREE.InstancedMesh(
        carGeo,
        new THREE.MeshStandardMaterial({
          roughness: 0.35,
          metalness: 0.25,
          emissiveIntensity: 0.6,
        }),
        n,
      );
      theatre.cars.forEach((c, i) => {
        cars.setColorAt(i, new THREE.Color(teamColour(c.lineage)));
      });
      scene.add(cars);
      const dummy = new THREE.Object3D();
      const at = (u: number) => {
        const f = ((u % 1) + 1) % 1;
        const x = f * m;
        const i = Math.floor(x);
        const a = path[i % m] as import("three").Vector3;
        const b = path[(i + 1) % m] as import("three").Vector3;
        return { pos: a.clone().lerp(b, x - i), dir: b.clone().sub(a) };
      };

      const resize = () => {
        const w = el.clientWidth;
        const h = el.clientHeight;
        renderer.setSize(w, h, false);
        camera.aspect = w / Math.max(h, 1);
        camera.updateProjectionMatrix();
      };
      resize();
      const ro = new ResizeObserver(resize);
      ro.observe(el);
      let visible = true;
      const io = new IntersectionObserver(([e]) => {
        visible = !!e?.isIntersecting;
      });
      io.observe(el);

      let last = performance.now();
      let tick = 0;
      const loop = (now: number) => {
        frame = requestAnimationFrame(loop);
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        if (!visible) return;
        const st = state.current;
        const r = theatre.races[st.race] as TheatreRace;
        if (st.playing) st.t = Math.min(st.t + dt * st.speed, end);
        const prog = progressAt(r, st.t, n);
        // spread cars on the grid before the start
        prog.forEach((p, i) => {
          const g = r.grid[i] ?? i;
          const u = p == null ? -0.004 * (g + 1) : p - (st.t === 0 ? 0.004 * (g + 1) : 0);
          const { pos, dir } = at(u);
          const lateral = st.t === 0 ? (g % 2 === 0 ? 0.6 : -0.6) : 0;
          dummy.position.copy(pos).setY(pos.y + 0.45);
          dummy.lookAt(
            pos
              .clone()
              .add(dir)
              .setY(pos.y + 0.45),
          );
          dummy.rotateY(Math.PI / 2);
          if (lateral) dummy.translateZ(lateral);
          const out = retiredAt(r, i);
          const gone = out !== null && p !== null && p >= out;
          dummy.scale.setScalar(gone ? 0.001 : 1);
          dummy.updateMatrix();
          cars.setMatrixAt(i, dummy.matrix);
        });
        cars.instanceMatrix.needsUpdate = true;
        controls.update();
        renderer.render(scene, camera);
        if (++tick % 6 === 0) setClock(st.t);
      };
      frame = requestAnimationFrame(loop);
      cleanup = () => {
        cancelAnimationFrame(frame);
        ro.disconnect();
        io.disconnect();
        controls.dispose();
        renderer.dispose();
        renderer.domElement.remove();
      };
    })();
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      cleanup();
    };
  }, [circuit, theatre, end, n]);

  const prog = progressAt(race, clock, n);
  const order = theatre.cars
    .map((c, i) => ({ c, i, p: prog[i] ?? -1, out: retiredAt(race, i) }))
    .sort((a, b) => b.p - a.p);
  const lap = Math.min(Math.floor(Math.max(...prog.map((p) => p ?? 0))) + 1, theatre.laps);
  const underSc = race.sc_laps.includes(lap);
  const winner = theatre.cars[race.finish[0] ?? 0];

  return (
    <div className="space-y-6">
      <section className="glass relative overflow-hidden">
        <div
          ref={mount}
          className="h-[62vh] min-h-[380px] w-full"
          role="img"
          aria-label={`3D replay of a simulated ${theatre.name} Grand Prix`}
        />
        {(!circuit || failed) && (
          <div className="absolute inset-0 grid place-items-center p-8 text-center text-sm text-[var(--color-carbon-300)]">
            {circuit
              ? "3D is not available in this browser — the order and traces below still update."
              : "No track geometry for this circuit yet."}
          </div>
        )}
        <div className="pointer-events-none absolute left-4 top-4 space-y-2">
          <p className="eyebrow">
            Simulated race {raceIdx + 1} of {theatre.races.length}
          </p>
          <p className="font-display text-3xl font-bold leading-none">
            Lap {lap}
            <span className="text-[var(--color-carbon-500)]">/{theatre.laps}</span>
          </p>
          {underSc && (
            <span className="inline-block rounded bg-[oklch(0.85_0.15_90)] px-2 py-0.5 font-mono text-xs font-bold text-[var(--color-carbon-950)]">
              SAFETY CAR
            </span>
          )}
        </div>
        <ol className="pointer-events-none absolute right-3 top-3 hidden w-44 space-y-0.5 sm:block">
          {order.slice(0, 10).map(({ c, i, out, p }, k) => (
            <li
              key={c.driver_id}
              className="flex items-center gap-2 rounded bg-[oklch(0.12_0.006_265/0.75)] px-2 py-0.5 font-mono text-xs backdrop-blur"
            >
              <span className="w-4 text-right text-[var(--color-carbon-500)]">{k + 1}</span>
              <span
                className="h-3 w-1 rounded-full"
                style={{ background: teamColour(c.lineage) }}
              />
              <span className={out !== null && p >= out ? "line-through opacity-50" : ""}>
                {c.code}
              </span>
              <span className="sr-only">{i}</span>
            </li>
          ))}
        </ol>
        <div className="flex flex-wrap items-center gap-3 border-t border-white/5 px-4 py-3">
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            className="rounded-full bg-[var(--color-signal)] px-4 py-1.5 font-display text-xs font-semibold text-[var(--color-carbon-950)]"
          >
            {playing ? "Pause" : "Play"}
          </button>
          <div className="flex rounded-full bg-[var(--color-carbon-850)] p-1">
            {SPEEDS.map((s) => (
              <button
                key={s}
                type="button"
                aria-pressed={speed === s}
                onClick={() => setSpeed(s)}
                className={`rounded-full px-2.5 py-1 font-mono text-[0.7rem] ${speed === s ? "bg-white text-[var(--color-carbon-950)]" : "text-[var(--color-carbon-300)]"}`}
              >
                ×{s}
              </button>
            ))}
          </div>
          <input
            type="range"
            min={0}
            max={end}
            step={1}
            value={clock}
            onChange={(e) => {
              state.current.t = Number(e.target.value);
              setClock(state.current.t);
            }}
            className="min-w-40 flex-1 accent-[var(--color-signal)]"
            aria-label="Race time"
          />
          <button
            type="button"
            onClick={() => setRaceIdx((r) => (r + 1) % theatre.races.length)}
            className="rounded-full border border-white/10 px-4 py-1.5 font-display text-xs font-semibold hover:bg-white/5"
          >
            Roll again
          </button>
        </div>
      </section>

      <section className="glass p-5">
        <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
          <div>
            <p className="eyebrow">Running order, lap by lap</p>
            <p className="mt-1 text-sm text-[var(--color-carbon-300)]">
              This simulated race was won by{" "}
              <span className="text-[var(--color-heat)]">{winner?.name}</span>
              {race.sc_laps.length > 0 && " · shaded laps ran behind the safety car"}.
            </p>
          </div>
          {theatre.result && (
            <p className="text-xs text-[var(--color-carbon-500)]">
              Actual winner: {theatre.cars.find((c) => c.driver_id === theatre.result?.[0])?.name}
            </p>
          )}
        </div>
        <Traces race={race} cars={theatre.cars} />
      </section>
    </div>
  );
}
