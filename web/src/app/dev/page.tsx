import { DevPanel } from "@/components/DevPanel";

export const metadata = { title: "Dev mode" };

export default function DevPage() {
  return (
    <div className="space-y-8">
      <header>
        <p className="eyebrow">Local only · talks to f1p dev serve on this machine</p>
        <h1 className="font-display text-4xl font-bold">Dev mode</h1>
        <p className="mt-2 max-w-3xl text-sm text-[var(--color-carbon-300)]">
          Run the checks and the M6 checklist, and watch them live: output, progress, and the
          machine&apos;s load, memory and throttling. Jobs run one at a time by default; the holdout
          (B12) is never part of a pipeline and needs a typed confirmation.
        </p>
      </header>
      <DevPanel />
    </div>
  );
}
