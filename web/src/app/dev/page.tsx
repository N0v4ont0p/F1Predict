import { DevPanel } from "@/components/DevPanel";

export const metadata = { title: "Dev mode" };

export default function DevPage() {
  return (
    <div className="space-y-6">
      <header>
        <p className="eyebrow">Runs on this Mac</p>
        <h1 className="font-display text-4xl font-bold">Dev mode</h1>
        <p className="mt-2 max-w-3xl text-sm text-[var(--color-carbon-300)]">
          Pick a preset and press Start. While it runs you can Pause, Stop or Restart it;
          afterwards, Retry re-runs only what failed. Live output shows below. The holdout is never
          part of a preset.
        </p>
      </header>
      <DevPanel />
    </div>
  );
}
