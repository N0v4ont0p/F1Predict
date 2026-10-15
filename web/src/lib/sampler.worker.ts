// Runs the staged Plackett-Luce sampler off the main thread for the What-if Lab.
import { type SampleInput, sample } from "./sampler";

self.onmessage = (event: MessageEvent<{ id: number; input: SampleInput }>) => {
  const { id, input } = event.data;
  postMessage({ id, output: sample(input) });
};
