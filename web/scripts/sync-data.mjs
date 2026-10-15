// Copy the published marts (`f1p model publish`) into public/data for the static build.
import { cpSync, existsSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const src = resolve(process.argv[2] ?? "../artifacts/marts");
const dest = resolve("public/data");
if (!existsSync(src)) {
  console.error(`no marts at ${src} — run \`uv run f1p model publish\` first`);
  process.exit(1);
}
rmSync(dest, { recursive: true, force: true });
cpSync(src, dest, { recursive: true });
console.log(`synced ${src} → ${dest}`);
