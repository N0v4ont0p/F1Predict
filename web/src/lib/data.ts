// Build-time data access: the marts are read from public/data during static generation.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type {
  ChampionshipMart,
  CircuitMart,
  EventMart,
  IndexMart,
  ModelMart,
  RatingsMart,
  TheatreMart,
} from "./types";

const ROOT = join(process.cwd(), "public", "data");

function read<T>(rel: string): T | null {
  const path = join(ROOT, rel);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

export function getIndex(): IndexMart {
  const idx = read<IndexMart>("index.json");
  if (!idx) throw new Error("public/data/index.json missing — run `pnpm sync-data`");
  return idx;
}

export function getEvent(id: string): EventMart {
  const ev = read<EventMart>(`events/${id}.json`);
  if (!ev) throw new Error(`no mart for ${id}`);
  return ev;
}

export const getRatings = (): RatingsMart | null => read<RatingsMart>("ratings.json");
export const getModel = (): ModelMart | null => read<ModelMart>("model.json");
export const getChampionship = (): ChampionshipMart | null =>
  read<ChampionshipMart>("championship.json");

export function featuredEvent(idx: IndexMart): string {
  if (idx.next_event) return idx.next_event;
  const raced = idx.events.filter((e) => e.status === "raced");
  return (raced.at(-1) ?? idx.events[0])?.event_id ?? "";
}

export const hasChampionshipHistory = (): boolean =>
  existsSync(join(ROOT, "championship_history.json"));

export const hasTheatre = (id: string): boolean => existsSync(join(ROOT, "theatre", `${id}.json`));
export const getTheatre = (id: string): TheatreMart | null =>
  read<TheatreMart>(`theatre/${id}.json`);
export const readCircuit = (id: string): CircuitMart | null =>
  read<CircuitMart>(`circuits/${id}.json`);
