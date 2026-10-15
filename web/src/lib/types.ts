// Shapes of the JSON marts written by `f1p model publish` (schema_version 1).

export type Checkpoint = "C1" | "C4";

export interface Stamp {
  schema_version: number;
  generated_at: string;
  lake_snapshot: number | null;
}

export interface Car {
  rank: number;
  driver_id: string;
  name: string;
  code: string;
  number: string | null;
  constructor_id: string;
  team: string;
  lineage: string;
  p_win: number;
  p_podium: number;
  p_points: number;
  p_finish: number;
  expected_position: number;
  position_probs: number[];
}

export interface CheckpointForecast {
  model: string;
  as_of: string;
  meta: Record<string, number>;
  cars: Car[];
}

export interface ResultRow {
  driver_id: string;
  name: string;
  code: string;
  constructor_id: string;
  team: string;
  lineage: string;
  position: number | null;
  classified: boolean;
  status: string;
  grid: number | null;
  points: number | null;
}

export interface WeatherBlock {
  wet_prob?: number;
  precip_mm: number;
  temp_c: number | null;
  wet?: boolean;
}

export type Scores = Partial<
  Record<
    "winner_logloss" | "winner_hit" | "podium_brier" | "points_brier" | "finisher_pl_loglik",
    number
  >
>;

export interface EventMart extends Stamp {
  event_id: string;
  season: number;
  round: number;
  name: string;
  official_name: string;
  circuit_id: string;
  race_date: string;
  race_start_utc?: string | null;
  sprint: boolean;
  status: "raced" | "next" | "upcoming";
  weather: { climatology?: WeatherBlock; forecast?: WeatherBlock; observed?: WeatherBlock };
  checkpoints: Partial<Record<Checkpoint, CheckpointForecast>>;
  scores: Partial<Record<Checkpoint, Record<string, Scores>>>;
  result: ResultRow[] | null;
}

export interface Favourite {
  driver_id: string;
  code: string;
  p_win: number;
}

export interface CalendarRow {
  event_id: string;
  round: number;
  name: string;
  circuit_id: string;
  race_date: string;
  sprint: boolean;
  status: "raced" | "next" | "upcoming";
  favourite_C1?: Favourite;
  favourite_C4?: Favourite;
  winner?: { driver_id: string; code: string; team: string };
  winner_logloss_C1?: number;
  winner_logloss_C4?: number;
  p_winner_C1?: number;
  p_winner_C4?: number;
  wet?: boolean;
}

export interface IndexMart extends Stamp {
  season: number;
  next_event: string | null;
  events: CalendarRow[];
}

export interface Trajectory {
  event_id: string;
  mu: number;
  sd: number;
}

export interface RatingsMart extends Stamp {
  season: number;
  drivers: (Pick<Car, "driver_id" | "name" | "code"> & {
    mu: number;
    sd: number;
    series: Trajectory[];
  })[];
  teams: { lineage: string; team: string; mu: number; sd: number; series: Trajectory[] }[];
}

export interface Comparison {
  reference: string;
  mean_diff: number;
  lo: number;
  hi: number;
  significant: boolean;
}

export interface ModelMart extends Stamp {
  model: string;
  seasons: [number, number];
  leaderboard: ({
    model: string;
    checkpoint: Checkpoint;
    events: number;
    ece_top3?: number;
  } & Scores)[];
  comparisons: Record<Checkpoint, Record<string, Comparison>>;
  reliability: Record<string, { predicted: number; observed: number; count: number }[]>;
  combiner: Partial<
    Record<
      Checkpoint,
      { weights: Record<string, number>; stage_temperature: number[]; rain_gamma: number }
    >
  >;
}

export interface Standing {
  driver_id?: string;
  name?: string;
  code?: string;
  constructor_id?: string;
  team: string;
  lineage?: string;
  current: number;
  p_title: number;
  expected: number;
  quantiles: number[];
  p_position: number[];
}

export interface ChampionshipMart extends Stamp {
  season: number;
  remaining: string[];
  sprints_remaining: number;
  n_sims: number;
  quantiles: number[];
  drivers: Standing[];
  constructors: Standing[];
}
