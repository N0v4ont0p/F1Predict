# Feature Glossary

All features are temporal-safe (computed using only prior races).

| Feature | Description |
| --- | --- |
| `f_grid` | Starting grid position (0/unknown imputed to field median). |
| `f_pole` | 1 if starting from pole position. |
| `f_front_row` | 1 if starting on the front row (grid <= 2). |
| `f_top10_start` | 1 if starting inside the top 10. |
| `f_driver_pos_3` | Driver mean finishing position over the previous 3 races. |
| `f_driver_pos_5` | Driver mean finishing position over the previous 5 races. |
| `f_driver_pos_10` | Driver mean finishing position over the previous 10 races. |
| `f_driver_pts_3` | Driver mean points over the previous 3 races. |
| `f_driver_pts_5` | Driver mean points over the previous 5 races. |
| `f_driver_pts_10` | Driver mean points over the previous 10 races. |
| `f_driver_finrate_3` | Driver finish (classified) rate over the previous 3 races. |
| `f_driver_finrate_5` | Driver finish (classified) rate over the previous 5 races. |
| `f_driver_finrate_10` | Driver finish (classified) rate over the previous 10 races. |
| `f_driver_career_pos` | Driver career mean finishing position (expanding, excl. current). |
| `f_driver_races` | Number of races contested before this one (experience). |
| `f_rookie` | 1 if fewer than 5 career races (cold-start flag). |
| `f_cons_pos_3` | Constructor mean finishing position over previous 3 race-entries. |
| `f_cons_pos_5` | Constructor mean finishing position over previous 5 race-entries. |
| `f_cons_pos_10` | Constructor mean finishing position over previous 10 race-entries. |
| `f_cons_finrate_3` | Constructor reliability (finish rate) over previous 3 entries. |
| `f_cons_finrate_5` | Constructor reliability (finish rate) over previous 5 entries. |
| `f_cons_finrate_10` | Constructor reliability (finish rate) over previous 10 entries. |
| `f_driver_track_pos` | Driver historical mean position at this circuit. |
| `f_cons_track_pos` | Constructor historical mean position at this circuit. |
| `f_teammate_grid_delta` | Driver grid minus team mean grid (car-adjusted qualifying skill). |
| `f_season_progress` | Round number normalised by season length (0..1). |
| `f_grid_x_driverform` | Interaction: grid position x recent driver form. |
| `f_grid_sq` | Grid position squared (non-linear). |
| `f_grid_log` | log(1 + grid) (non-linear). |