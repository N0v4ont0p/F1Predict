# Data Dictionary — master dataset

One row == one driver's result in one session.

| Column | Type | Description |
| --- | --- | --- |
| `season` | int32 | Championship year. |
| `round` | int32 | Round number within the season. |
| `date` | str | Race date (ISO). |
| `race_name` | str | Grand Prix name. |
| `circuit_id` | str | Circuit identifier. |
| `country` | str | Host country. |
| `session` | str | Session type: R (race), S (sprint), Q (qualifying). |
| `driver_id` | str | Driver identifier. |
| `driver_name` | str | Driver full name. |
| `constructor_id` | str | Constructor identifier. |
| `constructor_name` | str | Constructor name. |
| `grid` | int32 | Starting position (0 == pit lane/unknown). |
| `position` | int32 | Classified finishing position (0 == DNF/unclassified). |
| `status` | str | Result status (Finished / +1 Lap / Accident / Engine / ...). |
| `points` | float32 | Championship points scored. |
| `laps` | int32 | Laps completed. |
| `finished` | int32 | 1 if classified finish else 0. |