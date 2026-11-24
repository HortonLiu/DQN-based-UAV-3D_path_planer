# Smoke-run data

These two examples were generated with the [Quick start commands](../../README.md#quick-start) using Windows x64, Python 3.9.9 and PyTorch on CPU. Each run uses seed 42, initial weights, one episode, a two-second base budget, and optimizer updates and saves disabled. [Dependency constraints](environment-win-py39.txt)

| File | Scenario | Control rate | UAV speed |
| --- | --- | ---: | ---: |
| [static-1hz-v1.json](static-1hz-v1.json) | 26 static obstacles | 1 Hz | 1 m/s |
| [mixed-480hz-v4.json](mixed-480hz-v4.json) | 20 static and 6 moving obstacles | 480 Hz | 4 m/s |

[scores.csv](scores.csv) standardizes the two original score rows. It excludes wall-clock duration, timestamp, and absolute report path. `trajectory_json` references the corresponding JSON. The `control_updates` field is the number of physics updates. The inherited `step` counter increases by two per update. Path length and remaining distance are in meters. The `meet_threaten` field is the recorded contact count.

In the JSON, `Threaten` contains the initial obstacle definitions. Each `path` row is `[x, y, z]` in meters, at the corresponding `path_time_s` sample in simulated seconds. A moving obstacle's `trajectory_txyz` stores `[t, x, y, z]` with the same units. `timing` captures the update rate and step counts. Both smoke runs end at the two-second limit with `success=0` and `failed=1`.
