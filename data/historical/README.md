# Archived simulation data

These files are curated from the research archive. They cover three SAC + VOAPF 2.0 S2R runs with full episode score records and seven earlier DDPG and TD3 runs with outcomes extracted from TensorBoard. [Source paths and SHA-256 manifest](manifest.csv)

## SAC + VOAPF 2.0 S2R

| Archive date | Scenario | UAV speed | Successful episodes | Final EMA | Episode files |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-10-30 | 20 static and 6 moving obstacles | about 1 m/s | 873 / 1,000 | 86.78% | [Scores](mixed-480hz/scores.csv) · [Episode 412](mixed-480hz/episode-412.json) |
| 2026-10-31 | 26 static obstacles | 1 m/s | 782 / 1,000 | 79.22% | [Scores](static-480hz-v1/scores.csv) · [Episode 301](static-480hz-v1/episode-301.json) |
| 2026-11-02 | 26 static obstacles | 4 m/s | 840 / 1,000 | 85.64% | [Scores](static-480hz-v4/scores.csv) · [Episode 740](static-480hz-v4/episode-740.json) |

Sampled trajectory JSON records confirm 480 Hz in `timing.control_hz`. Speed is corroborated by the run labels and the path length relative to simulation time. The EMA uses a decay of 0.99 and starts at zero. These values describe the archived runs listed here.

These score series also supply the cases in manuscript Figures 16(b) and 17(c). The exact vector build uses initialization-corrected EMA at the same decay. The three representative JSONs here are examples from these score runs. The manuscript trajectory panels have separate sources linked from the [manuscript figure bundle](../manuscript/README.md).

Each score CSV retains 1,000 episode records. `report_id` replaces the original absolute `html_path`. The same UUID identifies the archived HTML and JSON in the run's `experience/` directory. All 3,000 score references were matched to those files. The selected trajectories are successful rollouts with zero recorded contacts. The JSON content retains the original bytes and uses display-friendly filenames.

`meet_threaten` counts updates at which a collision flag was recorded by the historical simulator. In the mixed run, 809 of 873 logged successful episodes have a nonzero count. Use the original `success` field to reproduce the displayed goal-arrival curves. Each run has a three-file SAC checkpoint set from near the end of training. The manifest records its archive paths and hashes. [Checkpoint placement and verified commands](../../docs/reproduction.md#load-a-verified-historical-checkpoint)

## DDPG and TD3

[ddpg-td3-success.csv](ddpg-td3-success.csv) contains 7,000 records with `run`, `episode` and `success` from TensorBoard's `Train/Success_Rate` scalar. The export uses the archival run names and excludes wall-clock timestamps and the original event filenames. Manifest entries for TensorBoard point to the event directory and give the hash of its event file.

The verified outer archive folder and nested TensorBoard folder for the 2026-06-06 11:40 DDPG run both use the `+VOAPF` label. The [figure manifest](../manuscript/data/manifest.csv) maps its outcomes to Figure 15(c), whose panel is labeled DDPG+APF. The archive folder name and deployed controller implementation require independent run configuration evidence. Many early score CSVs lack usable episode-level outcomes.
