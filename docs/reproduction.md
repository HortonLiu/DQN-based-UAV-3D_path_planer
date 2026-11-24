# Reproduction guide

## Verify the current simulator

Run the two commands under [Quick start](../README.md#quick-start) from a fresh checkout without checkpoints. The expected outputs are in [the example data](../data/examples/README.md).

| Scenario | Simulation time | Control updates | Legacy `step` | Path length | Moving obstacles |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static, 1 Hz, 1 m/s | 2 s | 2 | 4 | about 2 m | 0 |
| Mixed, 480 Hz, 4 m/s | 2 s | 960 | 1920 | about 8 m | 6 |

Both runs reach the time limit with `success=0`, `failed=1`, and zero contacts. Use `control_updates` for the actual number of updates. The legacy `step` counter increases by two per update. Simulation time is `control_updates × control_dt_s`. Timestamps, report paths, and wall-clock measurements are run-specific. Float outputs can vary slightly across environments. The current simulator seeds Python, NumPy, and PyTorch with 42.

To use the verified Windows x64 Python 3.9 dependency versions, add `-c data/examples/environment-win-py39.txt` to the pip install command. Open a generated report in `DataBase/experience/` and keep its sibling `assets/` directory for offline charts. Run `.\.venv\Scripts\python.exe -m tensorboard.main --logdir logs` to inspect training scalars. The simulator can also write a per-UAV CSV in `logs/` and a root `path.csv` after successful navigation.

## Set up an experiment

| Scenario | Environment, UAV, and trainer configuration | SAC checkpoint root |
| --- | --- | --- |
| `static` | `config/PathPlan_City_Static.xml`, `config/UAV_Static.xml`, `config/Trainer_Static.xml` | `Mod/S2R_Static_time_consistent/` |
| `mixed` | `config/PathPlan_City.xml`, `config/UAV.xml`, `config/Trainer.xml` | `Mod/S2R_time_consistent/` |

| CLI option | Values |
| --- | --- |
| `--hardware-hz` | `0` for 1 Hz, `1` for 480 Hz |
| `--scene` | `static` for 26 static obstacles, `mixed` for 20 static and six moving obstacles |
| `--uav-speed` | `1` or `4` m/s |
| `--episodes`, `--max-seconds` | Override the episode count and base time budget. The default run is 1,000 episodes with a 1,200 s base budget per episode |
| `--no-train` | Disable optimizer updates and checkpoint saves. The simulator still loads a complete checkpoint for the selected scenario when available |

The simulator prompts for the first three options when omitted. The scenarios share `config/buildings.xml` and the current simulator randomizes the map every two episodes. The moving obstacles translate along X, Y, or an oblique direction, with two obstacles assigned to each pattern and a total speed of 4 m/s. Their centers reverse at the workspace boundary. `Dynamic_Obstacles/half_range` in the mixed XML can set the one-sided travel distance. The original endpoints for Figure 15 require source-run confirmation.

Both control modes use a one-second SAC decision interval and `gamma=0.99`. At 480 Hz, modeled RTK and LiDAR updates run at 20 and 10 Hz respectively. At 1 Hz, both are sampled at every update. The stagnation detector uses a 10 s window and a 3 m displacement threshold. Virtual-obstacle generation can extend the base episode budget by 50 s.

The historical DDPG, TD3 and SAC versions use independent entry points and configurations. [Baseline verification instructions](../baselines/README.md)

## Load a verified historical checkpoint

The historical manifest identifies three checkpoint sets and their SHA-256 values. Obtain the actor and two critics from the same run and place them together under the matching path. The filenames are `actor_SAC_UAV_0.pth`, `critic_1_SAC_UAV_0.pth`, and `critic_2_SAC_UAV_0.pth`. From a clean checkout, an incomplete set causes the simulator to start from initial weights. [Source manifest](../data/historical/manifest.csv)

| Run | Directory under the repository | Saved epoch |
| --- | --- | ---: |
| Mixed, 480 Hz, nominally 1 m/s | `Mod/S2R_time_consistent/v1/` | 858000 |
| Static, 480 Hz, 1 m/s | `Mod/S2R_Static_time_consistent/v1/` | 861100 |
| Static, 480 Hz, 4 m/s | `Mod/S2R_Static_time_consistent/v4/` | 417000 |

All three sets loaded the models and optimizer state on CPU and ran for 960 updates over two simulated seconds. For the mixed scenario, run the following command from the root.

```powershell
.\.venv\Scripts\python.exe simulator.py --hardware-hz 1 --scene mixed --uav-speed 1 --episodes 1 --max-seconds 2 --no-train
```

For static 480 Hz runs, use `--scene static` and set the matching speed. For 1 Hz checkpoints, add `one_second/` between the checkpoint root and the speed directory. The `--no-train` mode still samples from the learned policy distribution.

These three checkpoint sets come from near the end of their training runs. Exact retraining requires the original initial state, run configuration and code. To train with initial weights in the current simulator, use an empty checkpoint directory and omit `--no-train`. Existing complete checkpoints restore weights, optimizer state, and the saved epoch. Exact continuation would also require the original replay buffer and RNG state. Preserve the run command, XML, environment, score CSV, trajectory JSON, and TensorBoard events.

## Recalculate the archived training outcomes

The example below recomputes the proportion of successful episodes and the final exponential moving average (EMA) for each included run. The EMA starts at zero, with a decay of 0.99. Plot the updated `ema` against `episode` to recreate each archived run's curve. [Data descriptions and source IDs](../data/historical/README.md)

```powershell
@'
import csv
from collections import defaultdict
from pathlib import Path

root = Path('data/historical')
series = defaultdict(list)
for scores in sorted(root.glob('*/scores.csv')):
    with scores.open(encoding='utf-8', newline='') as handle:
        series[scores.parent.name] = [
            (int(row['Episode']), int(row['success'])) for row in csv.DictReader(handle)
        ]
with (root / 'ddpg-td3-success.csv').open(encoding='utf-8', newline='') as handle:
    for row in csv.DictReader(handle):
        series[row['run']].append((int(row['episode']), int(row['success'])))

for run, outcomes in series.items():
    outcome_sum = 0
    ema = 0.0
    for episode, success in sorted(outcomes):
        outcome_sum += success
        ema = 0.99 * ema + 0.01 * success
    print(f'{run}: {outcome_sum}/{len(outcomes)}, EMA={ema:.2%}')
'@ | .\.venv\Scripts\python.exe -
```

Regenerate the README's trajectory figure with `.\.venv\Scripts\python.exe scripts/plot_archived_trajectory.py`. The source is episode 412 of the archived mixed run.

The three SAC archives yield final EMA values of 86.78%, 79.22%, and 85.64%. The manuscript's Figure 15 reports 98.1% and 95.3% for its mixed and static comparisons. The figure's original run IDs and analysis settings are pending confirmation. [Figure and release status](release-checklist.md)
