# Autonomous Drone Navigation based on Reinforcement Learning integrated with Artificial Potential Field

**Ming-Chieh Liu · Chen-Yang Lan**  
Department of Mechanical Engineering, National Taiwan University of Science and Technology, Taipei, Taiwan

Research software and simulation data for the accompanying manuscript on learning-assisted local drone navigation. The figure reproduction inputs here follow the author-final Overleaf export dated 2026-11-26.

[Run the simulator](#quick-start) · [Manuscript alignment](docs/manuscript-alignment.md) · [Rebuild Figures 14–17](data/manuscript/README.md) · [Reproduction guide](docs/reproduction.md)

![Plan view of the archived mixed-scene rollout, showing the UAV trajectory, static obstacle footprints, and six moving-obstacle tracks](assets/figures/archived-mixed-trajectory.png)

*Archived SAC + VOAPF episode 412, simulated at 480 Hz. The scene has 20 static and six moving obstacles. The moving-obstacle lines show recorded center positions.* [Trajectory data](data/historical/mixed-480hz/episode-412.json) · [README plot script](scripts/plot_archived_trajectory.py) · [Manuscript Figure 16(a) source](data/manuscript/README.md)

## Method

A Soft Actor–Critic (SAC) policy uses a 121-cell local ray-casting map and five kinematic features to tune the artificial potential field (APF) navigator. The policy adjusts steering bias and attraction and repulsion gains. Select `--controller-profile manuscript` to run the VOAPF 2.0 midpoint rule and baseline reward for new experiments. The default `legacy` profile uses trapped-position placement and the original S2R reward. [Implementation and training scope](docs/reproduction.md#manuscript-protocol)

The manuscript evaluates fixed-altitude planar navigation in randomized static and mixed scenes, including asynchronous control and sensor updates. The simulator models a 500 × 500 m horizontal workspace. The 480 Hz mode models 20 Hz RTK and 10 Hz LiDAR updates.

## Quick start

From the repository root in Windows PowerShell, create an environment and install the pinned dependencies. The smoke tests were verified on Windows x64 with Python 3.9.9 and PyTorch 2.5.1 on CPU. The standard simulator can run without a database server or hardware.

```powershell
py -3.9 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install -r requirements.txt
& $python -m unittest discover -s tests -v
& $python simulator.py --scene static --hardware-hz 0 --uav-speed 1 --episodes 1 --max-seconds 2 --no-train
& $python simulator.py --scene mixed --hardware-hz 1 --uav-speed 4 --episodes 1 --max-seconds 2 --no-train
```

In a fresh checkout, these two smoke runs end at the two-second time limit. Expect 2 and 960 control updates and path lengths of approximately 2 and 8 m. Scores and TensorBoard logs are written to `logs/`. HTML and JSON trajectories are written to `DataBase/experience/`. [Expected outputs and schema](data/examples/README.md)

Test the manuscript controller and the seeded Level 3 reset with this two-second run.

```powershell
& $python simulator.py --controller-profile manuscript --map-level 3 --scene static --hardware-hz 0 --uav-speed 1 --episodes 1 --max-seconds 2 --no-train
```

Level 2 uses the same dimensions and applies the 2 m AABB placement rule. These commands create new simulation results. The figure rebuild uses the archived outcomes listed below. [Protocol and supporting evidence](docs/manuscript-alignment.md)

To regenerate the figure, run `& $python scripts/plot_archived_trajectory.py` from the repository root.

## Simulation data

| Included data | Coverage |
| --- | --- |
| [Verified examples](data/examples/) | Two fresh-checkout smoke runs with score summaries and trajectory JSON |
| [Manuscript Figures 14–17](data/manuscript/README.md) | 20 curve series from 17 TensorBoard exports, three original trajectory HTMLs, and vector figure rebuild scripts |
| [Archived SAC + VOAPF S2R](data/historical/README.md#sac--voapf-20-s2r) | Three 480 Hz training runs with 1,000 episode scores and one additional trajectory each |
| [Archived DDPG and TD3](data/historical/ddpg-td3-success.csv) | Episode outcomes for seven runs extracted from TensorBoard, 7,000 rows total |
| [Provenance manifest](data/historical/manifest.csv) | Source paths and SHA-256 values for archived data and associated SAC checkpoints |

The 480 Hz outcome runs supply Figures 16(b) and 17(c). The author-final vectors use initialization-corrected EMA with `β = 0.99`. Their final values are 86.79% for mixed obstacles, 79.22% for static obstacles at 1 m/s, and 85.64% for static obstacles at 4 m/s. Figures 16(a), 17(a), and 17(b) use illustrative trajectories from earlier runs. The archived v0.1.0 package contains the three score series and additional trajectory examples. The figure-specific inputs and scripts were added to this working tree after that release. [Reproduction guide](docs/reproduction.md) · [Paper and archive cross-check](docs/release-checklist.md)

## Repository map

| Location | Contents |
| --- | --- |
| [`simulator.py`](simulator.py) and [`config/`](config/) | Current SAC + VOAPF simulation and experiment settings |
| [`Agents/`](Agents/), [`Envs/`](Envs/), [`Obstacles/`](Obstacles/), [`Trainer/`](Trainer/) | Agent dynamics, environments, obstacle models and learning algorithms |
| [`baselines/`](baselines/) | Independent DDPG + APF, TD3 and earlier SAC snapshots with their own configurations |
| [`data/`](data/) and [`scripts/`](scripts/) | Simulation data and figure generation |
| [`docs/`](docs/) and [`tests/`](tests/) | Reproduction instructions, artifact status and simulator regression tests |

## Citation

Cite the version of the software or figure inputs you use and the accompanying manuscript. [`CITATION.cff`](CITATION.cff) · [Zenodo version 0.2.0](https://doi.org/10.5281/zenodo.24556113) · [Zenodo version 0.1.0](https://doi.org/10.5281/zenodo.24494002)

## Acknowledgments and licenses

The research software is licensed under MIT. The packaged simulation scores, episode outcomes, and trajectory data are licensed under CC BY 4.0. [Software license](LICENSE) · [Simulation data license](data/LICENSE)

The simulator builds on the RLGF project by young how. The upstream copyright notice and the licenses for the adapted training examples and offline chart bundles accompany the software. [RLGF upstream](https://github.com/young-how/DQN-based-UAV-3D_path_planer) · [Training code notices](licenses/) · [Chart bundle notices](assets/pyecharts/)
