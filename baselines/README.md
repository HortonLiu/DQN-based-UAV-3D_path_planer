# Historical simulator snapshots

| Snapshot | Original directory | Default trainer | Base episode step limit |
| --- | --- | --- | ---: |
| `ddpg-apf/` | `DQN-based-UAV-3D_path_planer_RL(DDPG)+APF` | `DDPG_Trainer` | 1800 |
| `td3-apf/` | `DQN-based-UAV-3D_path_planer_RL(TD3)+APF` | `PG_ITD3_Trainer` | 3600 |
| `sac-legacy/` | `DQN-based-UAV-3D_path_planer-main` (earlier source) | `SAC_Trainer` | 1200 |

Each snapshot retains its own source files and simulation XML. In `BaseClass/BaseTrainer.py` the replay buffer is imported from the same snapshot, replacing the original machine-specific import path. Third-party notices apply across the repository. [Upstream MIT notice](../LICENSE) · [Third-party licenses](../licenses/)

## Verify each snapshot

Install the pinned [dependencies](../README.md#quick-start) and run this PowerShell block from the repository root without checkpoints. The baseline programs use their own simulation XML by default.

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$smoke = @'
from simulator import simulator

sim = simulator()
try:
    uav = sim.env.Agents[0]
    uav.Trainer.Is_Train = 0
    uav.original_max_step = 4
    uav.Max_Step = 4
    result = sim.env.run_eposide(0.0)
    print(type(uav.Trainer).__name__)
    print({key: result[key] for key in ('step', 'success', 'lose', 'meet_threaten')})
finally:
    sim.writer.close()
'@

foreach ($version in 'ddpg-apf', 'td3-apf', 'sac-legacy') {
    Push-Location "baselines/$version"
    try {
        $smoke | & $python -
        if ($LASTEXITCODE -ne 0) { throw "Smoke test failed for $version" }
    } finally {
        Pop-Location
    }
}
```

Each verified snapshot reports its trainer name, `step=4`, `success=0`, `lose=1`, and `meet_threaten=0`. Each writes an HTML report in `DataBase/experience/` and a UAV CSV in `logs/` relative to its own directory. The legacy chart reports default to CDN assets. The smoke run uses the inherited episode bookkeeping, so run it in a fresh copy.

## Compare DDPG and fixed-gain APF

Use this procedure to run one evaluation rollout per method with evaluation seed 10123. A clean checkpoint-free verification produces two rows in `logs/ddpg_apf_experiment_*.csv`, both ending on contact.

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
Push-Location .\baselines\ddpg-apf
try {
    & $python run_ddpg_apf_experiment.py --train-episodes 0 --eval-episodes 1 --seed 123 --no-tensorboard --no-render-html
} finally {
    Pop-Location
}
```

The DDPG snapshot defaults to `Load_Model=0`. Its loader expects the `Mod/DDPG_actor_UAV_0.pth` and `Mod/DDPG_critic_UAV_0.pth` pair when enabled. The TD3 snapshot looks for `Mod/PG_ITD3_actor_UAV_0.pth`, `Mod/PG_ITD3_critic_1_UAV_0.pth`, and `Mod/PG_ITD3_critic_2_UAV_0.pth`. Use trusted checkpoints with these legacy loaders.

The archived TD3 runs are labeled `+VOAPF`. Match each archive's original code, map, overlap conditions, and starting weights before comparing results with these snapshots. [Historical data](../data/historical/README.md)

## Run the baselines on shared scenes

From each baseline directory, pass `--map-level 2` for separated obstacles or `--map-level 3` for layouts that allow overlap. Both use the repository's shared scene generator, so the level and episode select the same start, goal, heading, and obstacle geometry as the manuscript profile. The baseline trainers, rewards, altitude conventions, and episode limits follow their own snapshot settings.

```powershell
# From baselines/ddpg-apf
python run_ddpg_apf_experiment.py --map-level 2 --train-episodes 1000 --eval-episodes 0 --no-render-html

# From baselines/td3-apf
python simulator.py --map-level 2 --episodes 1000
```

The CSV output identifies `map_level`, `map_episode`, and `map_seed`. With shared maps enabled, DDPG evaluation defaults to zero episodes. To request an evaluation, allocate up to 1,000 episodes in total. The fixed-gain APF and DDPG evaluation rollouts then share the scene IDs immediately following training. These generated layouts support new matched comparisons. The curve manifests identify the separate historical runs used in the paper.

Shared runs read and write checkpoints under each baseline's `Mod/manuscript/level2/` or `Mod/manuscript/level3/`. DDPG loads checkpoints when `Load_Model=1`.
