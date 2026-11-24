# Historical simulator snapshots

| Snapshot | Original directory | Default trainer | Base episode step limit |
| --- | --- | --- | ---: |
| `ddpg-apf/` | `DQN-based-UAV-3D_path_planer_RL(DDPG)+APF` | `DDPG_Trainer` | 1800 |
| `td3-apf/` | `DQN-based-UAV-3D_path_planer_RL(TD3)+APF` | `PG_ITD3_Trainer` | 3600 |
| `sac-legacy/` | `DQN-based-UAV-3D_path_planer-main` (earlier source) | `SAC_Trainer` | 1200 |

Each snapshot retains its own source files and simulation XML. In `BaseClass/BaseTrainer.py` the replay buffer is imported from the same snapshot, replacing the original machine-specific import path. Third-party notices apply across the repository. [Upstream MIT notice](../LICENSE) · [Third-party licenses](../licenses/)

## Verify each snapshot

Install the pinned [dependencies](../README.md#quick-start) and run this PowerShell block from the repository root without checkpoints. The legacy `simulator.py` programs read their own XML and lack the current simulator's CLI options.

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
