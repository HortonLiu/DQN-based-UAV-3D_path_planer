# Manuscript alignment

This comparison uses the **live Overleaf source downloaded on 26 November 2026**. The manuscript, bibliography and all 16 Figure 14–17 vector assets matched the earlier author-final export byte for byte. The SHA-256 of `main.tex` is `75a2532795ddf9f4c7bd5b4dfbba25763939cf009fff8d71637d791e8a2d10a5`. The [v0.1.0 archive](https://doi.org/10.5281/zenodo.24494002) preserves the earlier simulator and curated simulation data.

## Differences from v0.1.0

| Area | Author-final export | v0.1.0 and current evidence |
| --- | --- | --- |
| Figures 14–15 | Method and map-complexity success curves | v0.1.0 included selected outcome series. The figure-specific inputs, mapping and rebuild scripts are now in [the manuscript data bundle](../data/manuscript/README.md). |
| Figures 16(b) and 17(c) | Three 480 Hz outcome series, with static 1 m/s shared between panels | v0.1.0 included the three score series. The figure bundle adds the corresponding raw outcome exports and initialization-corrected EMA. |
| Figures 16(a), 17(a) and 17(b) | Three representative historical trajectories | v0.1.0 included three different JSON examples. The figure bundle includes the original HTMLs used for the exported vectors. |
| Virtual-source placement and sensing | VOAPF 2.0 places a new source at the midpoint to the first obstacle boundary along the goal ray. The export describes a 240° LiDAR field of view, 683 rays and 5.6 m effective range. | The archived default S2R implementation places sources at the trapped position, samples an observation lattice with 10 m spacing and uses a 30 m physical repulsion threshold. The opt-in manuscript profile implements the midpoint and scan parameters for new runs. |
| Fixed Level 1 map | Fixed non-overlapping obstacles | The June DDPG HTMLs establish a fixed layout with overlapping obstacles. The scene tool requires a verified caller-supplied Level 1 map. |
| Randomized baseline maps | Level 2 excludes obstacle overlap and Level 3 permits it. The export specifies the same obstacle dimensions across methods, a new scene each episode and seed `123 + (m - 1)` for episode `m`. | The archived default S2R simulator randomizes maps every two episodes and its historical baselines use independent resets. The new SAC, DDPG and PG-ITD3 opt-in modes share the seeded Level 2 and 3 scenes. |
| Collision outcomes | The manuscript describes terminal collisions in the baseline protocol. | The historical 480 Hz run logs obstacle contacts while allowing goal arrival in the same episode. Mixed case [1] has 809 successful episodes with nonzero contact counts out of 873 successful episodes. The hardware-rate panels therefore represent logged goal arrival. |

## Verified figure sources

The E-drive events in the [curve manifest](../data/manuscript/data/manifest.csv) match the 17 TensorBoard CSVs packaged on C by event hash and all 16,994 episode, wall-time and binary outcome values. Twenty plotted series reuse those exports. Six source runs end at episode 999 and the others at 1000. The three original HTMLs match their E-drive reports byte for byte. No matching JSON was found for these three HTMLs. [Build commands and source hashes](../data/manuscript/README.md)

| Panel | Archive source | Episode estimate | What the HTML records |
| --- | --- | ---: | --- |
| 16(a) | 6 August `RL+VOAPF2.1_5x5_dynamic6_V=1` | 1 | 764 drone points, 20 static and six moving obstacles, six motion tracks |
| 17(a) | 30 April `RL+VOAPF2.0_S2R_V=1` | 106 | 711 drone points, 26 static obstacles and three virtual sources |
| 17(b) | 5 May `RL+VOAPF2.0_S2R_V=4` | 531 | 1,660 drone points, 26 static obstacles and five virtual sources |

Episode estimates come from report ordering and log timing. The HTMLs establish geometry without timing or controller settings. The `VOAPF2.1` folder label records provenance for Figure 16(a). All 16 rebuilt PDF page-content streams matched those in the supplied author-final vector PDFs byte for byte.

## Fixed-map evidence

The 6 June DDPG archive labeled `fixed_nonoverlap` contains 1,000 HTMLs. The 26 obstacle surfaces in `771e551b-541a-4b96-a65d-90341c719c5f.html` match the E-drive DDPG and TD3 `config/buildings.xml` centers and radii with the report's 50 m height override. Five reports sampled from the beginning, middle and end have identical obstacle geometry. The matched map has 14 overlapping circle pairs and 20 pairs that fail the manuscript's 2 m axis-aligned clearance rule.

The 3 June TD3 archive has score logs, event logs and weights. The surviving TD3 source copy shares the DDPG obstacle XML. The run's actual scene remains unverified because its archive lacks a saved map, HTML or scene JSON. The 20 August SAC archive labeled `5x5_fixed` has different obstacle centers in its first, middle and last sampled HTMLs. The default [26-obstacle dimensions](../config/buildings.xml) have a maximum radius of 45.807 m, compared with the manuscript's 45.506 m. The older [legacy SAC XML](../baselines/sac-legacy/config/buildings.xml) reaches 45.506 m with only 10 obstacles. These sources leave the manuscript's fixed non-overlapping Level 1 map unverified.

## Run the opt-in simulation

Run this two-second static Level 3 check from a clean checkout with the [simulator dependencies](../README.md#quick-start) installed.

```powershell
python simulator.py --controller-profile manuscript --map-level 3 --hardware-hz 0 --scene static --uav-speed 1 --episodes 1 --max-seconds 2 --no-train
```

Select `--map-level 2` for non-overlapping randomized maps. For a fixed Level 1 run, use `--map-level 1 --fixed-map-xml PATH` with a verified non-overlapping map. The static SAC simulator exports scene metadata as JSON. Its Level 2 and 3 plans are shared by the [DDPG and PG-ITD3 entry points](../baselines/README.md#run-the-baselines-on-shared-scenes). Connecting new runs to the historical results would require each run's source revision, complete configuration, exact scene schedule, initial model state and environment. Exact continuation also needs its replay buffer and random-number generator state. The elapsed times and sensor rates describe simulation, with separate evidence required for hardware validation.

## Author review of the manuscript

- State the initialization correction used by the final vectors. They plot `W_m / (1 - 0.99^m)`, with `W_0 = 0` and `W_m = 0.99 W_(m-1) + 0.01 X_m`. The exported equation gives the uncorrected recurrence. Update the abstract, highlights, results, tables, discussion and conclusion from the verified endpoints. For example, Figure 14(a) ends at **51.52%** versus **53.9%** in the text, and Figure 14(h) at **95.15%** versus **95.3%**. DDPG + APF Level 3 ends at **50.02%** versus **51.0%** in the results and **45.6%** in the discussion. [Full endpoint comparison](release-checklist.md#manuscript-text-cross-check)
- Describe **95.15%** as the overlapping Level 3 baseline result and **86.79%** as the 480 Hz mixed result. Keep the hardware-rate goal-arrival definition explicit when discussing collision-terminal baseline results.
- Resolve the reused `r02` outcome series cited under different labels in Figures 14(d) and 15(a) Level 2, and report its 999 observed episodes. Figure 16(a) also needs confirmation of its controller version and timing if the caption attributes those settings to that exact trajectory.
