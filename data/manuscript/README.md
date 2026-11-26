# Manuscript Figures 14–17

These inputs and plotting scripts accompany the author-final manuscript export of 26 November 2026. They reproduce 13 success-rate panels and three trajectory panels. The [v0.1.0 archive](https://doi.org/10.5281/zenodo.24494002) contains the simulator and selected historical results. This bundle provides the additional figure-specific inputs.

| Panel | Inputs |
| --- | --- |
| 14(a–h) | Eight success-rate curves for APF and the SAC variants |
| 15(a–c) | Eight curves comparing SAC + VOAPF 2.0, PG-ITD3 and DDPG + APF across map complexity levels |
| 16(b), 17(c) | Four hardware-rate success curves across three runs. The static 1 m/s series is shared |
| 16(a), 17(a–b) | Three original ECharts trajectory reports |

## Rebuild the panels

Use Python 3.12 or newer in a separate environment for plotting. From the repository root in PowerShell:

```powershell
$figureEnv = Join-Path $env:TEMP 'uav-manuscript-figures'
py -3.12 -m venv $figureEnv
$figurePython = Join-Path $figureEnv 'Scripts/python.exe'
& $figurePython -m pip install -r data/manuscript/requirements.txt
& $figurePython data/manuscript/rebuild_plots.py --output dist/manuscript/curves
& $figurePython data/manuscript/trajectory/rebuild_trajectories.py --output dist/manuscript/trajectories
```

The scripts produce vector PDFs, PNG previews and validation reports. Success panels use artwork numbers 18–28, 30 and 33. Trajectory panels use artwork numbers 29, 31 and 32. The trajectory script parses the ECharts option data from the HTML locally, with no browser or JavaScript dependency. PDF creation timestamps may vary between runs.

Verification on Windows with Python 3.12.14 and pypdf 7.1.1 reproduced the PDF drawing content of all 16 panels byte for byte against the author-final vector artwork, along with the 20 curve endpoints and three trajectory-coordinate hashes.

## Success-rate inputs and smoothing

The [curve manifest](data/manifest.csv) lists the figure and panel, archive run, source TensorBoard event SHA-256, actual episode range and CSV pair for all 20 curves. The 17 native exports preserve `Wall time`, `Step` and the original binary `Train/Success_Rate` values. All 16,994 outcome rows were checked against the matching archived event files. Six runs end at episode 999 and the others at 1000.

The vector success plots use the raw binary outcome `X_m`, with `W_0 = 0` and `β = 0.99`. The displayed series uses the TensorBoard-style initialization correction:

```text
W_m = β W_(m-1) + (1 - β) X_m
S_m = W_m / (1 - β^m)
```

The script validates the episode sequence, outcomes and recomputed values against the curve CSVs. It writes the full-precision endpoints and the percentages rounded to two decimal places. The CSVs also retain an uncorrected `β = 0.99` column and a corrected `β = 0.999` reference column for the earlier artwork. The corrected `β = 0.99` vector endpoints can differ from values in the manuscript body. For example, Figure 14(a) gives 51.52% in the vector reconstruction and 53.9% in the body text. The earlier `β = 0.999` reference for that curve gives 53.93%.

Three series are reused across panels. Figure 14(d) and Figure 15(a) Level 2 share `r02`. The latter is identified as SAC + VOAPF 2.0 by the author and stored under a legacy `RL+APF` archive name. Figure 14(h) and Figure 15(a) Level 3 share `r29`. Figure 16(b) and Figure 17(c) share case [2], `r40`. Each reused sequence is one run.

The manifest retains the delivered figure labels and archive names. For Figure 15(c) Level 1, the archived `r24` TensorBoard directory carries `+VOAPF`, as shown in `Event_Dir_Relative_to_Data`. Its event hash matches the manifest. The panel label is DDPG + APF.

## Trajectory inputs

| Panel | Archived run | Episode | Source HTML | Path samples |
| --- | --- | ---: | --- | ---: |
| 16(a) | `08_06_2026(11_59_08)_RL+VOAPF2.1_5x5_dynamic6_V=1` | 1 | [95201273…](trajectory/experience_html/95201273-ff28-4956-acb9-8187fd9e8a32.html) | 764 |
| 17(a) | `04_30_2026(22_45_53)_RL+VOAPF2.0_S2R_V=1` | 106 | [0f2fb3ad…](trajectory/experience_html/0f2fb3ad-448a-4c82-91ad-3965703aa3af.html) | 711 |
| 17(b) | `05_05_2026(13_14_51)_RL+VOAPF2.0_S2R_V=4` | 531 | [0acd1113…](trajectory/experience_html/0acd1113-3e42-4835-a4cb-3bdbea617ef7.html) | 1,660 |

The HTML files match the archived reports byte for byte. The episode numbers were inferred from file order and experiment log timing. The `VOAPF2.1` folder name for Figure 16(a) records the source location. Each trajectory panel uses its own historical run. The 480 Hz success curves in Figures 16(b) and 17(c) use the October–November runs listed in the curve manifest. The original HTML series contain coordinates without sample timestamps. They verify the shown geometry. Confirming the original integration rate and controller configuration requires additional run records.

The trajectory script preserves path, track and surface vertices and uses an orthographic view at 78° elevation and −45° azimuth. The red frame in Figure 17(b) is a figure annotation. [Source hashes, geometry and render settings](trajectory/trajectory_validation.json)

## Licenses

The simulation outcomes and recorded trajectory data use [CC BY 4.0](../LICENSE). Attribute the data to Ming-Chieh Liu and Chen-Yang Lan. The plotting scripts use the [MIT software license](../../LICENSE). The original HTML references Apache ECharts and ECharts GL, which retain their [third-party licenses](../../assets/pyecharts/).
