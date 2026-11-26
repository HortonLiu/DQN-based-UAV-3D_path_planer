# Artifact completeness and release checklist

## Repository contents

The repository includes the current simulator, three historical source snapshots with their XML files, regression tests, pinned dependencies, two smoke-run examples, selected archived results, and the inputs and scripts for manuscript Figures 14–17. [Repository overview](../README.md)

Local database settings, unreviewed checkpoints, legacy illustrations, raw output directories, and manuscript working files are omitted from the current artifact tree. The Git ignore rules cover generated files and local database settings. The RLGF copyright, upstream fork history, and third-party notices identify the source of reused code.

## Archived evidence

The source research archive contains 92 CSV files, 122 model files, 41 TensorBoard event files, 4,001 trajectory JSON files, and 40,697 HTML reports. It is approximately 134.87 GB, with 118.25 GB in HTML reports. Three SAC training runs have 3,000 score records with corresponding HTML and JSON reports, and three complete checkpoint sets verified with the current CPU simulator. The repository includes these score series, a representative trajectory for each, and 7,000 outcomes extracted from seven DDPG and TD3 runs. [Curated data and provenance](../data/historical/README.md)

## Coverage of the 2026-11-26 Overleaf export

| Panels | Reproduction source | In Zenodo v0.1.0 |
| --- | --- | --- |
| Figure 14(a–h), method comparisons | Eight curves from the matching binary TensorBoard outcome exports | Figure-specific mapping and the complete inputs were added later |
| Figure 15(a–c), complexity levels | Eight curves and their run and episode ranges | Seven DDPG/TD3 outcome series are included. The figure-specific mapping and complete inputs were added later |
| Figure 16(b) and 17(c), 480 Hz success comparisons | Three underlying outcome runs, with static 1 m/s shared between panels | All three score series are included. The vector plotting workflow was added later |
| Figure 16(a), 17(a), and 17(b), representative trajectories | Three original experience HTMLs, matched to the E drive archive and author-final figure source | Three different JSON trajectory examples are included. The exact figure HTMLs were added later |

The [figure manifest and build instructions](../data/manuscript/README.md) retain the author-final panel labels and source run IDs. Figure 14(d) and Figure 15(a) Level 2 share the same binary outcome series, `r02`, although the panels display different method names. The author-final provenance records that the author confirmed the Figure 15(a) Level 2 assignment. The duplication does not establish two independently identifiable runs. Figure 14(h) and Figure 15(a) Level 3 also share a series, `r29`, under the same method label.

The Figure 16(a) HTML is from the August 6 archive folder labeled `VOAPF2.1`, while its manuscript caption places it in the VOAPF 2.0 evaluation. Its UUID and spatial data match the author-final figure source. The HTML contains positions and obstacle tracks without timestamps or controller configuration. The same limitation applies to the original static Figure 17(a/b) HTMLs. Timing evidence for the October and November outcome runs is available from their separate score and trajectory records.

The manuscript describes terminal collision in the baseline protocol. The archived mixed run records nonzero `meet_threaten` in 809 of its 873 successful episodes. The current S2R simulator rejects obstacle-intersecting movement, counts it and continues the episode unless goal or timeout terminates it. Figure 16(b) uses the logged goal-arrival outcome and the manuscript should state this scope before comparing it with collision-terminal baseline results. [Mixed run scores](../data/historical/mixed-480hz/scores.csv) · [Simulator protocol](reproduction.md#manuscript-protocol)

## Manuscript text cross-check

The author-final vector plots use initialization-corrected EMA at `β = 0.99`. The prose and tables in the same export retain several earlier percentages. The endpoints below come from the matching binary outcomes and retain full precision until display rounding.

| Figure 14 method | Text or summary table | Rebuilt vector |
| --- | ---: | ---: |
| (a) APF | 53.90% | 51.52% |
| (b) SAC only | 0.00% | 0.00% |
| (c) SAC+APF, overlapping | 61.30% | 61.48% |
| (d) SAC+APF, non-overlapping | 98.70% | 99.04% |
| (e) SAC+VAPF | 62.70% | 64.34% |
| (f) SAC+PAPF | 42.30% | 41.68% |
| (g) SAC+VOAPF 1.0 | 82.70% | 82.41% |
| (h) SAC+VOAPF 2.0 | 95.30% | 95.15% |

| Figure 15 method | Level 1 | Level 2 | Level 3 |
| --- | ---: | ---: | ---: |
| (a) SAC+VOAPF 2.0 | — | 99.04% | 95.15% |
| (b) PG-ITD3 | 89.86% | 64.72% | 55.28% |
| (c) DDPG+APF | 92.67% | 93.26% | 50.02% |

To synchronize the manuscript with these figure sources, specify the initialization correction in the EMA definition and update the abstract, highlights, results, summary tables, discussion, and conclusion using these endpoints. The manuscript gives DDPG+APF Level 3 as 51.0% in the results and complexity-level table, and 45.6% in the discussion. The highlights also describe the 95.3% baseline Level 3 result as a dynamic-environment result. The mixed run in Figure 16(b) ends at 86.79%. Refer to the randomized overlapping baseline for the Figure 14(h)/15(a) Level 3 result. Use the observed interval of episodes 1–999 for the sourced Level 2 SAC curve when describing the displayed data.

## Evidence needed for exact retraining

Link each paper run to its original source revision, experiment XML, simulator and library versions, initial weights or initialization, and map and RNG state. Checkpoint continuation also requires the matching replay buffer and RNG state. For the manuscript baseline, verify the goal-ray midpoint placement and the shared per-episode map generator across methods. For the illustrative trajectories, separate their verified spatial provenance from any controller version and integration-rate claims that require the original run configuration. [Simulation protocol](reproduction.md#manuscript-protocol)

## Versioned archive and DOI

Version `v0.1.0` was deposited in Zenodo as a Software record for GitHub commit `9e214387cbaec9aee66b468b61e46c2a011d6e50`. The GitHub tag ZIP and local archive are byte-identical and match the Zenodo file checksum (`MD5 3da477d4339b35d41040987de474c27c`). The manuscript export cites that version for its software and selected curated data. [Zenodo v0.1.0](https://doi.org/10.5281/zenodo.24494002) · [GitHub v0.1.0](https://github.com/HortonLiu/DQN-based-UAV-3D_path_planer/releases/tag/v0.1.0)

Version `v0.2.0` includes the expanded figure inputs and the manuscript simulation profile. Use its version DOI to cite these additions. Consult the alignment audit for the scope of reproducibility. [Zenodo v0.2.0](https://doi.org/10.5281/zenodo.24556113) · [Alignment audit](manuscript-alignment.md)

1. Review the manuscript author metadata in `CITATION.cff`. The research software uses MIT, the packaged simulation data use CC BY 4.0, and bundled third-party material retains its license notices. Confirm licensing separately for any model files or additional results before depositing them. [Software license](../LICENSE) · [Simulation data license](../data/LICENSE)
2. Connect GitHub in Zenodo, synchronize the repositories, and enable this repository. If both `CITATION.cff` and `.zenodo.json` exist at release time, Zenodo uses `.zenodo.json` for release metadata. [Repository integration](https://help.zenodo.org/docs/github/enable-repository/) · [Software metadata](https://help.zenodo.org/docs/github/describe-software/)
3. Publish a tagged GitHub Release and verify the archived files, creators, and version DOI in Zenodo. The GitHub import reads the MIT software license from `CITATION.cff`. Add CC BY 4.0 to the archive record for the bundled simulation data and verify both license entries. Add the version DOI to the README. [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/) · [Mixed-license records](https://help.zenodo.org/docs/deposit/describe-records/licenses/#mixed-license-uploads)
4. For a separate data deposit, bundle approved model and result files by run and attach a SHA-256 manifest. CSV, JSON and model files in the source archive total about 16.61 GB before compression. Include HTML as needed for figure provenance and verify the contents of each deposit. Link the software and dataset records by DOI. [Zenodo file limits](https://help.zenodo.org/docs/deposit/manage-files/) · [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
