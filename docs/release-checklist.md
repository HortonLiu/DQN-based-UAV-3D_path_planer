# Artifact completeness and release checklist

## Repository contents

The repository includes the current simulator, three historical source snapshots with their XML files, regression tests, pinned dependencies, two smoke-run examples, selected archived results and an original plot generated from archived data. [Repository overview](../README.md)

Local database settings, unreviewed checkpoints, legacy illustrations, raw output directories, and manuscript working files are omitted from the current artifact tree. The Git ignore rules cover generated files and local database settings. The RLGF copyright, upstream fork history, and third-party notices identify the source of reused code.

## Archived evidence

The source research archive contains 92 CSV files, 122 model files, 41 TensorBoard event files, 4,001 trajectory JSON files, and 40,697 HTML reports. It is approximately 134.87 GB, with 118.25 GB in HTML reports. Three SAC training runs have 3,000 score records with corresponding HTML and JSON reports, and three complete checkpoint sets verified with the current CPU simulator. The repository includes these score series, a representative trajectory for each, and 7,000 outcomes extracted from seven DDPG and TD3 runs. [Curated data and provenance](../data/historical/README.md)

## Files needed for manuscript reproduction

| Artwork or result | Missing source material |
| --- | --- |
| Figure 13, methods comparison | Mapping from each curve to its original run, dates, source revision, XML, initial state, and verified per-episode outcomes |
| Figure 14, environment-complexity comparison | Run and map identifiers, scenario settings, raw outcomes, and plotting inputs |
| Figure 15, trajectory and mixed/static success curves | Exact trajectory UUID, motion endpoints, mixed and static run identifiers, and plotting parameters for the reported final values of 98.1% and 95.3% |
| Figure 16, asynchronous sensing and displacement | The representative trajectory UUIDs, their experiment configurations, and the plot generation settings for the two baseline-equivalent displacement settings |
| Exact retraining and continuation | Initial and paper-specific checkpoint sets, replay buffer, RNG state, seed, run command, software versions, and repetitions |
| Figure generation | Original statistical definitions, EMA initialization, plot scripts or source projects, and mapping of 12 historical PNGs to the manuscript |

Of 42 early score CSV files, 40 have zero values throughout the score and outcome fields and two lack usable rows. TensorBoard and validated per-UAV logs are candidate sources. Archived TD3 runs are labeled `+VOAPF` while the supplied TD3 source directory is labeled `+APF`. One DDPG record also has inconsistent outer and inner folder labels. Confirm source-to-run mapping before using these for comparisons. [Data descriptions](../data/historical/README.md)

## Versioned archive and DOI

Version `v0.1.0` was deposited in Zenodo as a Software record containing the GitHub release's source ZIP. The DOI identifies that fixed version and its included simulation data. Future releases can use GitHub integration once repository synchronization is available. [Zenodo version DOI](https://doi.org/10.5281/zenodo.24494002) · [GitHub v0.1.0](https://github.com/HortonLiu/DQN-based-UAV-3D_path_planer/releases/tag/v0.1.0)

1. Review the manuscript author metadata in `CITATION.cff`. The research software uses MIT, the packaged simulation data use CC BY 4.0, and bundled third-party material retains its license notices. Confirm licensing separately for any model files or additional results before depositing them. [Software license](../LICENSE) · [Simulation data license](../data/LICENSE)
2. Connect GitHub in Zenodo, synchronize the repositories, and enable this repository. If both `CITATION.cff` and `.zenodo.json` exist at release time, Zenodo uses `.zenodo.json` for release metadata. [Repository integration](https://help.zenodo.org/docs/github/enable-repository/) · [Software metadata](https://help.zenodo.org/docs/github/describe-software/)
3. Publish a tagged GitHub Release and verify the archived files, creators, and version DOI in Zenodo. The GitHub import reads the MIT software license from `CITATION.cff`. Add CC BY 4.0 to the archive record for the bundled simulation data and verify both license entries. Add the version DOI to the README. [Archive a GitHub release](https://help.zenodo.org/docs/github/archive-software/github-upload/) · [Mixed-license records](https://help.zenodo.org/docs/deposit/describe-records/licenses/#mixed-license-uploads)
4. For a separate data deposit, bundle approved model and result files by run and attach a SHA-256 manifest. CSV, JSON and model files in the source archive total about 16.61 GB before compression. Include HTML as needed for figure provenance and verify the contents of each deposit. Link the software and dataset records by DOI. [Zenodo file limits](https://help.zenodo.org/docs/deposit/manage-files/) · [GitHub large-file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
