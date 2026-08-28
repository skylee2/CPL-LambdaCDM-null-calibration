# CPL LambdaCDM Null Calibration

## Overview

This repository supports the manuscript:

> “Apparent Dynamical-Dark-Energy Exclusions and Weak-Mode Amplification in a Calibrated LambdaCDM Ensemble”

It contains compact realization-level derived products, the final nine artwork PDFs, a deterministic representative posterior subsample, and the production analysis code needed to inspect the reported ensemble calibration. It provides compact realization-level derived products, the final nine artwork PDFs, a deterministic representative posterior subsample, and the production analysis code used to inspect and reproduce the reported ensemble calibration.

## Scientific purpose

One hundred coherent BAO + compressed-CMB + SNe mock realizations are generated from a single exact spatially flat LambdaCDM cosmology. Every matched probe triplet is analyzed with a six-parameter CPL posterior, and the nested LambdaCDM and CPL models are also optimized independently to obtain realization-level likelihood-ratio statistics. The experiment measures the repeated-sampling behavior of apparent CPL displacements under the exact LambdaCDM null for the adopted compressed likelihood.

## Main reported results

- The generating point lies inside the primary direct joint HPD region in 64/100 realizations at 68.27% and 96/100 at 95.45%.
- The four primary 95.45% HPD exclusions are N100-21, N100-61, N100-65, and N100-87.
- Empirical LR quantiles are 2.548, 4.426, 5.529, and 9.187 at 68%, 90%, 95%, and 99%, respectively.
- The largest LR statistic is 16.411 for N100-65. Its joint-optimum probe contributions are BAO 9.798, CMB 4.857, and SNe 1.755.
- The median CPL covariance-eigenvalue ratio is 68.05, corresponding to a median principal-axis ratio of 8.25.

The complete manuscript comparison is in `manuscript_support/reported_values.md`.

## Repository structure

- `config/`: generating cosmology, exact open priors, random-stream mapping, and likelihood conventions.
- `data/`: compact 100-row derived tables plus a 160,000-draw representative subsample used only for Figures 6 and 8.
- `scripts/`: release-table export, figure regeneration, fail-fast consistency audit, and copies of the final production pipeline.
- `figures/`: the final PDU artwork (`Fig1a.pdf`, `Fig1b.pdf`, and `Fig2.pdf` through `Fig8.pdf`).
- `manuscript_support/`: reported-value cross-check and release notes.

## Reproduction

### Inspect the released results

From the repository root:

```bash
python scripts/audit_scientific_consistency.py
python scripts/make_figures.py --all
python scripts/make_figures.py --check --output-dir figures/generated
```

The first command verifies the frozen numerical landmarks directly from the compact CSV files. Figure output is presentation-level reproducibility: PDF metadata, fonts, and crop boxes can vary by platform, while the numerical checks are invariant.

### Re-export compact tables from a production archive

If the frozen production products are available:

```bash
python scripts/export_release_tables.py \
  --per-realization /path/to/Main_N100_Final_N0_N99_PerRealization.csv \
  --ensemble-summary-json /path/to/Main_N100_Final_N0_N99_EnsembleSummary.json \
  --output-dir data
```

This is a column-preserving export and deterministic eigenmode-orientation step. It does not rerun any inference.

### End-to-end production

The actual final pipeline is preserved under `scripts/production/`. An end-to-end rerun requires separately obtained external survey inputs, CLASS/classy 3.3.4, rebuilding or verifying the supplied early-universe grid, generating the mock HDF5 file with master seed `2507013802`, running 100 potentially long MCMC jobs with sampler seed `20260800+i`, and finally running `src/inference/analyze_main_n100_final.py`. The analyzer reads every unthinned retained production sample and repeats 200 nested-model optimizations. This is intentionally not represented as a one-command workflow; see `scripts/README.md` and `scripts/production/README.md`.

## External data

DESI DR2 BAO source information, the Planck 2018 distance-prior source, and the Pantheon+ Hubble-diagram data and covariance must be obtained separately from their public providers. None of the complete official likelihoods or survey source products is redistributed here. Exact adopted conventions and references are documented in `config/analysis_conventions.md` and `data/README.md`.

## Data-volume note

The complete 246.4 million unthinned retained posterior samples (about 40.5 GB of N100 chain backends) are not included. The release instead provides all 100 realization-level posterior centers, coverage classifications, optimized statistics, probe decompositions, weak-mode quantities, pivots, and optimizer diagnostics. A deterministic 5.2 MB subsample from four representative posteriors is included solely to reconstruct the contour artwork; it is not used for the ensemble summary statistics.

The 100 mock observable vectors are also omitted. They are exactly regenerable using the released seed logic and production procedure once the public external inputs are supplied, and the original compact HDF5 container embeds a selected Pantheon+ covariance/redshift structure that this repository does not redistribute.

## Requirements

`requirements.txt` records the exact Python-package versions used in the verified environment. CLASS/classy 3.3.4 requires a compatible CLASS installation and is not listed as a misleading ordinary pip dependency. Figure inspection and compact-table audits do not invoke CLASS.

## Citation

Citation metadata is in `CITATION.cff`. A Zenodo identifier will be added after public release.

DOI: to be assigned

## AI-assisted development

Selected coding, review, documentation, and figure-workflow tasks used
generative-AI assistance under author verification; see
[AI_ASSISTANCE.md](AI_ASSISTANCE.md).

## License

- Code and scripts are released under the [MIT License](LICENSE).
- Derived data products, configuration summaries, scientific figures, and repository documentation are released under [CC BY 4.0](LICENSE-DATA.md).
- Third-party DESI, Planck, Pantheon+, and other observational inputs are not redistributed and remain subject to their original providers' licenses and terms.
