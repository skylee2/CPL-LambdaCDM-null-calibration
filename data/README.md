# Data products

All CSV files are UTF-8, have a header row, and use stable realization labels `N100-0` through `N100-99`. Floating-point fields retain the precision of the frozen final production export; displayed manuscript values are rounded separately.

## Compact tables

- `ensemble_summary.csv`: Table-IV-style ensemble recovery, bias, standard error, RMSE, and median-center pull summaries for six reported parameters.
- `lr_statistics.csv`: statistic A and the independently optimized nested-model LR statistic.
- `probe_contributions.csv`: BAO, CMB, and SNe contributions evaluated at the two common joint optima; components can be negative and sum to the total LR statistic.
- `posterior_centers.csv`: realization-level posterior means and medians for sampled and derived parameters.
- `coverage_results.csv`: marginal equal-tailed and joint direct-HPD/covariance-ellipse membership classifications.
- `weak_mode_summary.csv`: CPL covariance eigenvalues, invariant ratios, and deterministically oriented standardized displacements. `d_short` uses an eigenvector with positive `w0` component; `d_long` uses one with positive `wa` component.
- `pivot_summary.csv`: realization-specific pivot scale factor/redshift and `w_p` summaries.
- `optimization_diagnostics.csv`: independent minima, model-specific random-start seeds, success flags, methods, start counts, and restart chi-square spreads.
- `representative_points.csv`: posterior and CPL-optimum points for the four explicitly selected Figure 6/8 cases.

The authoritative source for these tables was `output/Main_N100_Final_N0_N99_PerRealization.csv`; `ensemble_summary.csv` additionally retains values from its matching `Main_N100_Final_N0_N99_EnsembleSummary.json`. Both production sources were inspected read-only and are not redistributed because the compact tables contain the relevant release fields.

## Representative posterior subsample

`representative_samples.npz` contains exactly 40,000 deterministic equal-stride samples for each of N100-41, N100-80, N100-65, and N100-98, plus their selected flat indices. It is 160,000 draws in total (about 0.065% of the 246.4 million retained ensemble draws) and exists only to reproduce the direct contours in Figures 6 and 8. `representative_samples_metadata.json` records the parameter order, selection, roles, upstream backend hashes, and HPD prescription. The upstream path strings in that metadata are repository-relative and identify no private filesystem location.

## Mock-data decision

The original 100-realization mock HDF5 container is not included. Although its observable vectors are synthetic and compact, the same container also embeds the selected 1590 x 1590 Pantheon+ covariance, Hubble-diagram redshifts, and identifiers. Redistributing that embedded third-party-derived structure is unnecessary because the mocks can be regenerated exactly from:

1. the public Pantheon+ files and the documented `z_HD > 0.01` mask;
2. the documented DESI DR2 reduced BAO blocks and Planck compressed covariance;
3. `scripts/production/src/generation/generate_joint_mock_triplets.py`;
4. master entropy `2507013802` and child spawn keys `[0]`, `[1]`, `[2]`.

The release does not include observed Pantheon+ residuals, Cepheid/SH0ES calibration, full DESI products, full Planck likelihoods/chains, or any third-party survey source tree.
