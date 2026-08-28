# Final reported values

Source of truth: final PDU manuscript, “Apparent Dynamical-Dark-Energy Exclusions and Weak-Mode Amplification in a Calibrated LambdaCDM Ensemble.” Values below preserve the manuscript's reported rounding.

## Generating cosmology and ensemble

- Realizations: 100 (`N100-0` through `N100-99`).
- `(omega_m, omega_b, H0, w0, wa, DeltaM) = (0.147, 0.02237, 70, -1, 0, 0)`.
- Derived `Omega_m0 = 0.3`.

## Ensemble recovery (posterior median per realization)

| Parameter | Generating | Ensemble mean | Bias | SE(bias) | RMSE |
|---|---:|---:|---:|---:|---:|
| omega_m | 0.147 | 0.146965 | -0.000035 | 0.000107 | 0.001069 |
| H0 | 70 | 70.04847 | +0.04847 | 0.06852 | 0.68344 |
| Omega_m0 | 0.3 | 0.299575 | -0.000425 | 0.000610 | 0.006089 |
| w0 | -1 | -0.998820 | +0.001180 | 0.005316 | 0.052910 |
| wa | 0 | -0.015076 | -0.015076 | 0.020635 | 0.205867 |
| wp | -1 | -1.001248 | -0.001248 | 0.002553 | 0.025430 |

Posterior-mean ensemble averages in the order `(H0, Omega_m0, w0, wa, wp)` are `(70.05175, 0.299633, -0.998103, -0.023529, -1.001348)`.

Median-center pull means in the order `(omega_m, H0, Omega_m0, w0, wa, wp)` are `(-0.015, 0.061, -0.093, -0.007, 0.019, -0.039)`; pull widths are `(1.096, 1.017, 1.034, 1.022, 1.079, 1.011)`.

## Coverage

- Primary direct HPD: 64/100 at 68.27%; 96/100 at 95.45%.
- Gaussian covariance ellipse: 66/100 at 68.27%; 96/100 at 95.45%.
- Primary 95.45% HPD exclusions: N100-21, N100-61, N100-65, N100-87.
- Marginal 68.27% / 95.45% counts: `w0` 68/94, `wa` 63/94, `wp` 70/95, `H0` 71/93, `Omega_m0` 65/94.

Across nine HPD grid/smoothing variants, 95.45% coverage ranges from 92/100 to 96/100 and the same four primary exclusions persist in every variant.

## Optimized LR calibration

- Empirical linear quantiles: `Q68=2.548`, `Q90=4.426`, `Q95=5.529`, `Q99=9.187`.
- Fixed-threshold exceedances: `Delta chi2_LR >= 2.30`: 36/100; `>= 6.18`: 4/100; `>= 9.21`: 1/100.
- Maximum: `Delta chi2_LR=16.411` for N100-65.
- For N100-65, statistic A is 24.349 and `Delta chi2_A - Delta chi2_LR = 7.938`.

## Primary exclusions and probe contributions

| ID | Delta chi2 A | Delta chi2 LR | BAO | CMB | SNe | CPL optimum (w0, wa) | Classification |
|---|---:|---:|---:|---:|---:|---|---|
| N100-21 | 19.691 | 8.175 | 1.302 | 0.275 | 6.599 | (-0.854, -0.413) | SNe dominated |
| N100-61 | 10.449 | 7.559 | -1.018 | 1.650 | 6.927 | (-0.924, -0.029) | SNe dominated |
| N100-65 | 24.349 | 16.411 | 9.798 | 4.857 | 1.755 | (-1.066, +0.496) | BAO+CMB coherent |
| N100-87 | 16.613 | 9.114 | -1.065 | 3.918 | 6.261 | (-1.028, -0.213) | SNe dominated |

N100-65 posterior median: `(w0, wa)=(-1.0623, 0.4777)`. Optimized backgrounds: LambdaCDM `(H0, Omega_m0)=(69.6401, 0.302455)` and CPL `(68.3819, 0.307197)`.

## Realization-level center correlations

- `r(H0, Omega_m0) = -0.934`.
- `r(w0, wa) = -0.882`.
- `r(H0, wp) = -0.868`.
- `r(Omega_m0, wp) = 0.69`.
- `r(Omega_m0, w0) = 0.57`.

These are correlations among the 100 realization-level posterior medians, not within-posterior parameter correlations.

## Weak mode and pivot

- Median covariance eigenvalue ratio `lambda_long/lambda_short = 68.05`.
- Central 16--84% eigenvalue-ratio range: `58.36--78.80`.
- Median principal-axis (standard-deviation) ratio: `8.25`.
- Realization-specific pivot-redshift median: `z_p=0.297`.
- Central 16--84% pivot-redshift range: `0.280--0.319`.
- Full pivot-redshift range: `0.263--0.412`.

## Numerical diagnostics

All 100 chains converged and all 200 independent optimization solutions satisfied the numerical criteria. The largest difference between the two best independent optimization starts was below `8e-10`. The maximum split statistic was `Rhat=1.00525`, the smallest per-chain ESS was 14,646, and no realization was excluded from analysis.
