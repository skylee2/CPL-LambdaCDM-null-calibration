# Scripts

## Release-facing scripts

- `audit_scientific_consistency.py` performs the mandatory fail-fast audit against the final manuscript landmarks.
- `make_figures.py` regenerates all nine artwork PDFs from the compact data. Figures 6 and 8 alone consume `representative_samples.npz`; the other figures use CSV tables.
- `export_release_tables.py` mechanically exports the public tables from the frozen final per-realization production CSV and matching ensemble-summary JSON.

These three scripts resolve paths relative to this repository and contain no private absolute paths.

## Production source

`production/` preserves the actual final scientific pipeline and its package structure:

- mock streams and matched BAO+CMB+SNe generation;
- CLASS grid construction and shared cosmological theory;
- probe and joint likelihoods;
- restart-safe emcee execution and convergence diagnostics;
- final all-realization summary extraction, direct HPD construction, coverage, nested-model optimization, probe decomposition, weak-mode analysis, and the original production figure/report writers.

Only path configuration and scan-neutral operator whitespace were sanitized for the release. Numerical methods and thresholds were not changed. The compact public figure writer is separate because it reads release tables rather than the omitted 40.5 GB backend archive.

Historical/interim scripts, pilot-only optimization drivers, cleanup utilities, obsolete duplicate figure writers, and unrelated Pedagogic2 analyses were intentionally omitted. In particular, the older manuscript figure writer used a nonproduction representative-contour cap/grid and is not the final HPD authority.
