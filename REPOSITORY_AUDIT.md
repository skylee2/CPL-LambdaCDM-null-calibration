# Repository audit

Audit scope: local package `release/CPL-LambdaCDM-null-calibration`. This preparation did not initialize Git, connect a remote, commit, push, alter repository visibility, or create a release.

**Original project files modified: NO**

## Included files

- Top-level metadata: `README.md`, `CITATION.cff`, `requirements.txt`, `.gitignore`, `LICENSE`, `LICENSE-DATA.md`, and this audit.
- Four human-readable configuration/convention files under `config/`.
- Eight principal compact CSV products, one four-row representative-point CSV, a 5.2 MB deterministic representative posterior subsample, its metadata, and `data/README.md`.
- Three release-facing scripts plus the final production source package, sanitized production configuration, and the 6.6 KB CLASS interpolation grid.
- Nine final PDU artwork PDFs: Figure 1 panels `Fig1a.pdf` and `Fig1b.pdf`, plus `Fig2.pdf` through `Fig8.pdf`.
- Two manuscript-support Markdown files.

No manuscript source/PDF, raw chain backend, full mock HDF5, cache, build by-product, private note, or unrelated project file is included.

## Derived-product provenance

| Release product | Read-only production source | Export/derivation |
|---|---|---|
| `ensemble_summary.csv` | `output/Main_N100_Final_N0_N99_EnsembleSummary.json` and matching per-realization CSV | Final median-estimator recovery/pull records; six manuscript parameters |
| `lr_statistics.csv` | `output/Main_N100_Final_N0_N99_PerRealization.csv` | Direct column export of statistic A and independently optimized LR |
| `probe_contributions.csv` | same | Direct export of joint-optimum BAO/CMB/SNe components and total |
| `posterior_centers.csv` | same | Direct export of posterior means/medians for sampled and derived parameters |
| `coverage_results.csv` | same | Direct export/rename of marginal and joint membership flags |
| `weak_mode_summary.csv` | same covariance JSON and posterior means | `numpy.linalg.eigh`; eigenvectors oriented as specified in the final PDU; invariant ratios unchanged |
| `pivot_summary.csv` | same | Direct export of realization-specific pivot and `w_p` fields |
| `optimization_diagnostics.csv` | same | Direct export of minima/success/restart spread plus documented deterministic seed mapping |
| `representative_points.csv` | same | Four predeclared Figure 6/8 cases |
| `representative_samples.npz` | four final read-only chain backends | Existing deterministic 40,000-row equal-stride selection per case; upstream hashes retained |

`scripts/export_release_tables.py` records the mechanical CSV export. It does not recompute sampling, minima, HPD regions, or scientific conclusions.

## Scripts included and omitted

Included production code is the final `src/inference/analyze_main_n100_final.py` lineage plus its actual theory/likelihood dependencies, matched-mock generator and random streams, MCMC runner/diagnostics, and CLASS-grid builder. The analyzer is included because it is the authority for summary extraction, direct HPD construction, coverage, independent nested-model optimization, probe decomposition, weak modes, pivots, and original production figures. The compact `make_figures.py` is a release-facing adaptation that reads only released products and uses the newly finalized Figure 1 legend text.

Omitted as obsolete, redundant, or out of scope: interim N0--N66 analysis; pilot-selection/optimization reports beyond the one small runner dependency; historical manuscript writers with nonproduction contour settings; standalone optimizer prototypes superseded by the analyzer's final repeated-start implementation; cleanup/audit utilities; observational-comparison placeholders; and all unrelated Pedagogic2 scripts.

## Licensing and source-code authorship

- Source code and scripts: MIT License, copyright 2026 Seokcheon Lee (`LICENSE`).
- Derived data, configuration summaries, scientific figures, and repository documentation: Creative Commons Attribution 4.0 International (`LICENSE-DATA.md`).
- Third-party observational products are excluded from both grants and are not redistributed.

Every included Python file was checked against its project provenance before headers were added. Twenty-three production files match the original project `src/` files byte-for-byte, one production release copy differs only by documented scan-neutral operator whitespace, and three scripts were created specifically for this release. No included Python file carried a third-party copyright/license header and none had uncertain provenance.

- Python files receiving the Seokcheon Lee/MIT header: 26.
- Python files retaining pre-existing third-party headers: 0.
- Python files left unchanged because provenance was uncertain: 0.

Header changes did not alter imports, executable behavior, numerical methods, or scientific content.

## Third-party and large data intentionally excluded

- Full DESI and Planck products/likelihoods and source chains.
- Pantheon+ source checkout, observed residuals, covariance, redshifts/identifiers, Cepheid/SH0ES calibration, and photometry.
- Original 19 MB mock HDF5 because it embeds the selected Pantheon+ covariance/redshift structure and the mock vectors are exactly regenerable from public inputs, released code, and seeds.
- All 100 MCMC backends and 246.4 million unthinned retained posterior draws (about 40.5 GB).

The 5.2 MB representative NPZ is a compact derived subset required for Figures 6 and 8, not a raw backend or ensemble-statistics input.

## Scientific consistency audit

`scripts/audit_scientific_consistency.py` passed. It verified:

- exactly 100 ordered IDs in every realization-level CSV;
- Table-IV recovery, biases, errors, RMSEs, pull means, and widths;
- key realization-level correlations;
- marginal coverage plus direct HPD counts 64/100 and 96/100;
- exactly N100-21, N100-61, N100-65, N100-87 as primary HPD95 exclusions;
- LR quantiles 2.548, 4.426, 5.529, 9.187 and maximum 16.411 at N100-65;
- threshold counts `>=6.18`: 4 and `>=9.21`: 1;
- N100-65 components 9.798, 4.857, 1.755 and their full-precision sum;
- eigenvalue-ratio median 68.05, axis ratio 8.25, and reported central range;
- pivot median/ranges and all optimizer success/restart-agreement criteria.

All nine packaged PDFs are byte-identical to the final PDU artwork. A clean temporary regeneration of all nine figures completed and the release figure checker passed.

**Scientific consistency audit passed: YES**

## Security and privacy audit

Recursive text-pattern scans covered email addresses, local absolute paths, user-home paths, credential markers, private-key material, private hostnames, and temporary path literals. JSON metadata and the NPZ member inventory were separately inspected. Results:

- no email address is present;
- no private user-home path, credential, or private-hostname material is present;
- the only username-like string is the explicitly required public GitHub repository owner in `CITATION.cff`;
- upstream representative backend locations are repository-relative and accompanied only by scientific SHA-256 provenance;
- no cache, `.aux`, `.log`, `.out`, `.synctex.gz`, `.DS_Store`, or compiled Python file remains.

**Security/privacy audit passed: YES**

## Phase 2 pre-Git safety audit

The final pre-initialization audit was rerun on 2026-08-28 after adding only licensing, authorship headers, and repository metadata. Scientific consistency and figure landmark checks passed. `CITATION.cff` parsed successfully and contains the required title, author, and repository URL without an ORCID or DOI. Recursive scans found no private absolute path, email address, credential material, cache/build artifact, editor temporary, local environment, raw chain/mock HDF5, or third-party observational source file. The package still contains nine artwork PDFs and nine compact CSV files; its largest file is 5,494,779 bytes.

**Final pre-Git safety audit passed: YES**

## Size audit

- Total files: 68.
- Exact total payload: `6051866` bytes.
- Files larger than 10 MB: none.
- Files larger than 50 MB: none.
- GitHub-unsuitable files: none. The largest file is the 5,494,779-byte representative NPZ; Git LFS is not required.

Largest 20 files (bytes):

```text
5494779 data/representative_samples.npz
52486 scripts/production/src/inference/analyze_main_n100_final.py
51142 figures/Fig6.pdf
47461 figures/Fig8.pdf
32883 data/posterior_centers.csv
26689 figures/Fig1b.pdf
25922 scripts/production/src/inference/run_joint_mcmc.py
24228 figures/Fig1a.pdf
23674 figures/Fig7.pdf
22670 figures/Fig3.pdf
21666 figures/Fig4.pdf
18776 data/pivot_summary.csv
16767 figures/Fig5.pdf
15595 data/optimization_diagnostics.csv
15558 figures/Fig2.pdf
14596 data/weak_mode_summary.csv
14350 scripts/make_figures.py
10209 data/probe_contributions.csv
9536 scripts/export_release_tables.py
8858 scripts/production/src/generation/generate_joint_mock_triplets.py
```

## Unresolved pre-publication items

1. Add the Zenodo DOI and any publication metadata only after assignment.
2. Human-review the external-input acquisition/checksum instructions and provider filenames before an end-to-end rerun.
3. Human-review regenerated figure appearance across the intended public environment; numerical checks already pass.

The content is scientifically and privacy-safe for human review. It can be made public after the remaining metadata decisions and final human approval.

**Ready for human review before GitHub push: YES**

## GitHub private-repository upload

- Repository URL: `https://github.com/skylee2/CPL-LambdaCDM-null-calibration`
- Branch: `main`
- Initial commit hash: `35b22814aa509f1029f35fd5daeed4aac760d240`
- Push success: YES
- Remote/local HEAD match: YES (verified immediately after the initial push)
- Expected remote contents: VERIFIED (67 tracked files, including the required top-level metadata and directory structure)
- Remote figure artwork: VERIFIED (nine PDF files)
- Repository visibility status: PRIVATE (authenticated push succeeded; anonymous repository access returned HTTP 404)
- Zenodo status: NOT YET CONNECTED
- Scientific content changed during Phase 2: NO

## AI-assisted development disclosure

`AI_ASSISTANCE.md` and the corresponding concise README link were added before
public release. This documentation-only addition changes no scientific data,
numerical result, production-code behavior, configuration, or figure.
