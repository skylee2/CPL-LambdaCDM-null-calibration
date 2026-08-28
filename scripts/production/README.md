# Final production pipeline

This directory is a compact copy of the final code lineage. Its internal layout is deliberately retained so imports such as `from src.likelihoods...` continue to resolve when commands are run from this directory.

## External inputs

Place separately downloaded inputs under the release repository's `external_data/` directory or edit `config/fiducial_lcdm.yaml`:

```text
external_data/PantheonPlus/Pantheon+SH0ES.dat
external_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov
external_data/DESI_DR2_BAO_observables_and_cov.txt
external_data/CMB_distance_priors_Planck2018.txt
```

The last two files are provenance checks for the reduced covariance conventions; the numerical reduced BAO and CMB covariance values used by the code are frozen in `config/fiducial_lcdm.yaml`. Verify provider filenames, provenance, and checksums before an end-to-end rerun.

## Honest execution outline

From this `scripts/production/` directory, with CLASS/classy 3.3.4 installed:

```bash
python src/early_universe/build_class_grid.py
python src/generation/generate_joint_mock_triplets.py \
  --nmock 100 --master-seed 2507013802 \
  --config config/fiducial_lcdm.yaml \
  --output mocks/joint/pedagogic2_joint_lcdm_N0100.h5
```

Then run each CPL chain separately, preserving the exact seed mapping:

```bash
python src/inference/run_joint_mcmc.py \
  --config config/mcmc_main_N100.yaml \
  --case N100-I --seed $((20260800 + I))
```

where `I` is each integer from 0 through 99. Do not use the command's generic default seed. Runs use 64 walkers, 3000 burn-in steps, unthinned production, and the frozen convergence/extension policy through at most 80,000 production steps.

After all 100 backends pass convergence checks:

```bash
python src/inference/analyze_main_n100_final.py --root .
```

That final command is expensive: it reads all retained samples and independently optimizes LambdaCDM and CPL for every realization. It is the source for summary extraction, direct HPD membership, coverage, LR statistics, joint-optimum probe decomposition, weak modes, pivots, and optimization diagnostics. The public package does not contain the production chain inputs needed to execute it.
