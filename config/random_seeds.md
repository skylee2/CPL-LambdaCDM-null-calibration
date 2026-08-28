# Random seeds and streams

Three independent kinds of random construction are used. They must not be interchanged.

## Mock-generation streams

- Master NumPy `SeedSequence` entropy: `2507013802`.
- Bit generator: `numpy.random.Generator(PCG64)` with NumPy 2.5.1.
- The master sequence spawns one child per probe in fixed order: BAO spawn key `[0]`, CMB `[1]`, and SNe `[2]`.
- Within each child stream, `standard_normal((100, dimension))` is called once. Rows therefore map in order to `N100-0` through `N100-99`.
- Probe vectors share a realization label and generating cosmology, not a stochastic fluctuation. The three noise streams are independent and no cross-probe noise covariance is imposed.

The implementation is in `scripts/production/src/generation/random_streams.py` and `generate_joint_mock_triplets.py`.

## MCMC sampler seeds

For realization index `i`:

```text
20260800 + i
```

These seeds initialize the sampler walkers; they do not change the already generated mock observable vectors.

## Optimization-start seeds

For realization index `i`, independent normalized-prior random starts use:

```text
910000 + i    LambdaCDM
920000 + i    CPL
```

The final analyzer uses seven total LambdaCDM starts and nine total CPL starts (three deterministic starts plus four and six random starts, respectively). The seeds and realized restart-agreement metrics are recorded in `data/optimization_diagnostics.csv`.
