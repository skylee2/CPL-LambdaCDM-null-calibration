# Analysis conventions and external inputs

This release calibrates a deliberately reduced Gaussian BAO + compressed-CMB + uncalibrated-SNe likelihood. It does not reproduce or claim to redistribute the complete official DESI, Planck, or Pantheon+ likelihoods.

## BAO

The BAO vector contains the anisotropic pair `(D_M/r_d, D_H/r_d)` at each of six DESI DR2 effective redshifts:

```text
0.510, 0.706, 0.934, 1.321, 1.484, 2.330
```

The ordering is interleaved by redshift. Each bin uses a reconstructed 2 x 2 covariance block from the reported marginal uncertainties and within-bin correlation. The adopted 12 x 12 covariance is block diagonal: within-bin transverse-radial covariance is retained and between-bin covariance is set to zero. The isotropic BGS point at `z_eff=0.295` is not included. This is a DESI-DR2-like reduced compression, not the exact collaboration likelihood. See the DESI first-year cosmology release ([arXiv:2404.03002](https://arxiv.org/abs/2404.03002)) and the DR2 BAO results cited by the manuscript.

## CMB

The compressed vector is `(R, l_A, omega_b)`. Its fixed 3 x 3 marginal covariance is the geometrical subset of a Planck 2018 flat-universe distance-prior compression based on `base_plikHM_TTTEEE_lowl_lowE_lensing`. The acoustic scale is

```text
l_A = pi D_M(z_star) / r_star
```

and therefore uses the sound horizon at photon decoupling `r_star`, not the drag-epoch `r_d` used by BAO. The observed Planck central vector is not the mock mean; the generating cosmology supplies the mean. See Chen, Huang & Wang, JCAP 02 (2019) 028 ([doi:10.1088/1475-7516/2019/02/028](https://doi.org/10.1088/1475-7516/2019/02/028)).

## Supernovae

The SNe block uses public Pantheon+ Hubble-diagram redshifts and the released unbinned statistical-plus-systematic covariance. The selection `z_HD > 0.01` retains 1590 entries and the full selected 1590 x 1590 covariance. Cosmological distance is evaluated at `z_HD`, while `(1 + z_HEL)` supplies the heliocentric luminosity-distance prefactor. The observed Pantheon+ residual pattern, Cepheid-host distances, and SH0ES absolute calibration are not used. A free additive nuisance parameter `DeltaM` is sampled and marginalized. See Brout et al., ApJ 938 (2022) 110 ([doi:10.3847/1538-4357/ac8e04](https://doi.org/10.3847/1538-4357/ac8e04)).

## Early-universe and posterior conventions

- CLASS/classy 3.3.4 with HyRec, BBN-consistent helium, `T_CMB=2.7255 K`, `N_eff=3.046`, `N_ncdm=0`, and zero curvature.
- Sampling parameters are `(omega_m, omega_b, H0, w0, wa, DeltaM)`. `Omega_m0` is derived sample by sample.
- Marginal intervals are equal-tailed at 68.27% and 95.45%.
- Primary joint direct-HPD regions use at most 40,000 deterministic equal-stride samples, 0.2/99.8 percentile limits with 8% padding, a 100 x 100 histogram, and Gaussian smoothing `sigma=1.15` grid bins. Density thresholds enclose 68.27% and 95.45% of grid-cell mass.
- Gaussian covariance ellipses are a secondary reference and use two-dimensional chi-square thresholds.
- `Delta_chi2_LR` independently optimizes the nested LambdaCDM and CPL models. Probe contributions are evaluated at the two common joint optima; they are not probe-only likelihood ratios.
