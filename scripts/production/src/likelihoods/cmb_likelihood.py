# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from src.theory.cmb import theory_vector

class CMBLikelihood:
    def __init__(self,observed,covariance):
        self.observed=np.asarray(observed,float);self.cov=np.asarray(covariance,float)
        if self.observed.shape!=(3,) or self.cov.shape!=(3,3): raise ValueError("CMB dimensions")
        self.cf=cho_factor(self.cov,lower=True,check_finite=True)
    def predict(self,theta,early): return theory_vector(theta,early)
    def chi2(self,theta,early):
        r=self.observed-self.predict(theta,early)
        return float(r@cho_solve(self.cf,r))
