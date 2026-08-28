# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from src.theory.bao import theory_vector

class BAOLikelihood:
    def __init__(self,observed,covariance,redshift):
        self.observed=np.asarray(observed,float);self.cov=np.asarray(covariance,float)
        if self.observed.shape!=(12,) or self.cov.shape!=(12,12): raise ValueError("BAO dimensions")
        self.cf=cho_factor(self.cov,lower=True,check_finite=True);self.redshift=np.asarray(redshift,float)
    def predict(self,theta,early): return theory_vector(theta,self.redshift,early["r_d"])
    def chi2(self,theta,early):
        r=self.observed-self.predict(theta,early)
        return float(r@cho_solve(self.cf,r))
