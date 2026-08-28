# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from src.theory.sne import theory_vector

class SNELikelihood:
    def __init__(self,observed,covariance,zHD,zHEL):
        self.observed=np.asarray(observed,float);self.cov=np.asarray(covariance,float)
        self.zHD=np.asarray(zHD,float);self.zHEL=np.asarray(zHEL,float)
        n=len(self.observed)
        if n!=1590 or self.cov.shape!=(n,n) or self.zHD.shape!=(n,) or self.zHEL.shape!=(n,):
            raise ValueError("SNe dimensions/order mismatch")
        self.cf=cho_factor(self.cov,lower=True,check_finite=True)
        # The fixed covariance permits a one-time precision construction.
        # This is algebraically the same quadratic form as cho_solve(cf, r)
        # and removes a repeated triangular solve from every MCMC evaluation.
        self.precision=cho_solve(self.cf,np.eye(n),check_finite=False)
        self.precision=0.5*(self.precision+self.precision.T)
    def predict(self,theta): return theory_vector(self.zHD,self.zHEL,theta)
    def chi2(self,theta):
        r=self.observed-self.predict(theta)
        return float(r @ self.precision @ r)
