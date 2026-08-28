# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import yaml

from src.theory.early_universe import EarlyUniverseEmulator,cmb_from_emulator
from .bao_likelihood import BAOLikelihood
from .cmb_likelihood import CMBLikelihood
from .sne_likelihood import SNELikelihood

PARAMETER_ORDER=("omega_m","omega_b","H0","w0","wa","DeltaM")
LCDM_ORDER=("omega_m","omega_b","H0","DeltaM")

class JointPrior:
    def __init__(self,path):
        self.path=Path(path);self.config=yaml.safe_load(self.path.read_text())
        self.parameters=self.config["parameters"];self.domain=self.config["emulator_domain"]
    def bounds(self,model):
        names=LCDM_ORDER if model=="lcdm" else PARAMETER_ORDER
        return [(float(self.parameters[n]["lower"]),float(self.parameters[n]["upper"])) for n in names]
    def vector_to_theta(self,x,model):
        x=np.asarray(x,float)
        names=LCDM_ORDER if model=="lcdm" else PARAMETER_ORDER
        if x.shape!=(len(names),): raise ValueError("Parameter-vector shape mismatch")
        t=dict(zip(names,map(float,x)))
        if model=="lcdm": t.update(w0=-1.,wa=0.)
        return t
    def contains(self,theta):
        for n,p in self.parameters.items():
            x=theta[n]
            if not (float(p["lower"])<x<float(p["upper"])): return False
        if not (theta["w0"]+theta["wa"]<0): return False
        if not (self.domain["omega_m"][0]<theta["omega_m"]<self.domain["omega_m"][1]): return False
        if not (self.domain["omega_b"][0]<theta["omega_b"]<self.domain["omega_b"][1]): return False
        h=theta["H0"]/100
        return theta["omega_m"]>theta["omega_b"] and theta["omega_m"]/h**2<1
    def boundary_flags(self,theta,fraction=1e-4):
        out={}
        for n,p in self.parameters.items():
            lo,hi=float(p["lower"]),float(p["upper"]);x=theta[n]
            out[n]=bool(min(x-lo,hi-x)<=fraction*(hi-lo))
        out["early_condition"]=bool(-(theta["w0"]+theta["wa"])<=fraction*3)
        return out

class JointLikelihood:
    def __init__(self,mock_path,realization_id,prior_path,grid_path):
        self.mock_path=Path(mock_path);self.realization_id=int(realization_id)
        self.prior=JointPrior(prior_path);self.emulator=EarlyUniverseEmulator(grid_path)
        with h5py.File(self.mock_path,"r") as h:
            ids=h["metadata/realization_id"][:]
            where=np.flatnonzero(ids==realization_id)
            if len(where)!=1: raise ValueError("Realization ID not uniquely present")
            i=int(where[0])
            self.bao=BAOLikelihood(h["bao/mocks"][i],h["bao/cov"][:],h["bao/redshift"][:])
            self.cmb=CMBLikelihood(h["cmb/mocks"][i],h["cmb/cov"][:])
            self.sne=SNELikelihood(h["sne/mocks"][i],h["sne/cov"][:],h["sne/zHD"][:],h["sne/zHEL"][:])
    def theory(self,theta):
        early=cmb_from_emulator(theta,self.emulator)
        return {"early":early,"bao":self.bao.predict(theta,early),
                "cmb":self.cmb.predict(theta,early),"sne":self.sne.predict(theta)}
    def evaluate(self,theta,apply_prior=True):
        if apply_prior and not self.prior.contains(theta):
            return {"total":np.inf,"bao":np.inf,"cmb":np.inf,"sne":np.inf}
        early=cmb_from_emulator(theta,self.emulator)
        parts={"bao":self.bao.chi2(theta,early),"cmb":self.cmb.chi2(theta,early),
               "sne":self.sne.chi2(theta)}
        parts["total"]=sum(parts.values())
        return parts
    def derived(self,theta):
        e=cmb_from_emulator(theta,self.emulator);h=theta["H0"]/100
        return {"h":h,"Omega_m":theta["omega_m"]/h**2,
                "omega_c":theta["omega_m"]-theta["omega_b"],
                "r_d":e["r_d"],"r_star":e["r_star"],"R":e["R"],
                "l_A":e["l_A"],"h_r_d":h*e["r_d"]}
    def objective(self,x,model):
        try:t=self.prior.vector_to_theta(x,model)
        except Exception:return 1e100
        v=self.evaluate(t)["total"]
        return v if np.isfinite(v) else 1e100
