#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Summarize statistical recovery in generated HDF5 engineering mocks."""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import solve_triangular


def probe_rows(h,probe):
    x=h[f"{probe}/mocks"][:]; mean=h[f"{probe}/mean"][:]; cov=h[f"{probe}/cov"][:]
    L=np.linalg.cholesky(cov)
    white=solve_triangular(L,(x-mean).T,lower=True).T
    n=len(x); rows=[]
    rows.append(dict(probe=probe,metric="whitened_mean_rms",value=float(np.sqrt(np.mean(white.mean(0)**2))),
                     expectation=f"approximately <= {3/np.sqrt(n):.6g}"))
    rows.append(dict(probe=probe,metric="whitened_variance_mean",value=float(np.mean(np.var(white,axis=0,ddof=1))),
                     expectation="approximately 1"))
    rows.append(dict(probe=probe,metric="mean_chi2_per_dimension",value=float(np.mean(np.sum(white**2,axis=1))/white.shape[1]),
                     expectation="approximately 1"))
    rows.append(dict(probe=probe,metric="finite_fraction",value=float(np.mean(np.isfinite(x))),expectation="1"))
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--csv",required=True)
    ap.add_argument("--json",required=True)
    args=ap.parse_args()
    with h5py.File(args.input,"r") as h:
        rows=sum((probe_rows(h,p) for p in ("bao","cmb","sne")),[])
        summary={"input":str(Path(args.input).resolve()),"nmock":int(h["metadata"].attrs["nmock"]),
                 "mask_count":int(h["sne/row_mask"][:].sum()),
                 "shapes":{k:list(h[k].shape) for k in
                           ("bao/mocks","cmb/mocks","sne/mocks","sne/cov")},
                 "fiducial":{k:float(v) for k,v in h["fiducial"].attrs.items()
                             if isinstance(v,(int,float,np.integer,np.floating))},
                 "early_universe":{k:float(v) for k,v in h["early_universe"].attrs.items()
                                   if isinstance(v,(int,float,np.integer,np.floating))},
                 "checksums":{k:str(v) for k,v in h["checksums"].attrs.items()},
                 "validation_pass":bool(
                     int(h["sne/row_mask"][:].sum())==1590
                     and all(r["value"]==1.0 for r in rows if r["metric"]=="finite_fraction")
                     and all(abs(r["value"]-1)<0.15 for r in rows
                             if r["metric"] in ("whitened_variance_mean","mean_chi2_per_dimension"))
                 ),
                 "rows":rows}
    Path(args.csv).parent.mkdir(parents=True,exist_ok=True)
    with open(args.csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    Path(args.json).write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
