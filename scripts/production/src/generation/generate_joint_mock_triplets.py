#!/usr/bin/env python3
# Copyright (c) 2026 Seokcheon Lee
# SPDX-License-Identifier: MIT
"""Generate coherent matched BAO+CMB+SNe LambdaCDM Gaussian mock triplets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml
import classy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.generation.random_streams import spawn_probe_streams
from src.theory import bao, cmb, sne
from src.theory.early_universe import EarlyUniverseEmulator, cmb_from_emulator


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def resolve_config_path(config_path, value):
    p=Path(value)
    return p if p.is_absolute() else (Path(config_path).resolve().parent/p).resolve()


def load_setup(config_path, emulator_path=None):
    config_path=Path(config_path).resolve()
    cfg=yaml.safe_load(config_path.read_text())
    p=cfg["joint_parameters"]
    theta=dict(omega_m=float(p["omega_m"]),omega_b=float(p["omega_b"]),
               H0=float(p["H0_km_s_Mpc"]),w0=float(p["w0"]),
               wa=float(p["wa"]),DeltaM=float(p["DeltaM_mag"]))
    em_path=Path(emulator_path or ROOT/"results/early_universe/class_grid.npz")
    emulator=EarlyUniverseEmulator(em_path)
    early=cmb_from_emulator(theta,emulator)
    bao_z=np.asarray(cfg["bao"]["redshift"],float)
    bao_cov=bao.covariance_from_config(cfg)
    cmb_cov=cmb.covariance_from_config(cfg)
    data_path=resolve_config_path(config_path,cfg["inputs"]["pantheon_data"])
    cov_path=resolve_config_path(config_path,cfg["inputs"]["pantheon_covariance"])
    bao_source_path=resolve_config_path(config_path,cfg["inputs"]["bao_source_read_only"])
    cmb_source_path=resolve_config_path(config_path,cfg["inputs"]["cmb_source_read_only"])
    if not bao_source_path.is_file() or not cmb_source_path.is_file():
        raise FileNotFoundError("Frozen BAO/CMB provenance input is missing")
    data,full_cov=sne.read_release(data_path,cov_path)
    mask,sn_cov=sne.select_official(data,full_cov)
    zhd=np.asarray(data["zHD"][mask],float); zhel=np.asarray(data["zHEL"][mask],float)
    cid=np.asarray(data["CID"][mask],str)
    means=dict(bao=bao.theory_vector(theta,bao_z,early["r_d"]),
               cmb=cmb.theory_vector(theta,early),
               sne=sne.theory_vector(zhd,zhel,theta))
    return locals()


def generate_arrays(setup,nmock,master_seed,zero_noise=False):
    streams,rng_meta=spawn_probe_streams(master_seed)
    ids=np.arange(nmock,dtype=np.int64)
    arrays={}
    for name,cov in (("bao",setup["bao_cov"]),("cmb",setup["cmb_cov"]),("sne",setup["sn_cov"])):
        mean=setup["means"][name]
        if zero_noise:
            arrays[name]=np.repeat(mean[None,:],nmock,axis=0)
        else:
            L=np.linalg.cholesky(cov)
            arrays[name]=mean+streams[name].standard_normal((nmock,len(mean)))@L.T
    return ids,arrays,rng_meta


def write_hdf5(path,setup,nmock,master_seed,zero_noise=False,overwrite=False):
    path=Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    ids,arr,rng_meta=generate_arrays(setup,nmock,master_seed,zero_noise)
    cfg_path=setup["config_path"]; cfg=setup["cfg"]; theta=setup["theta"]
    with h5py.File(path,"w") as f:
        md=f.create_group("metadata")
        md.attrs.update(schema_version="1.0",creation_timestamp_utc=datetime.now(timezone.utc).isoformat(),
                        description="coherent observable-level Gaussian BAO+CMB+SNe LambdaCDM mocks",
                        nmock=nmock,zero_noise=zero_noise,class_version=classy.__version__,
                        emulator_version=setup["emulator"].version,
                        neutrino_convention=cfg["physics"]["neutrino_convention"],
                        sum_mnu_eV=float(cfg["physics"]["sum_mnu_eV"]),
                        parameter_order=json.dumps(["omega_m","omega_b","H0","w0","wa","DeltaM"]),
                        code_commit_hash="unavailable: Pedagogic2 is not an independent Git worktree")
        md.create_dataset("realization_id",data=ids)
        fid=f.create_group("fiducial")
        for k,v in theta.items(): fid.attrs[k]=v
        fid.attrs["Omega_m"]=theta["omega_m"]/(theta["H0"]/100)**2
        eu=f.create_group("early_universe")
        for k,v in setup["early"].items():
            if np.isscalar(v): eu.attrs[k]=v
        eu.attrs["emulator_domain_omega_m"]=cfg["emulator"]["omega_m_range"]
        eu.attrs["emulator_domain_omega_b"]=cfg["emulator"]["omega_b_range"]
        eu.attrs["sound_horizon_convention"]="r_d=r_s(z_drag); r_star=r_s(z_star)"
        bg=f.create_group("bao")
        bg.create_dataset("redshift",data=setup["bao_z"])
        bg.create_dataset("observable_names",data=np.asarray(bao.OBSERVABLE_NAMES,dtype="S"))
        bg.create_dataset("mean",data=setup["means"]["bao"])
        bg.create_dataset("cov",data=setup["bao_cov"])
        bg.create_dataset("mocks",data=arr["bao"],compression="gzip",shuffle=True)
        bg.attrs.update(units="dimensionless",description=cfg["bao"]["description"])
        bg.attrs["ordering"]="interleaved by redshift: DM_over_rd, DH_over_rd"
        bg.attrs["covariance_ordering"]="six 2x2 within-bin blocks; zero cross-bin covariance"
        cg=f.create_group("cmb")
        cg.create_dataset("observable_names",data=np.asarray(cmb.OBSERVABLE_NAMES,dtype="S"))
        cg.create_dataset("mean",data=setup["means"]["cmb"])
        cg.create_dataset("cov",data=setup["cmb_cov"])
        cg.create_dataset("mocks",data=arr["cmb"],compression="gzip",shuffle=True)
        cg.attrs["units"]="R: dimensionless; l_A: dimensionless; omega_b: dimensionless"
        cg.attrs["ordering"]="R, l_A, omega_b"
        cg.attrs["acoustic_convention"]="l_A=pi*D_M(z_star)/r_star"
        sg=f.create_group("sne")
        sg.create_dataset("row_mask",data=setup["mask"].astype(np.uint8))
        sg.create_dataset("zHD",data=setup["zhd"]); sg.create_dataset("zHEL",data=setup["zhel"])
        sg.create_dataset("CID",data=np.asarray(setup["cid"],dtype="S"))
        sg.create_dataset("mean",data=setup["means"]["sne"])
        sg.create_dataset("cov",data=setup["sn_cov"],compression="gzip",shuffle=True)
        sg.create_dataset("mocks",data=arr["sne"],compression="gzip",shuffle=True)
        sg.attrs.update(units="mag",mask_definition="zHD > 0.01",retained_count=1590,
                        mean_definition="theory-only LambdaCDM; no MU_SH0ES/CEPH_DIST/observed residuals")
        rg=f.create_group("random_streams")
        rg.attrs["master_entropy"]=master_seed
        rg.attrs["generator_type"]=rng_meta["generator_type"]; rg.attrs["numpy_version"]=rng_meta["numpy_version"]
        for label,detail in rng_meta["streams"].items():
            rg.create_dataset(f"{label}_spawn_key",data=np.asarray(detail["spawn_key"],dtype=np.int64))
        checks=f.create_group("checksums")
        checks.attrs["config_sha256"]=sha256_file(cfg_path)
        checks.attrs["pantheon_data_sha256"]=sha256_file(setup["data_path"])
        checks.attrs["pantheon_covariance_sha256"]=sha256_file(setup["cov_path"])
        checks.attrs["bao_source_file_sha256"]=sha256_file(setup["bao_source_path"])
        checks.attrs["cmb_source_file_sha256"]=sha256_file(setup["cmb_source_path"])
        checks.attrs["bao_covariance_sha256"]=hashlib.sha256(setup["bao_cov"].tobytes()).hexdigest()
        checks.attrs["cmb_covariance_sha256"]=hashlib.sha256(setup["cmb_cov"].tobytes()).hexdigest()
        checks.attrs["sne_selected_covariance_sha256"]=hashlib.sha256(setup["sn_cov"].tobytes()).hexdigest()
        checks.attrs["emulator_grid_sha256"]=sha256_file(setup["em_path"])
        manifest=ROOT/"results/early_universe/emulator_manifest.json"
        if manifest.is_file():
            checks.attrs["emulator_manifest_sha256"]=sha256_file(manifest)
    return path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nmock",type=int,required=True)
    ap.add_argument("--master-seed",type=int,default=2507013802)
    ap.add_argument("--config",default=ROOT/"config/fiducial_lcdm.yaml")
    ap.add_argument("--output",required=True)
    ap.add_argument("--overwrite",action="store_true")
    ap.add_argument("--zero-noise",action="store_true",help="engineering closure only")
    args=ap.parse_args()
    if args.nmock<1: ap.error("--nmock must be positive")
    setup=load_setup(args.config)
    out=write_hdf5(args.output,setup,args.nmock,args.master_seed,args.zero_noise,args.overwrite)
    print(out.resolve())


if __name__=="__main__":
    main()
