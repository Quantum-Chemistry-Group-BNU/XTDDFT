#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from pyscf import tools
from pyscf.pbc.scf import chkfile


def _nkpts(scf_rec):
    kpts = scf_rec.get("kpts", None)
    if kpts is None:
        return 1
    kpts = np.asarray(kpts)
    if kpts.size == 0:
        return 1
    return int(kpts.reshape(-1, 3).shape[0])


def _case_from_occ(mo_occ, nkpts):
    occ = np.asarray(mo_occ)
    if occ.ndim == 1:
        return "restricted_gamma"
    if occ.ndim == 2:
        if nkpts == 1 and occ.shape[0] == 2:
            return "unrestricted_gamma"
        return "restricted_kpts"
    if occ.ndim == 3 and occ.shape[0] == 2:
        return "unrestricted_kpts"
    raise ValueError(f"Unsupported mo_occ shape for PBC checkpoint: {occ.shape}")


def _as_real_mo(mo, label, real_part, imag_tol):
    mo = np.asarray(mo)
    if not np.iscomplexobj(mo):
        return mo

    max_imag = float(np.max(np.abs(mo.imag))) if mo.size else 0.0
    if max_imag <= imag_tol:
        return np.asarray(mo.real)
    if real_part:
        print(f"warning: using real part of complex {label}; max |imag| = {max_imag:.3e}")
        return np.asarray(mo.real)
    raise ValueError(
        f"{label} has complex MO coefficients (max |imag| = {max_imag:.3e}). "
        "Molden normally stores real coefficients only. Use --real-part for "
        "qualitative visualization, or export a Gamma-point checkpoint."
    )


def _write_orbitals(cell, out_path, blocks, ignore_h):
    with open(out_path, "w") as fout:
        tools.molden.header(cell, fout, ignore_h=ignore_h)
        for block in blocks:
            tools.molden.orbital_coeff(
                cell,
                fout,
                block["mo_coeff"],
                spin=block["spin"],
                ene=block["mo_energy"],
                occ=block["mo_occ"],
                ignore_h=ignore_h,
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a PySCF PBC SCF checkpoint to a Molden file for orbital "
            "visualization. Molden does not encode periodic boundary conditions."
        )
    )
    parser.add_argument("chk", help="Input PBC SCF checkpoint")
    parser.add_argument(
        "molden",
        nargs="?",
        help="Output Molden file. Defaults to <chk stem>.k<kpt>.molden",
    )
    parser.add_argument(
        "--kpt",
        type=int,
        default=0,
        help="0-based k-point index for K-point checkpoints. Default: 0",
    )
    parser.add_argument(
        "--real-part",
        action="store_true",
        help="For complex non-Gamma orbitals, write only the real part.",
    )
    parser.add_argument(
        "--imag-tol",
        type=float,
        default=1e-10,
        help="Imaginary-part tolerance before an orbital is treated as complex.",
    )
    parser.add_argument(
        "--keep-high-l",
        action="store_true",
        help="Keep h/i/... shells instead of using PySCF's Molden-compatible truncation.",
    )
    args = parser.parse_args()

    chk_path = Path(args.chk)
    out_path = (
        Path(args.molden)
        if args.molden
        else chk_path.with_suffix(f".k{args.kpt}.molden")
    )

    cell, scf_rec = chkfile.load_scf(str(chk_path))
    mo_coeff = np.asarray(scf_rec["mo_coeff"])
    mo_occ = np.asarray(scf_rec["mo_occ"])
    mo_energy = np.asarray(scf_rec.get("mo_energy", None))
    nkpts = _nkpts(scf_rec)
    case = _case_from_occ(mo_occ, nkpts)

    if args.kpt < 0 or args.kpt >= nkpts:
        raise IndexError(f"--kpt {args.kpt} is out of range for nkpts={nkpts}")

    print("chk:", chk_path.resolve())
    print("output:", out_path.resolve())
    print("natm:", cell.natm)
    print("nelectron:", cell.nelectron)
    print("spin:", cell.spin)
    print("nkpts:", nkpts)
    print("case:", case)
    print("mo_coeff shape:", mo_coeff.shape)
    print("mo_occ shape:", mo_occ.shape)

    blocks = []
    if case == "restricted_gamma":
        blocks.append({
            "spin": "Alpha",
            "mo_coeff": _as_real_mo(mo_coeff, "Gamma restricted orbitals", args.real_part, args.imag_tol),
            "mo_energy": mo_energy if mo_energy.ndim == 1 else None,
            "mo_occ": mo_occ,
        })
    elif case == "unrestricted_gamma":
        for ispin, spin in enumerate(("Alpha", "Beta")):
            blocks.append({
                "spin": spin,
                "mo_coeff": _as_real_mo(mo_coeff[ispin], f"Gamma {spin} orbitals", args.real_part, args.imag_tol),
                "mo_energy": mo_energy[ispin] if mo_energy.ndim >= 2 else None,
                "mo_occ": mo_occ[ispin],
            })
    elif case == "restricted_kpts":
        blocks.append({
            "spin": "Alpha",
            "mo_coeff": _as_real_mo(mo_coeff[args.kpt], f"k-point {args.kpt} restricted orbitals", args.real_part, args.imag_tol),
            "mo_energy": mo_energy[args.kpt] if mo_energy.ndim >= 2 else None,
            "mo_occ": mo_occ[args.kpt],
        })
    else:
        for ispin, spin in enumerate(("Alpha", "Beta")):
            blocks.append({
                "spin": spin,
                "mo_coeff": _as_real_mo(mo_coeff[ispin, args.kpt], f"k-point {args.kpt} {spin} orbitals", args.real_part, args.imag_tol),
                "mo_energy": mo_energy[ispin, args.kpt] if mo_energy.ndim >= 3 else None,
                "mo_occ": mo_occ[ispin, args.kpt],
            })

    _write_orbitals(cell, str(out_path), blocks, ignore_h=not args.keep_high_l)


if __name__ == "__main__":
    main()
