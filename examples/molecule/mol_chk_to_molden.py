#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
from pyscf import tools
from pyscf.scf import chkfile


def main():
    parser = argparse.ArgumentParser(
        description="Convert a molecular PySCF SCF checkpoint to Molden format."
    )
    parser.add_argument("chk", help="Input molecular SCF checkpoint, e.g. scf.chk")
    parser.add_argument(
        "molden",
        nargs="?",
        help="Output Molden file. Defaults to <chk stem>.molden",
    )
    parser.add_argument(
        "--keep-high-l",
        action="store_true",
        help="Keep h/i/... shells instead of using PySCF's Molden-compatible truncation.",
    )
    args = parser.parse_args()

    chk_path = Path(args.chk)
    out_path = Path(args.molden) if args.molden else chk_path.with_suffix(".molden")

    mol, scf_rec = chkfile.load_scf(str(chk_path))
    mo_coeff = np.asarray(scf_rec["mo_coeff"])
    mo_occ = np.asarray(scf_rec["mo_occ"])

    print("chk:", chk_path.resolve())
    print("output:", out_path.resolve())
    print("natm:", mol.natm)
    print("nelectron:", mol.nelectron)
    print("spin:", mol.spin)
    print("mo_coeff shape:", mo_coeff.shape)
    print("mo_occ shape:", mo_occ.shape)

    tools.molden.from_chkfile(
        str(out_path),
        str(chk_path),
        ignore_h=not args.keep_high_l,
    )


if __name__ == "__main__":
    main()
