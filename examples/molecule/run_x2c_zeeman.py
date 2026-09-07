#!/usr/bin/env python
"""Small spin-separated X2C Zeeman example without an SCF calculation."""

import numpy as np
from pyscf import gto, scf

from XTDDFT_dev.XTDDFT.soc import get_zeeman
from XTDDFT_dev.utils.unit import BDF_c


def main():
    mol = gto.M(
        atom="C 0 0 0",
        basis="sto-3g",
        spin=0,
        verbose=0,
    )
    mf = scf.sfx2c(scf.RHF(mol))

    charges = mol.atom_charges()
    origin = np.einsum("z,zx->x", charges, mol.atom_coords()) / charges.sum()
    h10, h11 = get_zeeman(
        mf,
        mol,
        c=BDF_c,
        origin=origin,
        backend="cpu",
        debug=True,
    )

    print("h10 shape:", h10.shape)
    print("h11 shape:", h11.shape)
    print("||h10||:", np.linalg.norm(h10))
    print("||h11||:", np.linalg.norm(h11))
    print("h10 antisymmetry residual:", np.linalg.norm(h10 + h10.swapaxes(-1, -2)))
    print("h11 symmetry residual:", np.linalg.norm(h11 - h11.swapaxes(-1, -2)))


if __name__ == "__main__":
    main()
