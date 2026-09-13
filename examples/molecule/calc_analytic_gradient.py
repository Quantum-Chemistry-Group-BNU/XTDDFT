#!/usr/bin/env python
"""Calculate analytic gradients with a CPU or GPU backend."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_DYNAMIC"] = "False"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
from time import perf_counter
from pyscf import gto, scf
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xtda import XTDA
from XTDDFT.XTDDFT.grad.finite_difference import fd_gradient, fd_gradient_forth
from XTDDFT.utils.backend import set_backend


def parse_xyz_string(xyz: str):
    """Change the coordinate format from string to array."""
    atoms = []

    for line in xyz.strip().splitlines():
        parts = line.split()
        if not parts:
            continue

        atom = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append((atom, np.array((x, y, z))))

    return atoms


# ===== Manually edit these parameters on the server =====
method_kind = "xsc"  # "usf_up", "usf_down", "usc", "xsf_up", "xsf_down", "xsc"
use_gpu = False

xc = "b3lyp"
basis = '6-31g'
spin = 2
charge = 0
cs = 20  # collinear_samples
states = [1]
sf_method = 1
verbose = 4  # finite difference output level adjust line 19 in XTDDFT/grad/finite_difference.py 
atom = '''
    H   0.000000   0.934473  -0.588078
    H   0.000000  -0.934473  -0.588078
    C   0.000000   0.000000   0.000000
    O   0.000000   0.000000   1.221104
'''
# finite difference use this form coordinate
atom = parse_xyz_string(atom)
# ========================================================


def main():
    if use_gpu:
        import cupy as cp
    set_backend("gpu" if use_gpu else "cpu")
    kind = method_kind.lower()
    if kind not in ("usf_up", "usf_down", "usc", "xsf_up", "xsf_down", "xsc"):
        raise ValueError("method_kind must be 'usf_up', 'usf_down'," \
        " 'usc', 'xsf_up', 'xsf_down', 'xsc'")

    cp.cuda.runtime.deviceSynchronize()
    time0 = perf_counter()
    mol = gto.M(
        atom = atom,
        spin = spin,
        charge = charge,
        basis = basis,
        verbose = verbose,
    )
    if kind[0] == 'u':
        mf = scf.UKS(mol)
    else:
        mf = scf.ROKS(mol)
    mf.xc = xc
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.grids.level = 5
    if use_gpu:
        mf = mf.to_gpu()
    mf.kernel()
    cp.cuda.runtime.deviceSynchronize()
    time1 = perf_counter()
    print(f"SCF: {time1 - time0:.3f} s")

    if "sf_down" in kind:
        td = XSF_TDA_down(mf, method=sf_method, davidson=True, collinear_samples=cs)
    elif "sf_up" in kind:
        td = SF_TDA_up(mf, method=sf_method, davidson=True, collinear_samples=cs)
    elif "sc" in kind:
        td = XTDA(mf, davidson=True)
    else:
        raise ValueError
    td.kernel(nstates=max(max(states), 1) + 2)
    cp.cuda.runtime.deviceSynchronize()
    time2 = perf_counter()
    print(f"excited states: {time2 - time1:.3f} s")

    for state in states:
        gradient = td.nuc_grad_method(state=state).kernel()
        print('analytic:\n')
        print(gradient)

        # second order finite difference
        g_fd = fd_gradient(
            atom, state, mk=method_kind, charge=charge, spin=spin,
              xc=xc, basis=basis, method=sf_method, cs=cs
        )

        # # forth order finite difference
        # g_fd = fd_gradient_forth(
        #     atom, state, mk=method_kind, charge=charge, spin=spin,
        #       xc=xc, basis=basis, method=sf_method, cs=cs
        # )
        print('finite-diff:\n')
        print(np.array2string(g_fd, formatter={'float_kind': lambda x: f'{x: .10f}'}))
    cp.cuda.runtime.deviceSynchronize()
    time3 = perf_counter()
    print(f"Gradient: {time3 - time2:.3f} s")
    print(f"Total: {time3 - time0:.3f} s")


if __name__ == "__main__":
    main()
