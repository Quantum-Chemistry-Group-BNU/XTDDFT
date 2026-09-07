"""First-order uniform-magnetic-field operators in spin-separated X2C.

The implementation follows Eqs. (181) and (182) of JCP 141, 054111
(2014). It returns the orbital and spin-dependent Zeeman coefficients in
the contracted AO basis.
"""

import time

import numpy as np
import scipy.linalg

from ...utils.backend import asnumpy, require_cupy, resolve_backend
from .x2c_somf import sfx2c1e

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging
    logger = logging.getLogger(__name__)


def _get_tsfsd(mol):
    """Return the kinetic orbital and spin magnetic-field integrals."""
    nb = mol.nao_nr()
    tsf = -0.5 * mol.intor("int1e_cg_irxp", comp=3)
    tsd = np.zeros((3, 3, nb, nb))
    overlap = mol.intor_symmetric("int1e_ovlp")
    for axis in range(3):
        tsd[axis, axis] = 0.5 * overlap
    return tsf, tsd


def _get_wsfsd(mol):
    """Return the potential orbital and spin magnetic-field integrals."""
    nb = mol.nao_nr()
    integrals = mol.intor(
        "int1e_cg_sa10nucsp_sph", comp=12
    ).reshape(3, 4, nb, nb)
    wsf = (integrals - integrals.transpose(0, 1, 3, 2))[:, 3]
    wsd = -(
        integrals.transpose(1, 0, 3, 2)
        + integrals.transpose(1, 0, 2, 3)
    )[:, :3]
    return wsf, wsd


def _get_mag(a4, sinv, x, rp, h1e, tsf, wsf, sign):
    tmp1 = rp.T @ (
        tsf @ x
        + x.T @ tsf
        + x.T @ (a4 * wsf - tsf) @ x
    ) @ rp
    tmp2 = h1e @ sinv @ rp.T @ x.T @ tsf @ x @ rp
    return tmp1 - a4 * (tmp2 + sign * tmp2.T)


def get_zeeman(myhf, mol, c, origin, backend="auto", debug=False):
    """Build first-order uniform-magnetic-field X2C operators.

    Args:
        myhf: A PySCF scalar-X2C SCF object created with scf.sfx2c.
        mol: The corresponding molecular Mole object.
        c: Speed of light in atomic units.
        origin: Gauge origin with shape (3,).
        backend: "auto", "cpu" or "gpu" for matrix transforms.
        debug: Log norms and permutation-symmetry residuals.

    Returns:
        (h10, h11) in the contracted AO basis. Their shapes are
        (3, nao, nao) and (3, 3, nao, nao).
    """
    with_x2c = getattr(myhf, "with_x2c", None)
    if with_x2c is None:
        raise ValueError("get_zeeman requires an SCF object wrapped by scf.sfx2c")

    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,):
        raise ValueError(f"origin must have shape (3,), got {origin.shape}")

    time0 = time.time()
    mode = resolve_backend(backend)
    xp = require_cupy() if mode == "gpu" else np
    xmol, contr_coeff = with_x2c.get_xmol(mol)
    nb, nc = contr_coeff.shape
    logger.info(
        f"Begin to generate Zeeman operators: backend={mode}, "
        f"(nb,nc)=({nb},{nc})"
    )

    with xmol.with_common_orig(origin):
        t = xmol.intor_symmetric("int1e_kin")
        v = xmol.intor_symmetric("int1e_nuc")
        s = xmol.intor_symmetric("int1e_ovlp")
        w = xmol.intor_symmetric("int1e_pnucp")
        x, rp, h1e = sfx2c1e(t, v, w, s, c)
        sinv = scipy.linalg.pinv(s)
        tsf, tsd = _get_tsfsd(xmol)
        wsf, wsd = _get_wsfsd(xmol)

    x, rp, h1e, sinv, tsf, tsd, wsf, wsd = (
        xp.asarray(array)
        for array in (x, rp, h1e, sinv, tsf, tsd, wsf, wsd)
    )
    a4 = 0.25 / c**2
    h10 = xp.zeros((3, nb, nb))
    h11 = xp.zeros((3, 3, nb, nb))
    for magnetic_axis in range(3):
        h10[magnetic_axis] = _get_mag(
            a4, sinv, x, rp, h1e,
            tsf[magnetic_axis], wsf[magnetic_axis], -1.0,
        )
        for spin_axis in range(3):
            h11[magnetic_axis, spin_axis] = _get_mag(
                a4, sinv, x, rp, h1e,
                tsd[magnetic_axis, spin_axis],
                wsd[magnetic_axis, spin_axis],
                1.0,
            )

    coeff = xp.asarray(contr_coeff)
    h10 = xp.einsum("pi,xpq,qj->xij", coeff, h10, coeff)
    h11 = xp.einsum("pi,xypq,qj->xyij", coeff, h11, coeff)

    if debug:
        h10_cpu = asnumpy(h10)
        h11_cpu = asnumpy(h11)
        logger.info(
            f"Zeeman norms: h10={np.linalg.norm(h10_cpu):.12e}, "
            f"h11={np.linalg.norm(h11_cpu):.12e}"
        )
        logger.info(
            "Zeeman symmetry residuals: "
            f"h10+h10.T={np.linalg.norm(h10_cpu + h10_cpu.swapaxes(-1, -2)):.12e}, "
            f"h11-h11.T={np.linalg.norm(h11_cpu - h11_cpu.swapaxes(-1, -2)):.12e}"
        )

    logger.info(f"End of generating Zeeman operators, cost time={time.time()-time0:.2f}s")
    return h10, h11
