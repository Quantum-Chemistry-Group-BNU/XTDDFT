"""Compare PySCF UHF-CIS transition data with XTDA."""

import sys
from pathlib import Path

import numpy as np
from pyscf import ci, gto, scf
from pyscf.tdscf import uhf as pyscf_uhf_tdscf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from XTDDFT_dev.XTDDFT.xtda import XTDA
from XTDDFT_dev.utils.backend import set_backend


def test_uhf_cis_matches_xtda_transition_data():
    mol = gto.M(
        atom="""
            O    0.000000    0.000000    0.000000
            H    0.000000    0.757160    0.586260
            H    0.000000   -0.757160    0.586260
        """,
        basis="6-31G",
        charge=0,
        spin=0,
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged

    nstates = 5
    pyscf_cis = pyscf_uhf_tdscf.CIS(mf)
    pyscf_cis.conv_tol = 1e-10
    pyscf_cis.kernel(nstates=nstates)

    set_backend("cpu")
    xtda = XTDA(mf, davidson=False, davidson_backend="cpu")
    xtda.kernel(nstates=nstates)
    np.testing.assert_allclose(xtda.e, pyscf_cis.e, atol=1e-6, rtol=0)

    nmo = tuple(coeff.shape[1] for coeff in mf.mo_coeff)
    nocc = tuple(int(np.count_nonzero(occ > 0)) for occ in mf.mo_occ)
    nvir = tuple(nmo[s] - nocc[s] for s in (0, 1))
    zero_c1 = tuple(np.zeros((nocc[s], nvir[s])) for s in (0, 1))
    zero_c2 = (
        np.zeros((nocc[0], nocc[0], nvir[0], nvir[0])),
        np.zeros((nocc[0], nocc[1], nvir[0], nvir[1])),
        np.zeros((nocc[1], nocc[1], nvir[1], nvir[1])),
    )
    cis = ci.UCISD(mf)
    ground = cis.amplitudes_to_cisdvec(1.0, zero_c1, zero_c2)

    pyscf_vectors = []
    pyscf_tdm = []
    for (x_alpha, x_beta), _ in pyscf_cis.xy:
        pyscf_vectors.append(np.hstack((x_alpha.ravel(), x_beta.ravel())))
        excited = cis.amplitudes_to_cisdvec(
            0.0, (x_alpha, x_beta), zero_c2
        )
        dm_alpha, dm_beta = cis.trans_rdm1(ground, excited)
        pyscf_tdm.append(
            np.block([
                [dm_alpha, np.zeros((nmo[0], nmo[1]))],
                [np.zeros((nmo[1], nmo[0])), dm_beta],
            ])
        )

    pyscf_vectors = np.asarray(pyscf_vectors).T
    xtda_vectors = np.empty_like(xtda.v)
    xtda_vectors[xtda.order] = xtda.v
    signs = np.where(
        np.einsum("is,is->s", xtda_vectors, pyscf_vectors) < 0.0,
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(
        np.asarray([
            xtda.transition_density_matrix(state + 1, 0)
            for state in range(nstates)
        ]),
        signs[:, None, None] * np.asarray(pyscf_tdm),
        atol=1e-5,
        rtol=0,
    )
    np.testing.assert_allclose(
        xtda.transition_dipoles_ground(),
        signs[:, None] * pyscf_cis.transition_dipole(),
        atol=1e-5,
        rtol=0,
    )
