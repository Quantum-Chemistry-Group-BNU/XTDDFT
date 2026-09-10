"""System tests for XTDA state alignment, transition densities, and NACs.

The numerical integration test compares XTDA excitation energies, transition
reduced density matrices, and finite-difference NAC vectors against BDF output
for CH2NH+.
"""

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT.nac.nac import (
    _state_pairs,
    align_amplitudes,
    signed_permutation,
)


def test_alignment_reorders_and_phases_roots():
    reference = np.eye(3)
    displaced = reference[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])

    np.testing.assert_allclose(align_amplitudes(reference, displaced), reference)


def test_ambiguous_alignment_fails():
    with pytest.raises(ValueError, match="signed permutation"):
        signed_permutation(np.full((2, 2), 0.75), "state")


@pytest.mark.parametrize("pairs", [[], [0, 1.0], [-1, 0], [0, 0], [[0, 1, 2]]])
def test_state_pair_validation(pairs):
    with pytest.raises(ValueError):
        _state_pairs(pairs)


def test_one_pair_is_normalized_to_a_matrix():
    np.testing.assert_array_equal(_state_pairs((0, 1)), [[0, 1]])


def test_xsf_transition_density_uses_physical_excited_state_labels():
    from XTDDFT_dev.XTDDFT.xsf_tda_down import XSF_TDA_down

    method = object.__new__(XSF_TDA_down)
    method.type_u = False
    method.nstates = 2
    method.v = np.zeros((1, 2))
    calls = []
    method._transition_density_matrix_r = (
        lambda state_f, state_i: calls.append((state_f, state_i)) or np.eye(1)
    )

    np.testing.assert_array_equal(method.transition_density_matrix(2, 1), np.eye(1))
    assert calls == [(1, 0)]
    with pytest.raises(ValueError, match='starting at 1'):
        method.transition_density_matrix(0, 1)


# BDF reference: CH2NH_XTDA_itrans0_resp.out, HF/def2-SVP, raw NACMEs.
# Both BDF and the Python API use physical labels: S0 is the ground state.
REFERENCE_DATA = Path(__file__).with_name('CH2NH_XTDA_reference.npz')
CH2NH_GEOMETRY = '''
C -2.91400807 -0.55823559  0.03052752
H -2.37721823 -1.66302155  0.09965968
H -3.96527633 -0.77226144 -0.33224522
N -2.34030191  0.59563094  0.25221390
H -2.80790566  1.54696657  0.05511270
'''
PHYSICAL_STATE_PAIRS = tuple(
    (state_i, state_j)
    for state_i in range(7)
    for state_j in range(state_i + 1, 7)
)


def _load_bdf_ch2nh_reference():
    reference = np.load(REFERENCE_DATA)
    pairs = [tuple(pair) for pair in reference['nac_state_pairs']]
    assert pairs == list(PHYSICAL_STATE_PAIRS)
    nac = reference['nac_raw_bohr']
    assert nac.shape == (len(PHYSICAL_STATE_PAIRS), 5, 3)
    return reference['excitation_energies_hartree'], nac



@pytest.mark.slow
def test_ch2nh_hf_def2svp_nac_matches_bdf():
    """Compare XTDA energies/TDMs and all S0-S6 NAC pairs with BDF."""
    from pyscf import dft, gto, lib

    from XTDDFT_dev.XTDDFT.xtda import XTDA
    from XTDDFT_dev.utils.backend import set_backend

    set_backend('cpu')
    lib.num_threads(2)
    reference_energies, reference_nac = _load_bdf_ch2nh_reference()

    mol = gto.M(
        atom=CH2NH_GEOMETRY,
        basis='def2-svp',
        unit='Angstrom',
        charge=1,
        spin=1,
        symmetry=False,
        verbose=0,
    )
    mf = dft.ROKS(mol, xc='HF')
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged

    td = XTDA(mf, davidson=True, davidson_backend='cpu', so2st=False)
    td.kernel(nstates=10)
    assert np.all(td.converged)
    energy_error = np.abs(np.asarray(td.e)[:6] - reference_energies)
    print(
        f"BDF excitation energies: 6 states; "
        f"max |error| = {energy_error.max():.3e} hartree"
    )
    np.testing.assert_allclose(
        np.asarray(td.e)[:6], reference_energies, rtol=0.0, atol=2e-7
    )

    np.testing.assert_allclose(
        td.transition_density_matrix(0, 1),
        td.transition_density_matrix(1, 0).conj().T,
    )

    reference_data = np.load(REFERENCE_DATA)
    tdm_pairs = [tuple(pair) for pair in reference_data['tdm_state_pairs']]
    calculated_tdm = np.asarray(
        [td.transition_density_matrix(state_i, state_j) for state_i, state_j in tdm_pairs]
    )
    reference_tdm = reference_data['tdm_alpha_plus_beta']
    # BDF and PySCF may choose different signs for individual MOs.  A MO
    # phase change transforms gamma as D @ gamma @ D, so absolute entries are
    # invariant and are the appropriate cross-program comparison here.
    tdm_error = np.abs(np.abs(calculated_tdm) - np.abs(reference_tdm))
    print(
        f"BDF transition RDMs: {len(tdm_pairs)} excited-state pairs, "
        f"matrix shape {calculated_tdm.shape[1:]}; "
        f"max |absolute-entry error| = {tdm_error.max():.3e}"
    )
    np.testing.assert_allclose(
        np.abs(calculated_tdm),
        np.abs(reference_tdm),
        rtol=0.0,
        atol=1.0e-5,
    )

    result = td.nac_method(pairs=PHYSICAL_STATE_PAIRS, step=1e-4).kernel()
    calculated_nac = np.array([result[pair] for pair in PHYSICAL_STATE_PAIRS])

    # Each electronic eigenvector has an arbitrary global sign.  Align one sign
    # per state pair before comparing the complete 5-atom Cartesian vector.
    pair_phase = np.where(
        np.einsum('kax,kax->k', calculated_nac, reference_nac) < 0.0,
        -1.0,
        1.0,
    )
    phase_aligned_reference = reference_nac * pair_phase[:, None, None]
    nac_error = np.abs(calculated_nac - phase_aligned_reference)
    ground_pairs = np.array([state_i == 0 for state_i, _ in PHYSICAL_STATE_PAIRS])
    print(
        f"BDF raw NACMEs: {len(PHYSICAL_STATE_PAIRS)} state pairs, "
        f"{calculated_nac.size} Cartesian components"
    )
    print(
        f"ground-excited max |error| = {nac_error[ground_pairs].max():.3e} bohr^-1"
    )
    print(
        f"excited-excited max |error| = "
        f"{nac_error[~ground_pairs].max():.3e} bohr^-1"
    )
    np.testing.assert_allclose(
        calculated_nac,
        phase_aligned_reference,
        rtol=0.0,
        atol=5e-3,
    )
