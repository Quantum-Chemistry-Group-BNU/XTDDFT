from pathlib import Path
from copy import deepcopy
from functools import lru_cache
import sys
from io import StringIO
import re

import numpy as np
import pytest
from pyscf import dft, gto


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT.XTDDFT.sf_tda_up import SF_TDA_up
from XTDDFT.XTDDFT.xsf_tda_down import XSF_TDA_down
from XTDDFT.XTDDFT.xtda import XTDA
from XTDDFT.utils import unit


# Small triplet CH2 reference with closed, open, and virtual orbital blocks.
ATOM = (
    ("C", (0.00,  0.00,  0.00)),
    ("H", (0.00,  0.95,  0.15)),
    ("H", (0.10, -0.75,  0.65)),
)
CASES = (
    ("roks", "sc", 0),
    ("uks", "sc", 0),
    ("roks", "sfu", 1),
    ("uks", "sfu", 1),
    ("roks", "sfd", 1),
    ("uks", "sfd", 2),
)


def build_mf(reference, atom=ATOM):
    mol = gto.M(
        atom=atom,
        unit="Angstrom",
        spin=2,
        charge=0,
        basis="sto3g",
        verbose=0,
    )
    mf = dft.ROKS(mol) if reference == "roks" else dft.UKS(mol)
    mf.xc = "b3lyp"
    mf.conv_tol = 1e-11
    mf.grids.level = 4  # Level 3 fails the existing total-gradient tolerance.
    mf.kernel()
    assert mf.converged
    return mf


@pytest.fixture(scope="module")
def solve_td():
    # The six channels share two references at each of three geometries.
    cached_mf = lru_cache(maxsize=6)(build_mf)

    def solve(reference, channel, method, atom=ATOM):
        geometry = tuple((symbol, tuple(coord)) for symbol, coord in atom)
        # Solvers may mutate their reference: never expose the cached object.
        mf = deepcopy(cached_mf(reference, geometry))
        if channel == "sc":
            td = XTDA(mf, davidson=True)
        elif channel == "sfu":
            td = SF_TDA_up(mf, method=method, davidson=True, collinear_samples=20)
        else:
            td = XSF_TDA_down(mf, method=method, davidson=True, collinear_samples=20)
        td.kernel(nstates=1)
        return td

    yield solve
    cached_mf.cache_clear()


@pytest.mark.parametrize("reference,channel,method", CASES)
def test_analytic_gradient_smoke(reference, channel, method, solve_td):
    td = solve_td(reference, channel, method)
    mo_coeff = np.array(td.mf.mo_coeff, copy=True)
    mo_occ = np.array(td.mf.mo_occ, copy=True)

    gradient_method = td.nuc_grad_method(state=1)
    gradient_method.verbose = 4
    gradient_method.stdout = StringIO()
    gradient = gradient_method.kernel()
    output = gradient_method.stdout.getvalue()
    solver = "GMRES inner" if reference == "roks" else "UCPHF Krylov"
    match = re.search(r"Z-vector iterations \(" + solver + r"\): (\d+)", output)
    assert match is not None, output
    assert int(match[1]) > 0
    if reference == "roks":
        matvecs = re.search(r"matvec calls: (\d+)", output)
        assert int(matvecs[1]) >= int(match[1])

    assert gradient.shape == (len(ATOM), 3)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(td.mf.mo_coeff, mo_coeff)
    np.testing.assert_allclose(td.mf.mo_occ, mo_occ)


def test_uks_sc_analytic_gradient_honors_nontrivial_atmlst(solve_td):
    td = solve_td("uks", "sc", 0)
    gradient_method = td.nuc_grad_method(state=1)

    full = gradient_method.kernel()
    selected = gradient_method.kernel(atmlst=[2, 0])

    assert selected.shape == (2, 3)
    np.testing.assert_allclose(selected, full[[2, 0]], atol=1e-12, rtol=1e-12)


def displaced_atom(atom_index, axis, displacement):
    atom = [(symbol, np.array(coord, dtype=float)) for symbol, coord in ATOM]
    atom[atom_index][1][axis] += displacement
    return atom


@pytest.mark.slow
@pytest.mark.parametrize("reference,channel,method", CASES)
def test_analytic_gradient_matches_one_coordinate_finite_difference(
    reference, channel, method, solve_td
):
    atom_index, axis, step = 1, 1, 1e-3
    td = solve_td(reference, channel, method)
    analytic = td.nuc_grad_method(state=1).kernel()[atom_index, axis]

    td_plus = solve_td(
        reference, channel, method, displaced_atom(atom_index, axis, step)
    )
    td_minus = solve_td(
        reference, channel, method, displaced_atom(atom_index, axis, -step)
    )
    energy_plus = td_plus.mf.e_tot + td_plus.e[0]
    energy_minus = td_minus.mf.e_tot + td_minus.e[0]
    finite_difference = (energy_plus - energy_minus) / (2 * step / unit.bohr)

    assert analytic == pytest.approx(finite_difference, abs=5e-4, rel=5e-3)


@pytest.mark.parametrize("channel", ["sfd", "sfu"])
@pytest.mark.parametrize("omega", [None, 0.3])
@pytest.mark.parametrize("reference", ["uks", "roks"])
def test_df_dense_sf_exchange_matches_df_jk(channel, omega, reference):
    """Dense SF must use the same fitted exchange as the Davidson response."""
    from scipy.linalg import fractional_matrix_power
    from XTDDFT.XTDDFT.sf_tda_up import add_hf_a_b2a
    from XTDDFT.XTDDFT.xsf_tda_down import add_hf_a_a2b

    mol = gto.M(atom=ATOM, spin=2, basis="sto3g", verbose=0)
    mf = (dft.ROKS(mol) if reference == "roks" else dft.UKS(mol)).density_fit(auxbasis="weigend")
    coeff = fractional_matrix_power(mol.intor("int1e_ovlp"), -0.5)
    mf.mo_coeff = coeff if reference == "roks" else np.stack((coeff, coeff))
    occupied, virtual = coeff[:, :2], coeff[:, 2:5]
    build = add_hf_a_a2b if channel == "sfd" else add_hf_a_b2a
    actual = build(np.zeros((2, 3, 2, 3)), mf, occupied, virtual,
                   2, 3, hyb=0.4, omega=omega)
    expected = np.empty_like(actual)
    for j in range(2):
        for b in range(3):
            dm = np.outer(virtual[:, b], occupied[:, j])
            k = mf.get_k(mol, dm, hermi=0, omega=omega)
            expected[:, :, j, b] = -0.4 * (virtual.T @ k @ occupied).T
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_uks_sfd_pure_functional_gradient_smoke():
    """Unequal alpha/beta occupied sizes expose a transposed Q contraction."""
    mol = gto.M(atom=ATOM, spin=2, basis="sto3g", verbose=0)
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    td = XSF_TDA_down(mf, method=2, davidson=True)
    td.kernel(nstates=1)
    gradient = td.nuc_grad_method().kernel()
    assert gradient.shape == (3, 3)
    assert np.isfinite(gradient).all()


@pytest.fixture
def gpu_df_backend():
    """Enable CUDA only for the explicit server-side DF regression cases."""
    cp = pytest.importorskip("cupy")
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime is unavailable")
    if device_count == 0:
        pytest.skip("No CUDA device")
    pytest.importorskip("gpu4pyscf")
    from XTDDFT.utils.backend import backend, set_backend
    previous = backend.mode
    set_backend("gpu")
    try:
        yield cp
    finally:
        set_backend(previous)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("channel,method", [("sc", 0), ("sfd", 1), ("sfu", 1),
                                            ("sfd", 2), ("sfu", 2)])
@pytest.mark.parametrize("xc", ["pbe", "b3lyp", "cam-b3lyp"])
@pytest.mark.parametrize("reference", ["uks", "roks"])
def test_gpu_df_gradient_finite_difference(gpu_df_backend, channel, method, xc, reference):
    """Run on CUDA: pure/hybrid/RSH, both SF kernels, and SC with the same DF energy."""
    cp = gpu_df_backend

    def solve(atom):
        mol = gto.M(atom=atom, spin=2, basis="sto3g", unit="Angstrom", verbose=0)
        mf = (dft.ROKS(mol) if reference == "roks" else dft.UKS(mol))
        mf = mf.to_gpu().density_fit(auxbasis="weigend")
        mf.xc = xc
        mf.grids.level = 4
        mf.conv_tol = 1e-12
        mf.max_cycle = 200
        mf.kernel()
        assert mf.converged
        if channel == "sc":
            td = XTDA(mf, davidson=True)
        elif channel == "sfd":
            td = XSF_TDA_down(mf, method=method, davidson=True, collinear_samples=24)
        else:
            td = SF_TDA_up(mf, method=method, davidson=True, collinear_samples=24)
        td.kernel(nstates=3)
        assert np.asarray(td.converged).all()
        return td

    def scalar(value):
        return float(cp.asnumpy(value)) if isinstance(value, cp.ndarray) else float(value)

    td = solve(ATOM)
    grad = td.nuc_grad_method(state=1)
    full = cp.asnumpy(grad.kernel())
    assert np.isfinite(full).all()
    np.testing.assert_allclose(cp.asnumpy(grad.kernel(atmlst=[2, 0])),
                               full[[2, 0]], atol=1e-8, rtol=1e-8)
    # A root crossing would invalidate a fixed-root finite difference.
    assert scalar(td.e[1] - td.e[0]) > 1e-5
    step = 1e-3  # Angstrom; returned gradient is Hartree/Bohr.
    plus = solve(displaced_atom(1, 1, step))
    minus = solve(displaced_atom(1, 1, -step))
    def ao_transitions(state):
        # Compare physical transition functions, not MO coefficients whose
        # phases may change independently at the displaced geometries.
        from XTDDFT.utils.backend import asnumpy
        coeff = asnumpy(state.ctx.mo_coeff)
        occ = asnumpy(state.ctx.mo_occ)
        if channel == "sc":
            if reference == "roks":
                a, b = state._full_spin_conserving_amplitudes(asnumpy(state.v[:, :1]))
                amplitudes = (a[0], b[0])
            else:
                amplitudes = state._unrestricted_ground_amplitudes(0)
            blocks = [(0, 0, amplitudes[0]), (1, 1, amplitudes[1])]
        elif channel == "sfd":
            blocks = [(0, 1, state._spinflip_amplitude_matrix_u(0))]
        else:
            blocks = [(1, 0, state._state_amplitude(0))]
        return [coeff[h][:, occ[h] > 0] @ amplitude
                @ coeff[p][:, occ[p] == 0].T for h, p, amplitude in blocks]

    reference_transitions = ao_transitions(td)
    def norm(state, transitions):
        overlap = state.mol.intor("int1e_ovlp")
        return sum(np.einsum("ij,ij->", t, overlap @ t @ overlap.T) for t in transitions)
    for other in (plus, minus):
        cross = gto.intor_cross("int1e_ovlp", td.mol, other.mol)
        other_transitions = ao_transitions(other)
        overlap = abs(sum(np.einsum("ij,ij->", left, cross @ right @ cross.T)
                          for left, right in zip(reference_transitions, other_transitions)))
        overlap /= np.sqrt(norm(td, reference_transitions) * norm(other, other_transitions))
        assert overlap > 0.9, "Possible root change: inspect displaced states"
    fd = (scalar(plus.mf.e_tot + plus.e[0])
          - scalar(minus.mf.e_tot + minus.e[0])) / (2 * step / unit.bohr)
    assert full[1, 1] == pytest.approx(fd, abs=5e-4, rel=5e-3)


def test_roks_df_correction_integrals_match_reference():
    """XSFD's dense correction must fit both its HF Fock and four-index ERIs."""
    from XTDDFT.XTDDFT.base import _make_rohf_reference_mf, _ao2mo_full_gamma
    from scipy.linalg import fractional_matrix_power

    mol = gto.M(atom=ATOM, spin=2, basis="sto3g", verbose=0)
    mf = dft.ROKS(mol).density_fit(auxbasis="weigend")
    coeff = fractional_matrix_power(mol.intor("int1e_ovlp"), -0.5)
    dm = coeff[:, :3] @ coeff[:, :3].T
    hf = _make_rohf_reference_mf(mf)
    np.testing.assert_allclose(hf.get_jk(mol, dm), mf.get_jk(mol, dm), atol=1e-10)
    expected = mf.with_df.ao2mo([coeff] * 4, compact=False)
    np.testing.assert_allclose(_ao2mo_full_gamma(mf, coeff).reshape(expected.shape),
                               expected, atol=1e-10)


@pytest.mark.parametrize("channel", ["sc", "sfd", "sfu"])
def test_roks_df_dense_and_davidson_energies_agree(channel):
    mol = gto.M(atom=ATOM, spin=2, basis="sto3g", verbose=0)
    mf = dft.ROKS(mol).density_fit(auxbasis="weigend")
    mf.xc = "b3lyp"
    mf.grids.level = 1
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    energies = []
    for davidson in (False, True):
        if channel == "sc":
            td = XTDA(mf, davidson=davidson, use_delta_a=True)
        elif channel == "sfd":
            td = XSF_TDA_down(mf, method=2, davidson=davidson, SA=3)
        else:
            td = SF_TDA_up(mf, method=2, davidson=davidson)
        td.kernel(nstates=3)
        energies.append(np.asarray(td.e)[:3])
    np.testing.assert_allclose(*energies, atol=1e-8, rtol=1e-7)


@pytest.mark.gpu
def test_gpu_xsc_correction_pair_normalization(gpu_df_backend):
    """The mixed-pair correction equals polarization of the old RHF expression."""
    from XTDDFT.XTDDFT.grad.gradient_roks_sc import jk_energies_per_atom
    from gpu4pyscf.df.grad import tduks as df_grad
    from gpu4pyscf.grad import tduks as direct_grad
    from types import SimpleNamespace

    cp = gpu_df_backend
    mol = gto.M(atom=ATOM, spin=2, basis="sto3g", verbose=0)
    rng = np.random.default_rng(17)
    a, b = cp.asarray(rng.normal(size=(2, mol.nao, mol.nao)))
    a, b = a + a.T, b + b.T
    for fitted in (False, True):
        mf = dft.ROKS(mol).to_gpu()
        if fitted:
            mf = mf.density_fit(auxbasis="weigend")
            mf.with_df.build()
        grad = SimpleNamespace(base=SimpleNamespace(_scf=mf), auxbasis_response=True)
        get_veff = (df_grad if fitted else direct_grad).Gradients.get_veff
        def quadratic(dm):
            # A closed-shell density has two equal spin components.
            spin_dm = cp.stack((dm / 2, dm / 2)) if fitted else dm
            return get_veff(grad, mol, spin_dm, j_factor=0., k_factor=-1., hermi=1)
        expected = 4 * (quadratic(a + b) - quadratic(a) - quadratic(b))
        actual = jk_energies_per_atom(mf, [[a, b]], [0.], [-4.], hermi=1,
                                      sum_results=True)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.slow
@pytest.mark.parametrize("channel,method", [("sc", 0), ("sfd", 1), ("sfu", 1),
                                            ("sfd", 2), ("sfu", 2)])
@pytest.mark.parametrize("xc", ["pbe", "b3lyp", "cam-b3lyp"])
@pytest.mark.parametrize("reference", ["uks", "roks"])
def test_cpu_df_gradient_finite_difference(channel, method, xc, reference):
    def solve(atom):
        mol = gto.M(atom=atom, spin=2, basis="sto3g", unit="Angstrom", verbose=0)
        mf = (dft.ROKS(mol) if reference == "roks" else dft.UKS(mol))
        mf = mf.density_fit(auxbasis="weigend")
        mf.xc, mf.grids.level, mf.conv_tol = xc, 4, 1e-12
        mf.kernel()
        assert mf.converged
        if channel == "sc":
            td = XTDA(mf, davidson=True)
        elif channel == "sfd":
            td = XSF_TDA_down(mf, method=method, davidson=True, collinear_samples=24)
        else:
            td = SF_TDA_up(mf, method=method, davidson=True, collinear_samples=24)
        td.kernel(nstates=2)
        assert np.asarray(td.converged).all()
        assert td.e[1] - td.e[0] > 1e-5
        return td

    td = solve(ATOM)
    gradient = td.nuc_grad_method(state=1)
    full = gradient.kernel()
    assert np.isfinite(full).all()
    np.testing.assert_allclose(gradient.kernel(atmlst=[2, 0]), full[[2, 0]], atol=1e-9)
    step = 1e-3
    plus = solve(displaced_atom(1, 1, step))
    minus = solve(displaced_atom(1, 1, -step))
    fd = (plus.mf.e_tot + plus.e[0] - minus.mf.e_tot - minus.e[0]) / (2 * step / unit.bohr)
    assert full[1, 1] == pytest.approx(fd, abs=5e-4, rel=5e-3)


@pytest.mark.parametrize("spin_conserving", [False, True])
@pytest.mark.parametrize("omega", [0., 0.3])
def test_cpu_df_aux_response_matches_auxiliary_center_finite_difference(spin_conserving, omega):
    """Move only an auxiliary center: isolate the three-center/metric response."""
    from pyscf import df
    from XTDDFT.XTDDFT.grad._backend import cpu_df_get_jk, cpu_df_td_aux

    mol = gto.M(atom=ATOM, spin=2, basis="sto3g", verbose=0)
    mf = dft.UKS(mol).density_fit(auxbasis="weigend")
    mf.with_df.build()
    auxmol = mf.with_df.auxmol
    rng = np.random.default_rng(21)
    raw = rng.normal(size=(6, mol.nao, mol.nao)) * 0.1
    pa, pb = (d @ d.T for d in raw[:2])
    qa, qb = (d + d.T for d in raw[2:4])
    xa, xb = raw[4:]
    scale = 0.5 if spin_conserving else 0.25
    densities = np.stack((pa, qa, xa, xa.T, pb, qb, xb, xb.T)
                         if spin_conserving else (pa, qa, pb, qb))
    hyb, alpha = 0.2, 0.65
    vj, vk = cpu_df_get_jk(mf.nuc_grad_method(), densities, hyb, alpha, omega)
    actual = cpu_df_td_aux(vj, vk, scale)
    if not spin_conserving:
        _, vkx = cpu_df_get_jk(mf.nuc_grad_method(), np.stack((xa, xa.T)),
                               hyb, alpha, omega, with_j=False)
        actual -= vkx.aux[0, 1] + vkx.aux[1, 0]

    def energy(aux):
        def eri(w):
            with mol.with_range_coulomb(w), aux.with_range_coulomb(w):
                three = df.incore.aux_e2(mol, aux, intor="int3c2e", aosym="s1")
                metric = aux.intor("int2c2e")
            three = three.reshape(mol.nao**2, -1)
            return (three @ np.linalg.solve(metric, three.T)).reshape((mol.nao,) * 4)
        full = eri(0.)
        exchange = hyb * full
        if omega:
            exchange += (alpha - hyb) * eri(omega)
        def j(a, b):
            return np.einsum("ijkl,ji,lk->", full, a, b)
        def k(a, b):
            return np.einsum("ijkl,li,jk->", exchange, a, b)
        value = 0.5*j(pa+pb, pa+pb) + scale*j(pa+pb, qa+qb)
        value -= 0.5*(k(pa, pa)+k(pb, pb)) + scale*(k(pa, qa)+k(pb, qb))
        value -= k(xa, xa.T)
        if spin_conserving:
            value += 2*j(xa+xb, xa+xb) - k(xb, xb.T)
        return value

    step = 1e-4  # Bohr, fixed AO centers and fixed AO densities.
    coords = auxmol.atom_coords()
    plus, minus = coords.copy(), coords.copy()
    plus[1, 1] += step
    minus[1, 1] -= step
    finite_difference = (energy(auxmol.set_geom_(plus, unit="Bohr", inplace=False))
                         - energy(auxmol.set_geom_(minus, unit="Bohr", inplace=False))) / (2*step)
    assert actual[1, 1] == pytest.approx(finite_difference, abs=2e-8, rel=2e-6)


@pytest.mark.parametrize('sum_results', [False, True])
@pytest.mark.parametrize('omega', [None, 0.3])
def test_gpu_df_derivative_pairs_are_streamed(monkeypatch, sum_results, omega):
    """Preserve pair ordering/scaling while isolating the pure-J density pair."""
    from types import SimpleNamespace
    from XTDDFT.XTDDFT.grad._backend import df_jk_energies_per_atom
    calls = []
    x = np.array([[1., 2.], [3., 4.]])
    pairs = [[x, x], [2*x, x], [x, x.T]]
    jf, kf = [0.5, 0., 0.], [0., 0.2, 0.4]

    def native(grad, dm_list, j_factor, k_factor, **kwargs):
        assert grad.auxbasis_response
        assert len(dm_list) == 1
        calls.append((dm_list, kwargs))
        j = 0 if j_factor is None else j_factor[0]
        k = 0 if k_factor is None else k_factor[0]
        a, b = dm_list[0]
        # Deliberately sensitive to X versus X.T and density-pair ordering.
        value = j * a.sum() * b.sum() - k * np.sum(a * b.T)
        result = np.full((1, 2, 3), value)
        return result.sum(axis=0) if kwargs['sum_results'] else result

    monkeypatch.setitem(sys.modules, 'gpu4pyscf.df.grad.tdrhf',
                        SimpleNamespace(Gradients=SimpleNamespace(jk_energies_per_atom=native)))
    actual = df_jk_energies_per_atom(object(), pairs, jf, kf,
                                   omega=omega, sum_results=sum_results)
    expected = np.stack([np.full((2, 3), j*a.sum()*b.sum()-k*np.sum(a*b.T))
                         for (a, b), j, k in zip(pairs, jf, kf)])
    np.testing.assert_allclose(actual, expected.sum(axis=0) if sum_results else expected)
    assert len(calls) == 3
    assert calls[2][0][0][1] is x.T or np.array_equal(calls[2][0][0][1], x.T)
    assert all(c[1]['omega'] == (0. if omega is None else omega) for c in calls)


@pytest.mark.gpu
@pytest.mark.parametrize('sum_results', [False, True])
def test_gpu_df_streamed_matches_native(gpu_df_backend, sum_results):
    from types import SimpleNamespace
    from gpu4pyscf.df.grad.tdrhf import Gradients
    from XTDDFT.XTDDFT.grad._backend import df_jk_energies_per_atom
    cp = gpu_df_backend
    mol = gto.M(atom=ATOM, spin=2, basis='sto3g', verbose=0)
    mf = dft.UKS(mol).to_gpu().density_fit(auxbasis='weigend')
    mf.with_df.build()
    rng = np.random.default_rng(7)
    a, b, x = cp.asarray(rng.normal(size=(3, mol.nao, mol.nao)))
    a, b = a + a.T, b + b.T
    pairs = [[a, b], [a, a], [x, x.T]]
    jf, kf = [0.5, 0., 0.], [0., 0.2, 0.4]
    grad = SimpleNamespace(base=SimpleNamespace(_scf=mf), auxbasis_response=True)
    expected = Gradients.jk_energies_per_atom(
        grad, pairs, jf, kf, sum_results=sum_results)
    actual = df_jk_energies_per_atom(
        mf, pairs, jf, kf, sum_results=sum_results)
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
