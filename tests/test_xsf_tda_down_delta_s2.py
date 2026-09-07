from pathlib import Path
import inspect
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import xsf_tda_down


class XsfTdaDownDeltaS2Test(unittest.TestCase):
    def make_method(self, nstates=3):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.mf = object()
        method.ctx = object()
        method.type_u = True
        method.SA = 0
        method.re = False
        method.nc = 1
        method.no = 1
        method.nv = 2
        method.nstates = nstates
        method.v = np.arange(6 * nstates, dtype=float).reshape(6, nstates) / 10.0
        return method

    def test_preconditioner_uses_separate_diag_j_batch_size(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 2
        method.nv = 1
        method.nocc_a = 3
        method.nocc_b = 2
        method.SA = 3
        method.type_u = False
        method.delta_a_jk_batch_size = 7
        method.delta_a_diag_j_batch_size = 3
        seen = []

        def response_j_diagonals(batch_size=None):
            seen.append(batch_size)
            return np.zeros((method.nc, method.no)), np.zeros((method.no, method.nv))

        method._response_j_diagonals = response_j_diagonals
        fock_a = np.diag(np.arange(1.0, 5.0))
        fock_b = np.diag(np.arange(2.0, 6.0))
        fock_a_hf = np.diag(np.arange(3.0, 7.0))
        fock_b_hf = np.diag(np.arange(4.0, 8.0))

        method._build_preconditioner_hdiag(fock_a, fock_b, fglobal=1.0, fockA_hf=fock_a_hf, fockB_hf=fock_b_hf)

        self.assertEqual(seen, [3])

    def test_preconditioner_diag_j_batch_size_defaults_to_auto(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 2
        method.nv = 1
        method.nocc_a = 3
        method.nocc_b = 2
        method.SA = 3
        method.type_u = False
        method.delta_a_jk_batch_size = 5
        method.delta_a_diag_j_batch_size = None
        seen = []

        def response_j_diagonals(batch_size=None):
            seen.append(batch_size)
            return np.zeros((method.nc, method.no)), np.zeros((method.no, method.nv))

        method._response_j_diagonals = response_j_diagonals
        fock_a = np.diag(np.arange(1.0, 5.0))
        fock_b = np.diag(np.arange(2.0, 6.0))
        fock_a_hf = np.diag(np.arange(3.0, 7.0))
        fock_b_hf = np.diag(np.arange(4.0, 8.0))

        method._build_preconditioner_hdiag(fock_a, fock_b, fglobal=1.0, fockA_hf=fock_a_hf, fockB_hf=fock_b_hf)

        self.assertEqual(seen, [None])

    def test_debug_sa0_hdiag_skips_delta_a_diagonal(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 2
        method.nv = 1
        method.nocc_a = 3
        method.nocc_b = 2
        method.SA = 1
        method.type_u = False
        method.debug_sa0_hdiag = True
        method._response_j_diagonals = lambda **_: self.fail("Delta-A hdiag was evaluated")
        fock_a = np.diag(np.arange(1.0, 6.0))
        fock_b = np.diag(np.arange(2.0, 7.0))

        hdiag = method._build_preconditioner_hdiag(
            fock_a, fock_b, fglobal=1.0,
            fockA_hf=fock_a + 10.0, fockB_hf=fock_b + 10.0,
        )
        expected = np.array([5.0, 3.0, 4.0, 4.0, 3.0, 2.0, 3.0, 1.0, 2.0])

        self.assertTrue(np.allclose(hdiag, expected))

    def test_df_diagonal_backend_option_is_validated(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.mf = SimpleNamespace(mo_coeff=np.eye(2))

        old_base_init = xsf_tda_down.XTDDFT_base.__init__
        old_as_cpu_mf = xsf_tda_down._as_cpu_mf
        try:
            xsf_tda_down.XTDDFT_base.__init__ = lambda self, mf, method, davidson=True, df_cache=None: None
            xsf_tda_down._as_cpu_mf = lambda mf: SimpleNamespace(spin_square=lambda: (0.0, 3.0))

            xsf_tda_down.XSF_TDA_down.__init__(method, method.mf, method=1, delta_a_diag_df_backend="gpu")
            self.assertEqual(method.delta_a_diag_df_backend, "gpu")
            with self.assertRaises(ValueError):
                xsf_tda_down.XSF_TDA_down.__init__(method, method.mf, method=1, delta_a_diag_df_backend="bad")
        finally:
            xsf_tda_down.XTDDFT_base.__init__ = old_base_init
            xsf_tda_down._as_cpu_mf = old_as_cpu_mf


    def test_restricted_constructor_defaults_re_true(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.mf = SimpleNamespace(mo_coeff=np.eye(2))

        old_base_init = xsf_tda_down.XTDDFT_base.__init__
        old_as_cpu_mf = xsf_tda_down._as_cpu_mf
        try:
            xsf_tda_down.XTDDFT_base.__init__ = lambda self, mf, method, davidson=True, df_cache=None: None
            xsf_tda_down._as_cpu_mf = lambda mf: SimpleNamespace(spin_square=lambda: (0.0, 3.0))

            xsf_tda_down.XSF_TDA_down.__init__(method, method.mf, method=1)
        finally:
            xsf_tda_down.XTDDFT_base.__init__ = old_base_init
            xsf_tda_down._as_cpu_mf = old_as_cpu_mf

        self.assertTrue(method.re)

    def test_split_analysis_vectors_lazily_builds_removed_oo_basis(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 2
        method.nv = 1
        method.re = True
        value = np.arange(1 * 1 + 1 * 2 + 2 * 1 + 2 * 2 - 1, dtype=float)

        _cv, _co, _ov, oo = method._split_analysis_vectors(value)

        self.assertEqual(oo.shape, (2, 2))
        self.assertTrue(hasattr(method, "vects"))

    def test_pbc_df_response_j_diagonals_match_batched_j_path(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.nocc_a = 2
        method.nvir_b = 2
        method.occidx_a = np.array([0, 1])
        method.viridx_b = np.array([1, 2])
        method.mo_coeff = np.stack([np.eye(3), np.eye(3)])

        pair_idx = np.arange(9)
        cderi = np.arange(1, 37, dtype=float).reshape(4, 9) / 10.0
        method.mf = SimpleNamespace(
            cell=object(),
            with_df=SimpleNamespace(
                is_gamma_point=True,
                _cderi={0: cderi},
                _cderi_idx=(pair_idx, np.array([0, 4, 8])),
            ),
        )
        eri = np.einsum("Lp,Lq->pq", cderi, cderi).reshape(3, 3, 3, 3)

        def fake_get_j(mf, dm, hermi=0):
            return np.einsum("pqrs,xrs->xpq", eri, dm)

        old_get_j = xsf_tda_down._get_j
        try:
            xsf_tda_down._get_j = fake_get_j
            ref_co, ref_ov = method._response_j_diagonals(batch_size=1)
        finally:
            xsf_tda_down._get_j = old_get_j

        co_j, ov_j = method._response_j_diagonals_from_pbc_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

        method.delta_a_diag_method = "pbc_df"
        method.delta_a_diag_df_aux_batch_size = 2
        co_j, ov_j = method._response_j_diagonals(batch_size=1)

        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

    def test_molecular_df_response_j_diagonals_match_batched_j_path(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.nocc_a = 2
        method.nvir_b = 2
        method.occidx_a = np.array([0, 1])
        method.viridx_b = np.array([1, 2])
        method.mo_coeff = np.stack([np.eye(3), np.eye(3)])

        tri = np.tril_indices(3)
        cderi = np.arange(1, 25, dtype=float).reshape(4, 6) / 10.0
        cderi_full = np.zeros((4, 3, 3))
        cderi_full[:, tri[0], tri[1]] = cderi
        cderi_full[:, tri[1], tri[0]] = cderi
        method.mf = SimpleNamespace(
            mol=object(),
            with_df=SimpleNamespace(_cderi=cderi),
        )
        eri = np.einsum("Lpq,Lrs->pqrs", cderi_full, cderi_full)

        def fake_get_j(mf, dm, hermi=0):
            return np.einsum("pqrs,xrs->xpq", eri, dm)

        old_get_j = xsf_tda_down._get_j
        try:
            xsf_tda_down._get_j = fake_get_j
            ref_co, ref_ov = method._response_j_diagonals(batch_size=1)
        finally:
            xsf_tda_down._get_j = old_get_j

        co_j, ov_j = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

        method.delta_a_diag_method = "df"
        method.delta_a_diag_df_aux_batch_size = 2
        co_j, ov_j = method._response_j_diagonals(batch_size=1)

        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

    def test_molecular_gpu4pyscf_df_loop_tuple_matches_cderi_path(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.nocc_a = 2
        method.nvir_b = 2
        method.occidx_a = np.array([0, 1])
        method.viridx_b = np.array([1, 2])
        method.mo_coeff = np.stack([np.eye(3), np.eye(3)])
        method.delta_a_diag_df_backend = "cpu"

        tri = np.tril_indices(3)
        cderi = np.arange(1, 25, dtype=float).reshape(4, 6) / 10.0
        method.mf = SimpleNamespace(mol=object(), with_df=SimpleNamespace(_cderi=cderi))
        ref_co, ref_ov = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        class LoopDF:
            _cderi = None

            def __init__(self):
                self.intopt = SimpleNamespace(cderi_row=tri[0], cderi_col=tri[1])

            def loop(self, blksize=None, unpack=True):
                for p0 in range(0, cderi.shape[0], blksize):
                    yield (None, cderi[p0:p0 + blksize].T)

        method.mf = SimpleNamespace(mol=object(), with_df=LoopDF())
        co_j, ov_j = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

    def test_molecular_gpu_df_sorts_mo_coeff_to_cderi_ao_order(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        mo_coeff = np.arange(18.0).reshape(2, 3, 3)
        method.mo_coeff = mo_coeff
        tri = np.tril_indices(3)

        class IntOpt:
            def __init__(self):
                self.cderi_row = tri[0]
                self.cderi_col = tri[1]
                self.axes = []

            def sort_orbitals(self, mat, axis=None):
                self.axes.append(axis)
                return mat[:, [2, 0, 1], :]

        class LoopDF:
            _cderi = None

            def __init__(self):
                self.intopt = IntOpt()

            def loop(self, blksize=None, unpack=True):
                return iter(())

        LoopDF.__module__ = "gpu4pyscf.df.df"
        with_df = LoopDF()
        method.mf = SimpleNamespace(mol=object(), with_df=with_df)

        data = method._df_cderi_data(aux_batch_size=2)

        self.assertEqual(with_df.intopt.axes, [[1]])
        self.assertTrue(np.array_equal(data.mo_coeff, mo_coeff[:, [2, 0, 1], :]))

    def test_molecular_gpu_df_diagonal_reduces_all_device_cderi_slices(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.nocc_a = 2
        method.nvir_b = 2
        method.occidx_a = np.array([0, 1])
        method.viridx_b = np.array([1, 2])
        method.mo_coeff = np.stack([np.eye(3), np.eye(3)])

        tri = np.tril_indices(3)
        cderi = np.arange(1, 25, dtype=float).reshape(4, 6) / 10.0
        method.delta_a_diag_df_backend = "cpu"
        method.mf = SimpleNamespace(
            mol=object(), with_df=SimpleNamespace(_cderi=cderi)
        )
        ref_co, ref_ov = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1, aux_batch_size=1
        )

        current_device = threading.local()
        visited = set()

        class FakeDevice:
            def __init__(self, device_id=None):
                self.id = getattr(current_device, "value", 0) if device_id is None else device_id

            def __enter__(self):
                self.previous = getattr(current_device, "value", 0)
                current_device.value = self.id
                return self

            def __exit__(self, *_):
                current_device.value = self.previous

        fake_cp = SimpleNamespace(
            cuda=SimpleNamespace(
                Device=FakeDevice,
                runtime=SimpleNamespace(getDeviceCount=lambda: 2),
            ),
            asarray=np.asarray,
            result_type=np.result_type,
            zeros=np.zeros,
            einsum=np.einsum,
        )

        class MultiGpuDF:
            def __init__(self):
                self._cderi = [cderi[:2], cderi[2:]]
                self.intopt = SimpleNamespace(cderi_row=tri[0], cderi_col=tri[1])

            def loop(self, blksize=None, unpack=True):
                device_id = FakeDevice().id
                visited.add(device_id)
                shard = self._cderi[device_id]
                for p0 in range(0, shard.shape[0], blksize):
                    yield shard[p0:p0 + blksize]

        MultiGpuDF.__module__ = "gpu4pyscf.df.df"
        method.delta_a_diag_df_backend = "gpu"
        method.mf = SimpleNamespace(mol=object(), with_df=MultiGpuDF())
        with patch.object(xsf_tda_down, "require_cupy", return_value=fake_cp):
            co_j, ov_j = method._response_j_diagonals_from_df(
                mo_pair_batch_size=1, aux_batch_size=1
            )

        self.assertEqual(visited, {0, 1})
        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

    def test_df_response_accepts_tuple_blocks_from_data_source(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.nocc_a = 2
        method.nvir_b = 2
        method.occidx_a = np.array([0, 1])
        method.viridx_b = np.array([1, 2])
        method.mo_coeff = np.stack([np.eye(3), np.eye(3)])
        method.delta_a_diag_df_backend = "cpu"

        tri = np.tril_indices(3)
        cderi = np.arange(1, 25, dtype=float).reshape(4, 6) / 10.0
        method.mf = SimpleNamespace(mol=object(), with_df=SimpleNamespace(_cderi=cderi))
        ref_co, ref_ov = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        def iter_blocks():
            for p0 in range(0, cderi.shape[0], 2):
                yield None, cderi[p0:p0 + 2].T

        data = SimpleNamespace(
            mo_coeff=method.mo_coeff,
            pair_p=tri[0],
            pair_q=tri[1],
            symmetric_pairs=True,
            iter_blocks=iter_blocks,
        )
        method._df_cderi_data = lambda aux_batch_size=None: data

        co_j, ov_j = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        self.assertTrue(np.allclose(co_j, ref_co))
        self.assertTrue(np.allclose(ov_j, ref_ov))

    def test_gpu_df_backend_matches_cpu_df_backend(self):
        try:
            import cupy as cp

            cp.cuda.runtime.getDeviceCount()
        except Exception as err:
            self.skipTest(f"CUDA is not available: {err}")

        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.nc = 1
        method.no = 1
        method.nv = 1
        method.nocc_a = 2
        method.nvir_b = 2
        method.occidx_a = np.array([0, 1])
        method.viridx_b = np.array([1, 2])
        method.mo_coeff = np.stack([np.eye(3), np.eye(3)])
        method.delta_a_diag_df_aux_batch_size = 2

        cderi = np.arange(1, 25, dtype=float).reshape(4, 6) / 10.0
        method.mf = SimpleNamespace(mol=object(), with_df=SimpleNamespace(_cderi=cderi))

        method.delta_a_diag_df_backend = "cpu"
        cpu_co, cpu_ov = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )
        method.delta_a_diag_df_backend = "gpu"
        gpu_co, gpu_ov = method._response_j_diagonals_from_df(
            mo_pair_batch_size=1,
            aux_batch_size=2,
        )

        self.assertTrue(np.allclose(gpu_co, cpu_co))
        self.assertTrue(np.allclose(gpu_ov, cpu_ov))

    def test_hdiag_checkpoint_round_trip_checks_davidson_size(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        hdiag = np.array([1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hdiag.npy"

            method._save_hdiag_file(path, hdiag)
            loaded = method._load_hdiag_file(path, expected_size=3)

            self.assertTrue(np.allclose(loaded, hdiag))
            with self.assertRaises(ValueError):
                method._load_hdiag_file(path, expected_size=4)

    def test_uks_delta_s2_reuses_cpu_overlap_context(self):
        method = self.make_method()

        mo_coeff = np.stack([np.eye(4), np.eye(4)])
        ctx = SimpleNamespace(
            mo_coeff=mo_coeff,
            occidx_a=np.array([0, 1]),
            occidx_b=np.array([0]),
            viridx_a=np.array([2, 3]),
            viridx_b=np.array([1, 2, 3]),
        )
        calls = {"mf": 0, "ctx": 0, "ovlp": 0}

        def as_cpu_mf(mf):
            calls["mf"] += 1
            return mf

        def as_cpu_ctx(mf, old_ctx):
            calls["ctx"] += 1
            return ctx

        def get_ovlp(mf):
            calls["ovlp"] += 1
            return np.eye(4)

        old_as_cpu_mf = xsf_tda_down._as_cpu_mf
        old_as_cpu_ctx = xsf_tda_down._as_cpu_ctx
        old_get_ovlp = xsf_tda_down._get_ovlp
        try:
            xsf_tda_down._as_cpu_mf = as_cpu_mf
            xsf_tda_down._as_cpu_ctx = as_cpu_ctx
            xsf_tda_down._get_ovlp = get_ovlp

            ds2 = method.deltaS2()
        finally:
            xsf_tda_down._as_cpu_mf = old_as_cpu_mf
            xsf_tda_down._as_cpu_ctx = old_as_cpu_ctx
            xsf_tda_down._get_ovlp = old_get_ovlp

        self.assertEqual(ds2.shape, (3,))
        self.assertTrue(np.all(np.isfinite(ds2)))
        self.assertEqual(calls, {"mf": 1, "ctx": 1, "ovlp": 1})

    def test_delta_s2_optimized_contraction_matches_direct_formula(self):
        method = self.make_method(nstates=1)
        x_cv, x_co, x_ov, x_oo = method._split_analysis_vectors(method.v[:, 0])
        x_ba = np.concatenate([np.hstack([x_co, x_cv]), np.hstack([x_oo, x_ov])], axis=0).T
        sba_oo = np.array([[0.2 + 0.1j, -0.3 + 0.4j]])
        sba_vo = np.array([
            [0.5 - 0.2j, 0.1 + 0.3j],
            [-0.4 + 0.6j, 0.2 - 0.1j],
            [0.7 + 0.1j, -0.5 + 0.2j],
        ])

        expected = (
            xsf_tda_down.contract("ai,aj,jk,ki", x_ba.conj(), x_ba, sba_oo.T.conj(), sba_oo)
            - xsf_tda_down.contract("ai,bi,kb,ak", x_ba.conj(), x_ba, sba_vo.T.conj(), sba_vo)
            + xsf_tda_down.contract("ai,ai->", x_ba.conj(), sba_vo)
            * xsf_tda_down.contract("ai,ai->", x_ba, sba_vo.conj())
        )
        actual = method._deltaS2_U_from_overlaps(0, sba_oo, sba_vo)

        self.assertTrue(np.allclose(actual, expected))

    def test_delta_s2_u_helpers_use_wrapped_contract(self):
        source = (
            inspect.getsource(xsf_tda_down.XSF_TDA_down._deltaS2_U_overlaps)
            + inspect.getsource(xsf_tda_down.XSF_TDA_down._deltaS2_U_from_overlaps)
        )

        self.assertNotIn("np.einsum", source)
        self.assertNotIn("np.vdot", source)
        self.assertNotIn(" @ ", source)
        self.assertIn("contract(", source)

    def test_uks_transition_dipole_uses_wrapped_contract(self):
        source = inspect.getsource(xsf_tda_down.XSF_TDA_down._transition_dipole_matrix_u)

        self.assertNotIn("np.einsum", source)
        self.assertIn("contract(", source)

    def test_restricted_transition_density_contracts_to_transition_dipole(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.type_u = False
        method.re = False
        method.SA = 3
        method.nc = 1
        method.no = 2
        method.nv = 1
        method.ground_s = 1.0
        method.nstates = 2
        dim = (method.nc + method.no) * (method.no + method.nv)
        method.v = np.arange(1, dim * method.nstates + 1, dtype=float).reshape(dim, method.nstates) / 10.0

        nmo = method.nc + method.no + method.nv
        ints_mo = np.arange(1, 3 * nmo * nmo + 1, dtype=float).reshape(3, nmo, nmo) / 17.0
        method._dipole_mo_integrals = lambda: (ints_mo, None)

        gamma = method.transition_density_matrix(0, 1)
        tdm = method._transition_dipole_matrix_r()
        contracted = xsf_tda_down.contract("mn,xmn->x", gamma, ints_mo)

        self.assertEqual(gamma.shape, (nmo, nmo))
        self.assertTrue(np.allclose(contracted, tdm[0, 1]))

    def test_unrestricted_transition_density_contracts_to_transition_dipole(self):
        method = self.make_method(nstates=2)
        method.ctx = SimpleNamespace(
            mo_coeff=np.zeros((2, 4, 4)),
            occidx_a=np.array([0, 1]),
            viridx_b=np.array([1, 2, 3]),
        )
        ints_aa = np.arange(1, 3 * 4 * 4 + 1, dtype=float).reshape(3, 4, 4) / 13.0
        ints_bb = np.arange(101, 101 + 3 * 4 * 4, dtype=float).reshape(3, 4, 4) / 19.0
        method._dipole_mo_integrals = lambda: (ints_aa, ints_bb, method.ctx)

        gamma = method.transition_density_matrix(0, 1)
        tdm = method._transition_dipole_matrix_u()
        spin_block_integrals = np.zeros((3, 8, 8))
        spin_block_integrals[:, :4, :4] = ints_aa
        spin_block_integrals[:, 4:, 4:] = ints_bb
        contracted = xsf_tda_down.contract("mn,xmn->x", gamma, spin_block_integrals)

        self.assertEqual(gamma.shape, (8, 8))
        self.assertTrue(np.allclose(contracted, tdm[0, 1]))

    def test_nto_returns_singular_values(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.type_u = False
        method.re = False
        method.SA = 0
        method.nc = 1
        method.no = 1
        method.nv = 2
        method.ground_s = 0.5
        method.nstates = 2
        dim = (method.nc + method.no) * (method.no + method.nv)
        method.v = np.arange(1, dim * method.nstates + 1, dtype=float).reshape(dim, method.nstates) / 10.0

        singular_values, holes, particles = method.nto(0, 1, nroots=2)
        gamma = method.transition_density_matrix(0, 1)
        expected = np.linalg.svd(gamma, compute_uv=False)[:2]

        self.assertTrue(np.allclose(singular_values, expected))
        self.assertEqual(holes.shape, (gamma.shape[1], 2))
        self.assertEqual(particles.shape, (gamma.shape[0], 2))

    def test_block_nto_returns_spin_adapted_block_svds(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.type_u = False
        method.re = False
        method.nc = 2
        method.no = 2
        method.nv = 3
        method.nstates = 1
        dim = (method.nc + method.no) * (method.no + method.nv)
        method.v = np.arange(1, dim + 1, dtype=float).reshape(dim, 1) / 10.0

        result = method.block_nto(0, nroots=1)
        cv, co, ov, oo = method._split_analysis_vectors(method.v[:, 0])
        expected_matrices = {
            "CV": cv.T,
            "CO": co.T,
            "OV": ov.T,
            "OO": oo,
        }

        self.assertEqual(set(result), {"CV", "CO", "OV", "OO"})
        nmo = method.nc + method.no + method.nv
        for name, matrix in expected_matrices.items():
            block = result[name]
            expected_s = np.linalg.svd(matrix, compute_uv=False)
            self.assertTrue(np.allclose(block["singular_values"], expected_s[:1]))
            self.assertTrue(np.allclose(block["weights"], expected_s[:1] ** 2))
            self.assertAlmostEqual(block["block_weight"], np.sum(matrix * matrix))
            self.assertEqual(block["holes"].shape, (nmo, 1))
            self.assertEqual(block["particles"].shape, (nmo, 1))

        self.assertEqual(result["CV"]["source"], "C")
        self.assertEqual(result["CV"]["target"], "V")
        self.assertEqual(result["CO"]["source"], "C")
        self.assertEqual(result["CO"]["target"], "O")
        self.assertEqual(result["OV"]["source"], "O")
        self.assertEqual(result["OV"]["target"], "V")
        self.assertEqual(result["OO"]["source"], "O")
        self.assertEqual(result["OO"]["target"], "O")
        self.assertIn("trace_overlap", result["OO"])
        self.assertEqual(result["OO"]["trace_overlap"].shape, (1,))

    def test_davidson_matvec_batches_trial_vectors(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        zs = np.arange(5 * 3, dtype=float).reshape(5, 3)
        batch_sizes = []

        def vind(batch):
            batch_sizes.append(batch.shape[0])
            return batch + 1.0

        out = method._apply_davidson_matvec_batch(vind, zs, batch_size=2)

        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertTrue(np.allclose(out, zs + 1.0))

    def test_delta_a_jk_response_batches_density_matrices(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.delta_a_jk_batch_size = 2
        dm_hf = np.arange(5 * 2 * 2, dtype=float).reshape(5, 2, 2)
        batch_sizes = []

        def vresp_hf(batch):
            batch_sizes.append(batch.shape[0])
            return batch + 10.0, batch + 20.0

        v1_j, v1_k = method._apply_delta_a_jk_response(vresp_hf, dm_hf)

        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertTrue(np.allclose(v1_j, dm_hf + 10.0))
        self.assertTrue(np.allclose(v1_k, dm_hf + 20.0))

    def test_delta_a_jk_response_blocks_are_evaluated_separately(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.delta_a_jk_batch_size = None
        dm_blocks = [np.full((3, 2, 2), idx, dtype=float) for idx in range(4)]
        calls = []

        def vresp_hf(batch):
            calls.append(batch.copy())
            return batch + 10.0, batch + 20.0

        results = method._apply_delta_a_jk_response_blocks(vresp_hf, dm_blocks)

        self.assertEqual([call.shape[0] for call in calls], [3, 3, 3, 3])
        for call, block in zip(calls, dm_blocks):
            self.assertTrue(np.allclose(call, block))
        for (v1_j, v1_k), block in zip(results, dm_blocks):
            self.assertTrue(np.allclose(v1_j, block + 10.0))
            self.assertTrue(np.allclose(v1_k, block + 20.0))

    def test_response_j_batch_size_scales_with_available_gpu_memory(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.mo_coeff = SimpleNamespace(shape=(2, 4096, 4096), dtype=np.float64)
        method.nocc_a = 4096
        method.nvir_b = 2048

        class FakeRuntime:
            @staticmethod
            def memGetInfo():
                return 37 * 1024**3, 40 * 1024**3

        fake_xp = SimpleNamespace(
            dtype=np.dtype,
            cuda=SimpleNamespace(runtime=FakeRuntime),
        )
        old_backend = xsf_tda_down.backend
        old_xp = xsf_tda_down.xp
        try:
            xsf_tda_down.backend = SimpleNamespace(is_gpu=True)
            xsf_tda_down.xp = fake_xp
            batch_size = method._response_j_batch_size(10_000)
        finally:
            xsf_tda_down.backend = old_backend
            xsf_tda_down.xp = old_xp

        self.assertGreaterEqual(batch_size, 64)
        self.assertEqual(batch_size % 32, 0)

    def test_response_j_batch_size_uses_conservative_gpu_fallback(self):
        method = xsf_tda_down.XSF_TDA_down.__new__(xsf_tda_down.XSF_TDA_down)
        method.mo_coeff = SimpleNamespace(shape=(2, 1024, 1024), dtype=np.float64)
        method.nocc_a = 512
        method.nvir_b = 512

        class FakeRuntime:
            @staticmethod
            def memGetInfo():
                raise RuntimeError("CUDA memory unavailable")

        fake_xp = SimpleNamespace(
            dtype=np.dtype,
            cuda=SimpleNamespace(runtime=FakeRuntime),
        )
        old_backend = xsf_tda_down.backend
        old_xp = xsf_tda_down.xp
        try:
            xsf_tda_down.backend = SimpleNamespace(is_gpu=True)
            xsf_tda_down.xp = fake_xp
            batch_size = method._response_j_batch_size(10_000)
        finally:
            xsf_tda_down.backend = old_backend
            xsf_tda_down.xp = old_xp

        self.assertEqual(batch_size, 64)


if __name__ == "__main__":
    unittest.main()
