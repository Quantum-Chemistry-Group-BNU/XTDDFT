from pathlib import Path
import inspect
import tempfile
import unittest

import h5py
import numpy as np

from XTDDFT_dev.utils.streaming_pbc_df_jk import (
    _apply_ewald_exxdiv_for_gamma,
    _assert_cache_omega,
    _gamma_ewald_terms,
    _gamma_madelung,
    _is_gamma_kpts,
    _prepare_cache_for_omega,
    _should_apply_ewald_exxdiv,
    streaming_jk_from_cderi_block,
    streaming_jk_from_h5,
)


def _write_fake_cache(path: Path, cderi, pair_idx, nao):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("pbc_gamma")
        group.create_dataset("cderi_0", data=cderi)
        group.create_dataset("cderi_idx_pair", data=np.asarray(pair_idx, dtype=np.int64))
        group.create_dataset("cderi_idx_diag", data=np.array([], dtype=np.int64))
        group.attrs["nao"] = nao
        group.attrs["omega"] = 0.0


def _expand_full(cderi, pair_idx, nao):
    rows, cols = np.divmod(pair_idx, nao)
    full = np.zeros((cderi.shape[0], nao, nao))
    full[:, rows, cols] = cderi
    full[:, cols, rows] = cderi
    return full


class StreamingPbcDfJkTest(unittest.TestCase):
    def test_streaming_jk_matches_full_contraction_for_stacked_dm(self):
        nao = 4
        pair_idx = np.array([0, 1, 4, 5, 8, 10], dtype=np.int64)
        cderi = np.arange(30, dtype=float).reshape(5, 6) / 10.0
        dm = np.arange(32, dtype=float).reshape(2, nao, nao) / 17.0
        dm = dm + dm.transpose(0, 2, 1)

        full = _expand_full(cderi, pair_idx, nao)
        expected_vj = np.einsum("Lpq,xpq,Lrs->xrs", full, dm, full)
        expected_vk = np.einsum("Lpq,xqr,Lsr->xps", full, dm, full)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            _write_fake_cache(cache_path, cderi, pair_idx, nao)

            vj, vk = streaming_jk_from_h5(cache_path, dm, aux_block_size=2, use_gpu=False)

        self.assertTrue(np.allclose(vj, expected_vj))
        self.assertTrue(np.allclose(vk, expected_vk))

    def test_block_k_contraction_matches_full_reference(self):
        cderi = np.arange(45, dtype=float).reshape(5, 3, 3) / 13.0
        dm = np.arange(18, dtype=float).reshape(2, 3, 3) / 19.0
        dm = dm + dm.transpose(0, 2, 1)
        expected_vk = np.einsum("Lpq,xqr,Lsr->xps", cderi, dm, cderi)

        _, vk = streaming_jk_from_cderi_block(
            cderi, dm, with_j=False, with_k=True, xp=np
        )

        self.assertTrue(np.allclose(vk, expected_vk))

    def test_block_k_contraction_is_batched_not_per_aux_python_loop(self):
        source = inspect.getsource(streaming_jk_from_cderi_block)

        self.assertNotIn("for lidx in range", source)
        self.assertIn("einsum", source)

    def test_ewald_exxdiv_adds_madelung_sdm_s_to_k(self):
        dm = np.arange(18, dtype=float).reshape(2, 3, 3) / 11.0
        dm = dm + dm.transpose(0, 2, 1)
        vk = np.ones_like(dm)
        overlap = np.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 0.9, 0.2],
                [0.0, 0.2, 1.1],
            ]
        )
        madelung = 0.37

        corrected = _apply_ewald_exxdiv_for_gamma(vk, dm, overlap, madelung, np)

        expected = vk + madelung * np.stack([
            overlap @ dm_i @ overlap for dm_i in dm
        ])
        self.assertTrue(np.allclose(corrected, expected))

    def test_ewald_exxdiv_applies_to_short_range_k_too(self):
        self.assertTrue(_should_apply_ewald_exxdiv(with_k=True, exxdiv="ewald"))
        self.assertFalse(_should_apply_ewald_exxdiv(with_k=False, exxdiv="ewald"))
        self.assertFalse(_should_apply_ewald_exxdiv(with_k=True, exxdiv=None))

    def test_gamma_madelung_passes_explicit_gamma_kpts(self):
        class FakePbc:
            def __init__(self):
                self.kpts = None

            def madelung(self, cell, kpts):
                self.kpts = np.asarray(kpts)
                return 0.25

        fake_pbc = FakePbc()
        tools_module = type("Tools", (), {"pbc": fake_pbc})

        value = _gamma_madelung(object(), tools_module=tools_module)

        self.assertEqual(value, 0.25)
        self.assertEqual(fake_pbc.kpts.shape, (1, 3))
        self.assertTrue(np.allclose(fake_pbc.kpts, 0.0))

    def test_gamma_ewald_terms_use_range_coulomb_for_short_range_omega(self):
        class FakeCell:
            def __init__(self):
                self.omega = 0.0
                self.overlap_omega = None

            def pbc_intor(self, name, hermi=1, kpts=None):
                self.overlap_omega = self.omega
                return np.eye(2)

        class FakeRangeContext:
            def __init__(self, df, omega):
                self.df = df
                self.omega = omega
                self.old_omega = None

            def __enter__(self):
                self.old_omega = self.df.cell.omega
                self.df.cell.omega = self.omega
                return self.df

            def __exit__(self, exc_type, exc, tb):
                self.df.cell.omega = self.old_omega
                return False

        class FakeDf:
            def __init__(self):
                self.cell = FakeCell()
                self.range_omegas = []

            def range_coulomb(self, omega):
                self.range_omegas.append(omega)
                return FakeRangeContext(self, omega)

        class FakePbc:
            def __init__(self):
                self.madelung_omega = None

            def madelung(self, cell, kpts):
                self.madelung_omega = cell.omega
                return cell.omega

        gdf = FakeDf()
        fake_pbc = FakePbc()
        tools_module = type("Tools", (), {"pbc": fake_pbc})

        overlap, madelung = _gamma_ewald_terms(
            gdf, -0.11, tools_module=tools_module
        )

        self.assertTrue(np.allclose(overlap, np.eye(2)))
        self.assertEqual(madelung, -0.11)
        self.assertEqual(gdf.range_omegas, [-0.11])
        self.assertEqual(gdf.cell.overlap_omega, -0.11)
        self.assertEqual(fake_pbc.madelung_omega, -0.11)
        self.assertEqual(gdf.cell.omega, 0.0)

    def test_gamma_ewald_terms_prefer_gpu4pyscf_overlap_wrapper(self):
        class FakeCell:
            omega = 0.0

            def pbc_intor(self, name, hermi=1, kpts=None):
                raise AssertionError("GPU4PySCF overlap wrapper should be used")

        class FakeDf:
            cell = FakeCell()

        class FakeInt1e:
            def __init__(self):
                self.kpts = None

            def int1e_ovlp(self, cell, kpts=None):
                self.kpts = np.asarray(kpts)
                return 2.0 * np.eye(2)

        class FakePbc:
            def madelung(self, cell, kpts):
                return 0.0

        fake_int1e = FakeInt1e()
        tools_module = type("Tools", (), {"pbc": FakePbc()})
        kpts = np.zeros((1, 3))

        overlap, _ = _gamma_ewald_terms(
            FakeDf(), 0.0, kpts=kpts, tools_module=tools_module,
            int1e_module=fake_int1e,
        )

        self.assertTrue(np.allclose(overlap, 2.0 * np.eye(2)))
        self.assertEqual(fake_int1e.kpts.shape, (1, 3))

    def test_gamma_ewald_terms_squeeze_single_gamma_overlap_stack(self):
        class FakeCell:
            omega = 0.0

        class FakeDf:
            cell = FakeCell()

        class FakeInt1e:
            def int1e_ovlp(self, cell, kpts=None):
                return np.asarray([3.0 * np.eye(2)])

        class FakePbc:
            def madelung(self, cell, kpts):
                return 0.0

        tools_module = type("Tools", (), {"pbc": FakePbc()})

        overlap, _ = _gamma_ewald_terms(
            FakeDf(), 0.0, kpts=np.zeros((1, 3)),
            tools_module=tools_module, int1e_module=FakeInt1e(),
        )

        self.assertEqual(overlap.shape, (2, 2))
        self.assertTrue(np.allclose(overlap, 3.0 * np.eye(2)))

    def test_gamma_ewald_terms_do_not_enter_range_context_for_zero_omega(self):
        class FakeCell:
            omega = 0.0

            def pbc_intor(self, name, hermi=1, kpts=None):
                return np.eye(1)

        class FakeDf:
            def __init__(self):
                self.cell = FakeCell()

            def range_coulomb(self, omega):
                raise AssertionError("zero omega should not use range_coulomb")

        class FakePbc:
            def madelung(self, cell, kpts):
                return cell.omega

        tools_module = type("Tools", (), {"pbc": FakePbc()})

        overlap, madelung = _gamma_ewald_terms(
            FakeDf(), 0.0, tools_module=tools_module
        )

        self.assertTrue(np.allclose(overlap, np.eye(1)))
        self.assertEqual(madelung, 0.0)

    def test_gamma_kpt_arrays_are_streamable(self):
        self.assertTrue(_is_gamma_kpts(None))
        self.assertTrue(_is_gamma_kpts(np.zeros(3)))
        self.assertTrue(_is_gamma_kpts(np.zeros((1, 3))))
        self.assertFalse(_is_gamma_kpts(np.array([0.1, 0.0, 0.0])))
        self.assertFalse(_is_gamma_kpts(np.zeros((2, 3))))

    def test_prepare_cache_for_range_coulomb_sets_rsh_omega_without_mutating_cell(self):
        class FakeRshDf:
            def __init__(self):
                self.cell = type("Cell", (), {"omega": 0.0})()
                self.is_gamma_point = False

        class FakeRangeContext:
            def __init__(self, rsh_df):
                self.rsh_df = rsh_df

            def __enter__(self):
                return self.rsh_df

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeDf:
            def __init__(self):
                self.rsh_df = FakeRshDf()

            def range_coulomb(self, omega):
                self.requested_omega = omega
                return FakeRangeContext(self.rsh_df)

        calls = []

        def fake_prepare(gdf, config, omega=None):
            calls.append((gdf, config, omega))
            return "handle"

        gdf = FakeDf()
        out = _prepare_cache_for_omega(gdf, object(), fake_prepare, -0.11)

        self.assertEqual(out, "handle")
        self.assertEqual(gdf.requested_omega, -0.11)
        self.assertTrue(gdf.rsh_df.is_gamma_point)
        self.assertEqual(gdf.rsh_df._omega, -0.11)
        self.assertEqual(gdf.rsh_df.cell.omega, 0.0)
        self.assertEqual(calls[0][2], -0.11)

    def test_cache_omega_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"
            _write_fake_cache(cache_path, np.zeros((1, 1)), [0], 1)

            with self.assertRaisesRegex(RuntimeError, "omega mismatch"):
                _assert_cache_omega(cache_path, -0.11)


if __name__ == "__main__":
    unittest.main()
