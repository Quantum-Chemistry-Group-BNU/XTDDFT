from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np
from pyscf import df, gto

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.utils.df_cderi_cache import (
    DFCderiCacheConfig,
    ao2mo_from_molecular_cderi_cache,
    attach_pbc_gamma_cderi_cache,
    build_molecular_cderi_cache,
    build_pbc_gamma_cderi_cache,
    iter_molecular_cderi_blocks,
    prepare_pbc_gamma_df_cderi_cache,
)
from XTDDFT_dev.XTDDFT.base import _set_df_cache_config_on_mf
from XTDDFT_dev.utils.pbc_cderi_outcore import write_pbc_gamma_cderi_cache_from_blocks


def make_water():
    return gto.M(
        atom="""
O 0.000000 0.000000 0.000000
H 0.000000 0.757000 0.587000
H 0.000000 -0.757000 0.587000
""",
        basis="sto-3g",
        verbose=0,
    )


class MolecularCderiCacheTest(unittest.TestCase):
    def test_builds_hdf5_cache_and_reads_aux_blocks(self):
        mol = make_water()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "j3c.h5"

            info = build_molecular_cderi_cache(mol, cache_path, auxbasis="weigend", max_memory=64)

            self.assertEqual(Path(info.path), cache_path)
            self.assertEqual(info.dataset, "j3c")
            with h5py.File(cache_path, "r") as handle:
                self.assertIn("j3c", handle)
                self.assertEqual(handle["j3c"].shape[1], mol.nao_nr() * (mol.nao_nr() + 1) // 2)

            blocks = list(iter_molecular_cderi_blocks(cache_path, block_size=2))
            self.assertGreater(len(blocks), 1)
            self.assertEqual(sum(block.shape[0] for block in blocks), info.naux)

    def test_cached_ao2mo_matches_pyscf_df_ao2mo(self):
        mol = make_water()
        mf = mol.RHF().density_fit(auxbasis="weigend").run()
        occ = mf.mo_coeff[:, mf.mo_occ > 0]
        vir = mf.mo_coeff[:, mf.mo_occ == 0]
        mo_coeffs = (occ, occ, vir, vir)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "j3c.h5"
            build_molecular_cderi_cache(mol, cache_path, auxbasis="weigend", max_memory=64)

            cached = ao2mo_from_molecular_cderi_cache(cache_path, mo_coeffs, block_size=2)
            reference = df.DF(mol, auxbasis="weigend").ao2mo(mo_coeffs, compact=False)

        self.assertEqual(cached.shape, reference.shape)
        self.assertLess(np.max(np.abs(cached - reference)), 1e-9)


class FakeCell:
    dimension = 3


class FakePbcGammaGDF:
    def __init__(self):
        self.cell = FakeCell()
        self.nao = 4
        self.is_gamma_point = True
        self.auxbasis = {"C": "cc-pvdz-jkfit"}
        self._omega = -0.11
        self._cderi = {0: np.arange(30, dtype=float).reshape(5, 6)}
        self._cderip = {}
        self._cderi_idx = (
            np.array([0, 1, 4, 5, 8, 10], dtype=np.int64),
            np.array([0, 3], dtype=np.int64),
        )
        self.build_called = False

    def build(self):
        self.build_called = True
        return self


class PbcGammaCderiCacheTest(unittest.TestCase):
    def test_builds_pbc_gamma_cache_from_gdf(self):
        gdf = FakePbcGammaGDF()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "pbc_j3c.h5"

            info = build_pbc_gamma_cderi_cache(gdf, cache_path)

            self.assertEqual(Path(info.path), cache_path)
            self.assertEqual(info.dataset, "pbc_gamma/cderi_0")
            self.assertEqual(info.naux, 5)
            self.assertEqual(info.nao_pair, 6)
            with h5py.File(cache_path, "r") as handle:
                self.assertEqual(handle["pbc_gamma/cderi_0"].shape, (5, 6))
                self.assertEqual(handle["pbc_gamma/cderi_idx_pair"].shape, (6,))
                self.assertEqual(handle["pbc_gamma/cderi_idx_diag"].shape, (2,))
                self.assertEqual(handle["pbc_gamma"].attrs["nao"], 4)
                self.assertEqual(handle["pbc_gamma"].attrs["omega"], -0.11)

    def test_attach_pbc_gamma_cache_uses_hdf5_dataset_slices(self):
        source = FakePbcGammaGDF()
        target = FakePbcGammaGDF()
        target._cderi = None
        target._cderi_idx = None

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "pbc_j3c.h5"
            build_pbc_gamma_cderi_cache(source, cache_path)

            handle = attach_pbc_gamma_cderi_cache(target, cache_path)
            try:
                self.assertTrue(target.is_gamma_point)
                self.assertEqual(target.nao, 4)
                self.assertEqual(target._omega, -0.11)
                self.assertEqual(target._cderi[0].shape, source._cderi[0].shape)
                self.assertTrue(np.allclose(target._cderi[0][1:3], source._cderi[0][1:3]))
                self.assertTrue(np.array_equal(target._cderi_idx[0], source._cderi_idx[0]))
                self.assertTrue(np.array_equal(target._cderi_idx[1], source._cderi_idx[1]))
            finally:
                handle.close()

    def test_prepare_pbc_gamma_cache_reuses_one_file_per_omega(self):
        gdf = FakePbcGammaGDF()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DFCderiCacheConfig(mode="auto", cache_dir=tmpdir, tag="solid_c")

            handle1 = prepare_pbc_gamma_df_cderi_cache(gdf, config, omega=-0.11)
            handle2 = prepare_pbc_gamma_df_cderi_cache(gdf, config, omega=-0.11)
            try:
                self.assertIs(handle1, handle2)
                self.assertEqual(gdf._cderi[0].shape, (5, 6))
                self.assertIn("solid_c_omega_minus_0p11.h5", str(handle1.h5file.filename))
                self.assertEqual(len(list(Path(tmpdir).glob("*.h5"))), 1)
            finally:
                handle1.close()

    def test_outcore_writer_creates_attachable_pbc_gamma_cache_from_blocks(self):
        gdf = FakePbcGammaGDF()
        blocks = [
            (0, 2, np.arange(10, dtype=float).reshape(5, 2)),
            (2, 6, np.arange(20, 40, dtype=float).reshape(5, 4)),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "pbc_j3c_outcore.h5"

            info = write_pbc_gamma_cderi_cache_from_blocks(
                gdf,
                cache_path,
                cderi_blocks=blocks,
                cderi_shape=(5, 6),
                cderi_idx=gdf._cderi_idx,
                omega=-0.11,
            )

            self.assertEqual(info.naux, 5)
            self.assertEqual(info.nao_pair, 6)
            attached = attach_pbc_gamma_cderi_cache(gdf, cache_path)
            try:
                expected = np.concatenate([block for _, _, block in blocks], axis=1)
                self.assertTrue(np.allclose(gdf._cderi[0][:], expected))
            finally:
                attached.close()

    def test_prepare_outcore_backend_does_not_call_gdf_build_first(self):
        import XTDDFT_dev.utils.pbc_cderi_outcore as outcore

        gdf = FakePbcGammaGDF()
        original = outcore.build_pbc_gamma_cderi_cache_outcore

        def fake_outcore_builder(gdf_arg, path, group="pbc_gamma", overwrite=True, **kwargs):
            self.assertIs(gdf_arg, gdf)
            self.assertFalse(gdf.build_called)
            return write_pbc_gamma_cderi_cache_from_blocks(
                gdf_arg,
                path,
                group=group,
                overwrite=overwrite,
                cderi_blocks=[(0, 6, gdf_arg._cderi[0])],
                cderi_shape=gdf_arg._cderi[0].shape,
                cderi_idx=gdf_arg._cderi_idx,
                omega=-0.11,
            )

        outcore.build_pbc_gamma_cderi_cache_outcore = fake_outcore_builder
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config = DFCderiCacheConfig(
                    mode="auto",
                    cache_dir=tmpdir,
                    tag="solid_c",
                    build_backend="outcore",
                )

                handle = prepare_pbc_gamma_df_cderi_cache(gdf, config, omega=-0.11)
                try:
                    self.assertFalse(gdf.build_called)
                    self.assertEqual(gdf._cderi[0].shape, (5, 6))
                finally:
                    handle.close()
        finally:
            outcore.build_pbc_gamma_cderi_cache_outcore = original

    def test_base_preserves_existing_df_cache_config_when_none_is_passed(self):
        mf = SimpleNamespace()
        config = DFCderiCacheConfig(mode="auto", cache_dir="/tmp/xtddft-cache", tag="keep")
        mf._xtddft_df_cderi_cache_config = config

        out = _set_df_cache_config_on_mf(mf, None)

        self.assertIs(out, config)
        self.assertIs(mf._xtddft_df_cderi_cache_config, config)


if __name__ == "__main__":
    unittest.main()
