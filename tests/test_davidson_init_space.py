from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import inspect
import unittest

import numpy as np

import sys


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT_dev.XTDDFT import base, sf_tda_up, xsf_tda_down, xtda


class DavidsonInitSpaceTest(unittest.TestCase):
    def test_init_space_is_merged_with_default_guess(self):
        x0 = np.eye(4)[:2]
        init_space = np.eye(4)[:, 1:3]

        merged = base._prepare_davidson_init_space(x0, init_space)

        self.assertEqual(merged.shape, (3, 4))
        self.assertTrue(np.allclose(merged @ merged.T, np.eye(3)))
        projector = merged.T @ merged
        for vector in init_space.T:
            self.assertTrue(np.allclose(projector @ vector, vector))

    def test_all_methods_accept_init_space(self):
        for cls in (xtda.XTDA, xsf_tda_down.XSF_TDA_down, sf_tda_up.SF_TDA_up):
            self.assertIn("init_space", inspect.signature(cls.kernel).parameters)
            self.assertIn("init_space", inspect.signature(cls.davidson_process).parameters)

    def test_dft_setup_errors_are_not_silently_treated_as_hf(self):
        class FakeLibXC:
            @staticmethod
            def test_deriv_order(*args, **kwargs):
                raise RuntimeError("bad functional derivative")

        fake_mf = SimpleNamespace(
            xc="PBE",
            _numint=SimpleNamespace(libxc=FakeLibXC()),
        )
        with mock.patch.object(base, "_build_sf_context", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "bad functional derivative"):
                base.XTDDFT_base(fake_mf, method=0)


if __name__ == "__main__":
    unittest.main()
