import importlib.util
from pathlib import Path
import sys

import pytest
from pyscf import gto


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/molecule/opt_geometry.py"
PROJECT_PARENT = EXAMPLE.parents[3]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))
spec = importlib.util.spec_from_file_location("opt_geometry", EXAMPLE)
opt_geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt_geometry)


@pytest.mark.parametrize(
    "kind,reference,response",
    [
        ("xsc", "ROKS", "sc"),
        ("usc", "UKS", "sc"),
        ("xsf_up", "ROKS", "sf_up"),
        ("usf_up", "UKS", "sf_up"),
        ("xsf_down", "ROKS", "sf_down"),
        ("usf_down", "UKS", "sf_down"),
    ],
)
def test_all_six_gradient_routes(monkeypatch, kind, reference, response):
    mol = gto.M(atom="H 0 0 0", spin=1, basis="sto-3g", verbose=0)
    mf = opt_geometry.build_reference(mol, kind)
    assert reference in {cls.__name__.upper() for cls in type(mf).__mro__}

    def fake_response(label):
        class FakeResponse:
            def __init__(self, actual_mf, **kwargs):
                self.mf = actual_mf
                self.kwargs = kwargs
                self.response = label

        return FakeResponse

    monkeypatch.setattr(opt_geometry, "XTDA", fake_response("sc"))
    monkeypatch.setattr(opt_geometry, "SF_TDA_up", fake_response("sf_up"))
    monkeypatch.setattr(opt_geometry, "XSF_TDA_down", fake_response("sf_down"))

    td = opt_geometry.build_response(mf, kind)
    assert td.mf is mf
    assert td.response == response
    if response == "sc":
        assert td.kwargs == {"davidson": True}
    else:
        assert td.kwargs == {
            "method": opt_geometry.sf_method,
            "davidson": True,
            "collinear_samples": opt_geometry.collinear_samples,
        }


def test_rejects_unknown_gradient_route():
    mol = gto.M(atom="H 0 0 0", spin=1, basis="sto-3g", verbose=0)
    with pytest.raises(ValueError, match="method_kind"):
        opt_geometry.build_reference(mol, "bad")


def test_default_route_uses_current_method_kind(monkeypatch):
    mol = gto.M(atom="H 0 0 0", spin=1, basis="sto-3g", verbose=0)
    monkeypatch.setattr(opt_geometry, "method_kind", "usc")

    mf = opt_geometry.build_reference(mol)

    assert "UKS" in {cls.__name__.upper() for cls in type(mf).__mro__}
