from pathlib import Path
import sys


# Load this checkout even when its directory is not named XTDDFT.
import importlib.util

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "XTDDFT", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
package = importlib.util.module_from_spec(spec)
sys.modules["XTDDFT"] = package
spec.loader.exec_module(package)

from XTDDFT.utils.backend import set_backend

set_backend("cpu")

def pytest_sessionstart(session):
    """Run the complete pytest session on the CPU backend."""
