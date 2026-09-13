from pathlib import Path
import sys


PROJECT_PARENT = Path(__file__).resolve().parent.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from XTDDFT.utils.backend import set_backend

set_backend("cpu")

def pytest_sessionstart(session):
    """Run the complete pytest session on the CPU backend."""
