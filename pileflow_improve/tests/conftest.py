import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def synthetic_npz(tmp_path_factory):
    from synthetic import make_synthetic_npz
    return make_synthetic_npz(tmp_path_factory.mktemp("data") / "synthetic.npz")
