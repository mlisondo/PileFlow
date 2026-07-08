import os
import sys
import importlib
import numpy as np

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print()

packages = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "sklearn",
    "torch",
    "torchvision",
    "torchcfm",
    "torchdiffeq",
    "fastjet",
    "pythia8",
    "lhapdf",
    "yaml",
    "h5py",
    "tqdm",
    "corner",
    "pileflow_generator",
    "puppi",
    "flows",
    "comparison",
]

failed = []

for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "NO __version__")
        path = getattr(mod, "__file__", "NO __file__")
        print(f"[OK] {name:20s} version={version}")
        print(f"     {path}")
    except Exception as exc:
        failed.append((name, exc))
        print(f"[FAIL] {name:20s} {type(exc).__name__}: {exc}")

print()
try:
    import torch
    x = torch.randn(4, 3)
    y = x @ x.T
    print("[OK] torch tensor test:", y.shape, y.dtype)
except Exception as exc:
    failed.append(("torch runtime", exc))
    print("[FAIL] torch tensor test:", type(exc).__name__, exc)

print()
if failed:
    print("FAILED IMPORTS/RUNTIME TESTS:")
    for name, exc in failed:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1)

print("PileFlow environment check passed.")
