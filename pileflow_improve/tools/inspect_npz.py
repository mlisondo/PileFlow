import argparse
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('path')
a = p.parse_args()
with np.load(a.path, allow_pickle=False) as f:
    print('Keys:')
    for k in f.files:
        x = f[k]
        print(f'  {k:30s} shape={x.shape} dtype={x.dtype}')
