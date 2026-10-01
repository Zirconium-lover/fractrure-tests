#!/usr/bin/env python3
"""Torn-sheet hydride geometry + conforming mesh + pictures for review.

    ./make_geometry.py --seed 1 --fn 50 --out ../geom/fn50
writes <out>/mesh.npz, stats.json, view3d.png, sec_TD-ND.png, sec_ND-RD.png,
sec_TD-RD.png, zoom.png (no input deck yet - that comes after the geometry is approved)."""
import argparse, json, os, pickle, subprocess, sys
import numpy as np
from pack import pack
from volume import assemble, mesh_matrix
from prism import tet_vol
from sections import section_grid

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--fn', type=float, default=50)
ap.add_argument('--vf', type=float, default=1.1, help='hydride volume fraction, percent')
ap.add_argument('--h', type=float, default=1.5, help='in-plane element size of the sheets, um')
ap.add_argument('--hmax', type=float, default=9.0)
ap.add_argument('--grad', type=float, default=0.35)
ap.add_argument('--out', required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
L = (150., 150., 75.)
G = pack(a.seed, a.fn / 100, a.vf / 100, L=L, h=a.h, verbose=False)
A = assemble(G)
M = mesh_matrix(G, A, hmin=a.h, hmax=a.hmax, grad=a.grad, verbose=False)
X = M['X']
vm = np.abs(tet_vol(X, M['mtets'])).sum(); vh = np.abs(tet_vol(X, M['htets'])).sum()
# hydride tet quality (gamma = 12 (3V)^(2/3) / sum l^2, 1 for a regular tet)
def gamma(T):
    P = X[T]; V = np.abs(tet_vol(X, T))
    l2 = sum(((P[:, i] - P[:, j]) ** 2).sum(1) for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
    return 12 * (3 * V) ** (2 / 3) / l2
gh = gamma(M['htets']); gm = gamma(M['mtets'])
th = np.concatenate([m['t'] for m in G['meshes']])
kinds = [s.kind for s in G['sheets']]; br = [s.branch for s in G['sheets']]
st = dict(seed=a.seed, fn_target=a.fn, fn_actual=round(100 * G['fn'], 1), vf_actual=round(100 * vh / (vm + vh), 3),
          box_um=L, packets=int(G['owner'].max() + 1), leaves=len(kinds), petals=int(sum(br)),
          packets_radial=len(set(o for o, k in zip(G['owner'], kinds) if k == 'rad')),
          thickness_um=dict(min=round(float(th.min()), 2), median=round(float(np.median(th)), 2), max=round(float(th.max()), 2)),
          nodes=int(len(X)), tets_matrix=int(len(M['mtets'])), tets_hydride=int(len(M['htets'])),
          uc6_facets=int(len(M['surf'])), gamma_min_matrix=round(float(gm.min()), 4), gamma_min_hydride=round(float(gh.min()), 4),
          tets_gamma_below_0p1=int((gm < 0.1).sum() + (gh < 0.1).sum()))
json.dump(st, open(f'{a.out}/stats.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(st, ensure_ascii=False))
np.savez_compressed(f'{a.out}/mesh.npz', X=X, mtets=M['mtets'], htets=M['htets'], surf=M['surf'],
                    leaf=A['leaf'], surf_leaf=A['surf_leaf'], kinds=np.array(kinds), branch=np.array(br), owner=G['owner'])
pickle.dump(G, open(f'{a.out}/packets.pkl', 'wb'))
lab = f'Fn = {a.fn:g} % (факт. {100*G["fn"]:.0f} %), доля гидрида {st["vf_actual"]:.2f} %'
section_grid(M, L, 2, [8, 20, 32, 44, 56, 68], f'{a.out}/sec_TD-ND.png', f'{lab}: сечения TD–ND (z = const)')
section_grid(M, L, 0, [25, 50, 65, 85, 100, 125], f'{a.out}/sec_ND-RD.png', f'{lab}: сечения ND–RD (x = const)', swap=True, ncol=6)
section_grid(M, L, 1, [20, 45, 65, 85, 105, 130], f'{a.out}/sec_TD-RD.png', f'{lab}: сечения TD–RD (y = const)')
# zoom: one element through the thickness
c = X[M['htets'][np.argmin(np.abs(X[M['htets']].mean(1)[:, 2] - 37.5))]].mean(0)
section_grid(M, L, 2, [round(float(c[2]), 2)], f'{a.out}/zoom.png', f'{lab}: крупно, z = {c[2]:.1f} мкм (гидрид – один слой элементов)',
             zoom=(c[0] - 7, c[0] + 7, c[1] - 4.5, c[1] + 4.5), ncol=1, bar=2)
subprocess.run(['xvfb-run', '-a', '-s', '-screen 0 1920x1080x24', sys.executable, 'view3d.py', f'{a.out}/packets.pkl',
                f'{a.out}/view3d.png', f'Fn = {100*G["fn"]:.0f} %,  hydride {st["vf_actual"]:.2f} vol.%,  {st["packets"]} packets'], check=True)
