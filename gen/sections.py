"""Serial sections of the computational mesh, FIB-SEM style (matplotlib)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

AXN = {0: 'x (TD)', 1: 'y (ND)', 2: 'z (RD)'}
PLANE = {0: 'ND–RD', 1: 'TD–RD', 2: 'TD–ND'}


def cut(X, T, ax, val):
    """Polygons (2D, in the two other axes) of the tets crossing the plane x_ax = val."""
    oth = [a for a in range(3) if a != ax]
    z = X[T, ax] - val                                       # (n,4)
    hit = (z.max(1) > 0) & (z.min(1) < 0)
    T, z = T[hit], z[hit]
    P = X[T][:, :, oth]                                      # (n,4,2)
    E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    polys = []
    pts = np.full((len(T), 4, 2), np.nan); cnt = np.zeros(len(T), int)
    for i, j in E:
        s = (z[:, i] > 0) != (z[:, j] > 0)
        w = z[s, i] / (z[s, i] - z[s, j])
        q = P[s, i] + w[:, None] * (P[s, j] - P[s, i])
        idx = np.where(s)[0]
        pts[idx, cnt[idx]] = q; cnt[idx] += 1
    pts[cnt == 3, 3] = pts[cnt == 3, 2]
    c = np.nanmean(pts, 1, keepdims=True)
    ang = np.arctan2(pts[..., 1] - c[..., 1], pts[..., 0] - c[..., 0])
    pts = np.take_along_axis(pts, np.argsort(ang, 1)[..., None], 1)
    return pts, hit


def section_grid(M, L, ax, vals, out, title, hyd_color='#D7263D', zoom=None, ncol=3, swap=False, bar=20):
    oth = [a for a in range(3) if a != ax]
    if swap: oth = oth[::-1]
    n = len(vals); nrow = (n + ncol - 1) // ncol
    w = L[oth[0]]; h = L[oth[1]]
    if zoom: w, h = zoom[1] - zoom[0], zoom[3] - zoom[2]
    fw = (5.0 if not zoom else 9.0) * min(1.0, 1.6 * w / h); fh = fw * h / w
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * fw, nrow * fh + 0.9), squeeze=False)
    for k, v in enumerate(vals):
        a = axs[k // ncol][k % ncol]
        pm, _ = cut(M['X'], M['mtets'], ax, v)
        ph, _ = cut(M['X'], M['htets'], ax, v)
        if swap: pm = pm[..., ::-1]; ph = ph[..., ::-1]
        lw = 0.6 if zoom else 0.15
        a.add_collection(PolyCollection(pm, facecolors='#BDBDBD', edgecolors='#6a6a6a', linewidths=lw))
        a.add_collection(PolyCollection(ph, facecolors=hyd_color, edgecolors='white' if zoom else '#8c0f1f', linewidths=lw))
        x0, x1 = (zoom[0], zoom[1]) if zoom else (0, L[oth[0]]); y0, y1 = (zoom[2], zoom[3]) if zoom else (0, L[oth[1]])
        bx = x1 - 0.05 * (x1 - x0) - bar; by = y0 + 0.05 * (y1 - y0)
        a.plot([bx, bx + bar], [by, by], color='black', lw=3, solid_capstyle='butt')
        a.text(bx + bar / 2, by + 0.025 * (y1 - y0), f'{bar:g} мкм', ha='center', va='bottom', fontsize=9)
        if zoom: a.set_xlim(zoom[0], zoom[1]); a.set_ylim(zoom[2], zoom[3])
        else: a.set_xlim(0, L[oth[0]]); a.set_ylim(0, L[oth[1]])
        a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([])
        a.set_title(f'{AXN[ax][0]} = {v:g} мкм', fontsize=10)
        pass
    for k in range(n, nrow * ncol): axs[k // ncol][k % ncol].axis('off')
    fig.suptitle(title + f'\nпо горизонтали – {AXN[oth[0]]}, по вертикали – {AXN[oth[1]]}; серое – матрица Zr (сетка КЭ), красное – гидрид',
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
