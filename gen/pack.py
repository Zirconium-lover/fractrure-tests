"""Place torn-sheet packets (and their petals) in the box until the hydride
volume fraction is reached; Fn = radial share of the hydride volume."""
import numpy as np
import gmsh
from scipy.spatial import cKDTree
from sheets import make_packet, make_branch
from prism import extrude, prism_ok

GAP = 4.0          # um, clearance between different packets
GAP_PETAL = 1.2    # um, clearance between a petal and its own parent (surface to surface)


def sheet_volume(m):
    qmin, vol = prism_ok(np.r_[m['B'], m['T']], m['tri'], len(m['B']))
    return vol, qmin


def surf(m):
    import pyvista as pv
    n = len(m['B']); X = np.r_[m['B'], m['T']]; tri = m['tri']
    side = [(i, j, j + n) for i, j in m['bedge']] + [(i, j + n, i + n) for i, j in m['bedge']]
    F = np.vstack([tri, tri[:, ::-1] + n, np.array(side)])
    return pv.PolyData(X, np.c_[np.full(len(F), 3), F].ravel())


def collide(m1, m2):
    """True if the closed surfaces of two leaves touch or intersect (vtk)."""
    a, b = surf(m1), surf(m2)
    lo = np.maximum(a.bounds[0::2], b.bounds[0::2]); hi = np.minimum(a.bounds[1::2], b.bounds[1::2])
    if (lo > hi + 2.0).any():
        return False
    _, n = a.collision(b, contact_mode=0, box_tolerance=1e-3, cell_tolerance=0.0)
    if n > 0:
        return True
    # one inside the other without crossing: a node of b inside a
    sel = b.select_enclosed_points(a, check_surface=False)
    return bool(sel['SelectedPoints'].any())


def corner_ok(m, L, clear=3.0):
    """A leaf may cross one lateral face, but must keep away from the box edges
    (a ribbon on a face never touches another face)."""
    on = m['face'] >= 0
    if not on.any():
        return True
    P = np.r_[m['B'], m['T']][np.r_[on, on]]
    f = np.r_[m['face'][on], m['face'][on]]
    for k in range(6):
        ax, side = divmod(k, 2)
        d = np.abs(P[:, ax] - (0.0 if side == 0 else L[ax]))
        if (d[f != k] < clear).any():
            return False
    return True


def inside(m, L, margin_x):
    if not corner_ok(m, L) or m['near'].any():
        return False
    P = np.r_[m['B'], m['T']]
    return (P[:, 0].min() >= margin_x and P[:, 0].max() <= L[0] - margin_x and
            P[:, 1].min() >= -1e-6 and P[:, 1].max() <= L[1] + 1e-6 and
            P[:, 2].min() >= -1e-6 and P[:, 2].max() <= L[2] + 1e-6)


def pack(seed, fn, vf, L=(150., 150., 75.), h=1.5, margin_x=10., petal_p=0.7, verbose=True):
    rng = np.random.default_rng(seed)
    V = L[0] * L[1] * L[2]
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 0)
    sheets, meshes, owner, vols = [], [], [], []
    vrad = vtot = 0.0
    pts = []                                  # surface points of accepted leaves, per packet id
    tries = 0
    while vtot < vf * V and tries < 2000:
        tries += 1
        want_rad = (vrad / vtot < fn) if vtot > 0 else (fn >= 0.5)
        if fn <= 0: want_rad = False
        if fn >= 1: want_rad = True
        kind = 'rad' if want_rad else 'circ'
        sh = make_packet(rng, kind, L, margin_x)
        if sh.poly.area < 300: continue
        if vtot > 0 and vtot + 0.6 * sh.poly.area * 1.1 > 1.15 * vf * V: continue   # do not overshoot
        m = extrude(sh, h, L)
        vol, vmin = sheet_volume(m)
        if vmin <= 0 or not inside(m, L, margin_x): continue
        P = np.r_[m['B'], m['T']]
        if pts and min(cKDTree(q).query(P)[0].min() for q in pts) < GAP: continue
        pid = len(pts)
        leaves = [(sh, m, vol)]
        allp = [P]
        if rng.uniform() < petal_p:
            for _ in range(rng.integers(1, 4)):
                for _try in range(15):
                    br = make_branch(rng, sh)
                    if br is None: break
                    mb = extrude(br, h, L)
                    vb, vminb = sheet_volume(mb)
                    if vminb <= 0 or not inside(mb, L, margin_x): continue
                    Pb = np.r_[mb['B'], mb['T']]
                    if min(cKDTree(q).query(Pb)[0].min() for q in allp) < GAP_PETAL: continue
                    if any(collide(mb, ml) for _, ml, _ in leaves): continue
                    if pts and min(cKDTree(q).query(Pb)[0].min() for q in pts) < GAP: continue
                    leaves.append((br, mb, vb)); allp.append(Pb); break
        for s_, m_, v_ in leaves:
            sheets.append(s_); meshes.append(m_); owner.append(pid); vols.append(v_)
            vtot += v_; vrad += v_ if kind == 'rad' else 0.0
        pts.append(np.vstack(allp))
        if verbose:
            print(f'packet {pid}: {kind:4s} area {sh.poly.area:7.0f} um2  leaves {len(leaves)}  '
                  f'vol {sum(l[2] for l in leaves):7.0f} um3  -> vf {100*vtot/V:.2f} %  Fn {100*vrad/max(vtot,1e-9):.0f} %')
    gmsh.finalize()
    return dict(sheets=sheets, meshes=meshes, owner=np.array(owner), vols=np.array(vols),
                vf=vtot / V, fn=vrad / max(vtot, 1e-9), L=L, h=h, margin_x=margin_x)
