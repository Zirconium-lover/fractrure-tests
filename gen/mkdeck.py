#!/usr/bin/env python3
"""CalculiX deck for a torn-sheet geometry (geom/fn<P>/mesh.npz).

    ./mkdeck.py ../geom/fn50 -o ../runs/fn50/fn50.inp

Mesh: the matrix C3D4 of the conforming mesh; the hydride C3D4 (one prism
layer) on DUPLICATED nodes; one UC6 facet per hydride surface triangle facing
the matrix, nodes 1-3 on the matrix side, 4-6 the hydride duplicates,
(x2-x1)x(x3-x1) pointing into the hydride (cohesive_uc6.f: positive normal
separation = opening) - the same conventions as test/s3rad/mkseeddeck.py.

Everything after the mesh (materials, sections, step, boundary) is the
committed s3rad deck (hash checked), i.e. the BASE properties of the disc
series, with four scale changes for a 150 x 150 x 75 um box (units stay mm):
  * Kn 1e6 -> 1e8 N/mm^3: the disc decks had Kn ~ 0.75 E/t for a 90 um plate;
    the same ratio for a ~1 um sheet is 1e8 (Turon et al. 2007, K = alpha E/t).
    delta_f = 2 Gc / Tn0 does not depend on Kn, so the fracture energy and the
    softening are unchanged; only delta_0 and the interface compliance move.
  * gmin 2e-6 -> 2e-8: keeps gmin*Kn = 2 N/mm^3 of the disc decks (the stiffness
    a dead facet keeps in tension).
  * grip displacement 1.0 -> 0.0375 mm: the same 25 % nominal strain at theta = 1.
  * output: no per-facet El Print (55k facets x 3 points every 5 increments
    would be gigabytes); U/RF and PEEQ/SDV fields every 10 increments.
"""
import argparse, hashlib, json, os, sys
import numpy as np

SRC = '/home/user/ccx-arch2/test/s3rad/m12_s3rad_gc24_w.inp'
SRC_SHA = '2fb0cf4e3554282e1f85cf641c788939bc7a2fa5918dd842f54d38844dfa5391'
UM = 1e-3                       # um -> mm


def rows(ids, per=16):
    ids = list(ids)
    return '\n'.join(', '.join(str(i) for i in ids[j:j + per]) for j in range(0, len(ids), per))


def tail(kn, gmin, ux, uf=None, gc=None):
    """uf: {'ZR': u_f, 'ZRH': u_f} replaces the damage-evolution displacement (2nd
    constant of the first data line under *Damage Initiation of that material);
    gc: (Gc_seed, Gc_regular) replaces the 4th constant of the two *User Section lines."""
    src = open(SRC, 'rb').read()
    if hashlib.sha256(src).hexdigest() != SRC_SHA:
        sys.exit('source deck hash mismatch')
    t = src.decode('ascii'); t = t[t.index('*Material, Name=ZR'):]
    out, L = [], t.split('\n'); nrep = 0; mat = None; nuf = 0
    for j, l in enumerate(L):
        u = l.strip().upper()
        if u.startswith('*MATERIAL'):
            mat = u.split('NAME=')[1].strip()
        if uf and j > 0 and L[j - 1].strip().upper().startswith('*DAMAGE INITIATION') and mat in uf:
            f = l.split(','); f[1] = ' %.6g' % uf[mat]; l = ','.join(f); nuf += 1
        if j > 0 and L[j - 1].strip().upper().startswith('*USER SECTION'):
            f = l.split(','); f[0] = '%.6e' % kn; f[4] = ' %.6e' % gmin
            if gc:
                f[3] = ' %.6g' % gc[0 if 'SEED' in L[j - 1].upper() else 1]
            l = ','.join(f); nrep += 1
        if u.startswith('FACE_XL_NSET, 1, 1,'):
            l = 'FACE_XL_NSET, 1, 1, %.6f' % ux
        if u.startswith('*OUTPUT'):
            out += ['*Output, Frequency=10', '*Node File', 'U, RF', '*El File', 'PEEQ, SDV',
                    '*Node Print, Nset=FACE_XL_NSET, Totals=Yes', 'RF, U', '*End Step']
            break
        out.append(l)
    if nrep != 2:
        sys.exit('expected two *User Section cards, found %d' % nrep)
    if uf and nuf != len(uf):
        sys.exit('u_f replaced in %d materials, expected %d' % (nuf, len(uf)))
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('geom'); ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--kn', type=float, default=1e8)
    ap.add_argument('--gmin', type=float, default=2e-8)
    ap.add_argument('--strain', type=float, default=0.25, help='nominal strain at theta = 1')
    ap.add_argument('--uf-zr', type=float, help='matrix damage-evolution displacement u_f, mm')
    ap.add_argument('--uf-zrh', type=float, help='hydride damage-evolution displacement u_f, mm')
    ap.add_argument('--gc', type=float, nargs=2, metavar=('SEED', 'REGULAR'), help='interface Gc, N/mm')
    a = ap.parse_args()
    z = np.load(os.path.join(a.geom, 'mesh.npz'))
    st = json.load(open(os.path.join(a.geom, 'stats.json')))
    X = z['X'] * UM; mt = z['mtets']; ht = z['htets'].copy(); surf = z['surf']
    leaf = z['leaf']; kinds = z['kinds']
    Lx, Ly, Lz = [v * UM for v in st['box_um']]
    n0 = len(X)
    # duplicate every hydride node the matrix also uses (all of them: one layer)
    hn = np.unique(ht); shared = np.intersect1d(hn, np.unique(mt))
    dup = -np.ones(n0, int); dup[shared] = n0 + np.arange(len(shared))
    Xall = np.r_[X, X[shared]]
    ht = np.where(dup[ht] >= 0, dup[ht], ht)
    # orient tets (positive volume)
    def orient(T):
        P = Xall[T]; v = np.einsum('ij,ij->i', P[:, 1] - P[:, 0], np.cross(P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]))
        T = T.copy(); T[v < 0] = T[v < 0][:, [0, 2, 1, 3]]; return T
    mt = orient(mt); ht = orient(ht)
    # facets: matrix side = original ids, hydride side = duplicates; normal into the hydride
    face_tet = {}
    for e, T in enumerate(ht):
        for q in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            face_tet[tuple(sorted(T[list(q)]))] = e
    fac = []
    for tri in surf:
        d = dup[tri]
        if (d < 0).any(): sys.exit('facet on a node without a duplicate')
        e = face_tet.get(tuple(sorted(d)))
        if e is None: sys.exit('facet not on a hydride tet')
        opp = [n for n in ht[e] if n not in d][0]
        a_, b_, c_ = Xall[tri]
        if np.cross(b_ - a_, c_ - a_) @ (Xall[opp] - a_) < 0:
            tri = tri[[0, 2, 1]]; d = d[[0, 2, 1]]
        fac.append(np.r_[tri, d])
    fac = np.array(fac)
    # numbering (1-based)
    nm, nh, nf = len(mt), len(ht), len(fac)
    em = np.arange(1, nm + 1); eh = np.arange(nm + 1, nm + nh + 1); ef = np.arange(nm + nh + 1, nm + nh + nf + 1)
    rad = np.array([kinds[l] == 'rad' for l in leaf])
    cen = Xall[fac[:, :3]].mean(1); seed = int(np.argmin(np.linalg.norm(cen - np.array([Lx, Ly, Lz]) / 2, axis=1)))
    x0 = np.where(np.abs(Xall[:n0, 0]) < 1e-12)[0]; xl = np.where(np.abs(Xall[:n0, 0] - Lx) < 1e-12)[0]
    def at(p):
        k = int(np.argmin(np.linalg.norm(Xall[:n0] - np.array(p), axis=1)))
        if np.linalg.norm(Xall[k] - np.array(p)) > 1e-12: sys.exit('no node at %s' % (p,))
        return k + 1
    L = ['** Torn-sheet hydride packets in Zr, C3D4 + UC6 (fractrure-tests/gen/mkdeck.py)',
         '** geometry %s: %s' % (a.geom, json.dumps(st, ensure_ascii=True)),
         '** box %.4f x %.4f x %.4f mm, load along x; Kn %.3e, gmin %.3e, grip %.5f mm at theta=1'
         % (Lx, Ly, Lz, a.kn, a.gmin, a.strain * Lx),
         '*Node']
    L += ['%d, %.12E, %.12E, %.12E' % (i + 1, *x) for i, x in enumerate(Xall)]
    L.append('*User Element, Type=UC6, Nodes=6, Integration Points=3, MaxDof=3')
    L.append('*Element, Type=C3D4')
    L += ['%d, %d, %d, %d, %d' % (e, *(T + 1)) for e, T in zip(em, mt)]
    L += ['%d, %d, %d, %d, %d' % (e, *(T + 1)) for e, T in zip(eh, ht)]
    L.append('*Element, Type=UC6')
    L += ['%d, %d, %d, %d, %d, %d, %d' % (e, *(F + 1)) for e, F in zip(ef, fac)]
    L += ['*Elset, Elset=MATRIX', rows(em), '*Elset, Elset=PLATETANGENTIAL', rows(eh)]
    if rad.any(): L += ['*Elset, Elset=HYDRIDE_RADIAL', rows(eh[rad])]
    if (~rad).any(): L += ['*Elset, Elset=HYDRIDE_TANGENTIAL', rows(eh[~rad])]
    L += ['*Elset, Elset=INTERFACE_SEED', str(ef[seed]),
          '*Elset, Elset=INTERFACE_REGULAR', rows(np.delete(ef, seed)),
          '*Elset, Elset=INTERFACE', rows(ef),
          '*Nset, Nset=FACE_X0_NSET', rows(x0 + 1), '*Nset, Nset=FACE_XL_NSET', rows(xl + 1),
          '*Nset, Nset=FIXPOINTA', str(at((0., 0., 0.))), '*Nset, Nset=FIXPOINTB', str(at((0., Ly, 0.)))]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    uf = {k: v for k, v in (('ZR', a.uf_zr), ('ZRH', a.uf_zrh)) if v is not None}
    L.insert(4, '** scale: u_f ZR %s, u_f ZRH %s mm; interface Gc seed/regular %s N/mm (None = base deck)'
             % (a.uf_zr, a.uf_zrh, a.gc))
    open(a.out, 'w').write('\n'.join(L) + '\n' + tail(a.kn, a.gmin, a.strain * Lx, uf or None, a.gc))
    print('%s: nodes %d (duplicates %d), matrix %d, hydride %d (radial %d), UC6 %d, grip %.5f mm'
          % (a.out, len(Xall), len(shared), nm, nh, int(rad.sum()), nf, a.strain * Lx))


if __name__ == '__main__':
    main()
