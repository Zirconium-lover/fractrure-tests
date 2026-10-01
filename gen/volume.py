"""Conforming matrix mesh around the prism leaves (gmsh, discrete boundary)."""
import numpy as np
import gmsh
from scipy.spatial import cKDTree
from prism import prism_tets, side_tris


def assemble(G):
    """Global hydride nodes/tets and the hydride surface facing the matrix."""
    X, tets, surf, ribbons, leaf_of_tet, surf_leaf = [], [], [], [], [], []
    nb = 0
    for k, m in enumerate(G['meshes']):
        n = len(m['B'])
        X.append(np.r_[m['B'], m['T']])
        T = prism_tets(m['tri'], nb, n); tets.append(T); leaf_of_tet.append(np.full(len(T), k))
        tri = m['tri'] + nb
        faceflag = np.r_[m['face'], m['face']]
        st = side_tris(m['bedge'], nb, n)
        e_face = [(m['face'][i] if m['face'][i] == m['face'][j] else -1) for i, j in m['bedge']]
        e_face = np.repeat(np.array(e_face, int), 2)
        sk = np.vstack([tri, tri[:, ::-1] + n, st[e_face < 0]])
        surf.append(sk); surf_leaf.append(np.full(len(sk), k))
        for f in range(6):                       # ribbons: boundary edges lying on face f
            E = [tuple(sorted((i, j))) for i, j in m['bedge'] if m['face'][i] == f and m['face'][j] == f]
            if E:
                ribbons.append((f, nb, n, E))
        nb += 2 * n
    return dict(X=np.vstack(X), tets=np.vstack(tets), surf=np.vstack(surf), ribbons=ribbons,
                leaf=np.concatenate(leaf_of_tet), surf_leaf=np.concatenate(surf_leaf),
                on_face=np.array([any(f == k for f, _, _, _ in []) for k in range(len(G['meshes']))]))


def chains(E):
    """Order a set of undirected edges into open polylines."""
    from collections import defaultdict
    adj = defaultdict(list)
    for i, j in E:
        adj[i].append(j); adj[j].append(i)
    seen, out = set(), []
    ends = [v for v in adj if len(adj[v]) == 1]
    for s in ends + list(adj):
        if s in seen: continue
        line = [s]; seen.add(s); prev = None; cur = s
        while True:
            nxt = [w for w in adj[cur] if w != prev and w not in seen]
            if not nxt: break
            prev, cur = cur, nxt[0]; line.append(cur); seen.add(cur)
        out.append(line)
    return out


def leaf_of_base(G, nb):
    acc = 0
    for k, m in enumerate(G['meshes']):
        if acc == nb: return k
        acc += 2 * len(m['B'])
    raise ValueError(nb)


def size_fn(tree, hmin, hmax, grad):
    def f(dim, tag, x, y, z, lc):
        d = tree.query((x, y, z))[0]
        return float(min(hmax, hmin + grad * d))
    return f


def mesh_matrix(G, A, hmin=1.5, hmax=9.0, grad=0.35, verbose=True):
    L = G['L']; X = A['X']
    tree = cKDTree(np.r_[X, A['X'][A['surf']].mean(1)])
    gmsh.initialize(); gmsh.option.setNumber('General.Terminal', 1 if verbose else 0)
    gmsh.model.add('box')
    geo = gmsh.model.geo
    # --- the six box faces, conforming with the ribbons -------------------------------
    C = [geo.addPoint(x, y, z) for z in (0, L[2]) for y in (0, L[1]) for x in (0, L[0])]
    def cid(ix, iy, iz): return C[ix + 2 * iy + 4 * iz]
    edges = {}
    def line(a, b):
        k = (min(a, b), max(a, b))
        if k not in edges: edges[k] = geo.addLine(k[0], k[1])
        return edges[k] if k == (a, b) else -edges[k]
    faces = {0: [cid(0,0,0), cid(0,1,0), cid(0,1,1), cid(0,0,1)], 1: [cid(1,0,0), cid(1,1,0), cid(1,1,1), cid(1,0,1)],
             2: [cid(0,0,0), cid(1,0,0), cid(1,0,1), cid(0,0,1)], 3: [cid(0,1,0), cid(1,1,0), cid(1,1,1), cid(0,1,1)],
             4: [cid(0,0,0), cid(1,0,0), cid(1,1,0), cid(0,1,0)], 5: [cid(0,0,1), cid(1,0,1), cid(1,1,1), cid(0,1,1)]}
    ptag_to_global = {}
    rib_holes = {f: [] for f in range(6)}
    for f, nb, n, E in A['ribbons']:
        for ch in chains(E):
            b = [nb + i for i in ch]; t = [nb + n + i for i in ch]
            loop_nodes = b + t[::-1]
            pt = []
            for g in loop_nodes:
                p = geo.addPoint(*X[g]); ptag_to_global[p] = g; pt.append(p)
            ls = [geo.addLine(pt[i], pt[(i + 1) % len(pt)]) for i in range(len(pt))]
            for l_ in ls: geo.mesh.setTransfiniteCurve(l_, 2)
            rib_holes[f].append(geo.addCurveLoop(ls))
    surfs = []
    for f in range(6):
        q = faces[f]
        loop = geo.addCurveLoop([line(q[i], q[(i + 1) % 4]) for i in range(4)])
        surfs.append(geo.addPlaneSurface([loop] + rib_holes[f]))
    geo.synchronize()
    gmsh.model.mesh.setSizeCallback(size_fn(tree, hmin, hmax, grad))
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.model.mesh.generate(2)
    # collect box-face triangles with global ids
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    # gmsh node -> global id: ribbon points map onto hydride nodes, others are new
    gtag = {}
    ntag_of_point = {}
    for p, g in ptag_to_global.items():
        tags, _, _ = gmsh.model.mesh.getNodes(0, p)
        gtag[int(tags[0])] = g
    newX = []
    nextid = len(X)
    for t_, x_ in zip(nt, xyz):
        if int(t_) not in gtag:
            gtag[int(t_)] = nextid; newX.append(x_); nextid += 1
    face_tri = []
    for s in surfs:
        _, en = gmsh.model.mesh.getElementsByType(2, s)
        face_tri.append(np.array([gtag[int(t_)] for t_ in en]).reshape(-1, 3))
    face_tri = np.vstack(face_tri)
    Xall = np.r_[X, np.array(newX)]
    gmsh.model.remove()
    # --- discrete closed boundary -> volume -----------------------------------------
    gmsh.model.add('vol')
    # leaves that reach a box face are part of the OUTER shell; the others are holes
    face_leaf = set()
    for f, nb, n, E in A['ribbons']:
        face_leaf.add(int(A['leaf'][np.searchsorted(np.cumsum([0] + [2 * len(m['B']) for m in G['meshes']]), nb, side='right') - 1]) if False else leaf_of_base(G, nb))
    outer = np.r_[face_tri, A['surf'][np.isin(A['surf_leaf'], list(face_leaf))]]
    shells = [outer] + [A['surf'][A['surf_leaf'] == k] for k in range(len(G['meshes'])) if k not in face_leaf]
    allbt = np.vstack(shells); used = np.unique(allbt)
    gmsh.model.addDiscreteEntity(2, 1)
    gmsh.model.mesh.addNodes(2, 1, (used + 1).tolist(), Xall[used].ravel().tolist())
    loops = []
    for i, sh in enumerate(shells):
        tag = i + 1
        if i > 0: gmsh.model.addDiscreteEntity(2, tag)
        gmsh.model.mesh.addElementsByType(tag, 2, [], (sh + 1).ravel().tolist())
        loops.append(gmsh.model.geo.addSurfaceLoop([tag]))
    gmsh.model.geo.addVolume(loops)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.setSizeCallback(size_fn(tree, hmin, hmax, grad))
    gmsh.option.setNumber('Mesh.Algorithm3D', 1)
    gmsh.option.setNumber('Mesh.Optimize', 1)
    gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)
    gmsh.model.mesh.generate(3)
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    _, en = gmsh.model.mesh.getElementsByType(4)
    en = np.asarray(en, int).reshape(-1, 4)
    q = gmsh.model.mesh.getElementQualities(gmsh.model.mesh.getElementsByType(4)[0], 'gamma')
    gmsh.finalize()
    # renumber: keep the global ids of the boundary nodes, append new interior nodes
    ids = {}
    Xout = list(Xall)
    for t_, x_ in zip(nt, xyz):
        t_ = int(t_)
        if t_ - 1 < len(Xall) and (t_ - 1) in set_used(used):
            ids[t_] = t_ - 1
        else:
            ids[t_] = len(Xout); Xout.append(x_)
    mtets = np.vectorize(ids.get)(en)
    return dict(X=np.array(Xout), mtets=mtets, htets=A['tets'], face_tri=face_tri, surf=A['surf'],
                gamma=np.asarray(q))


_used_cache = {}
def set_used(used):
    k = id(used)
    if k not in _used_cache: _used_cache[k] = set(int(u) for u in used)
    return _used_cache[k]
