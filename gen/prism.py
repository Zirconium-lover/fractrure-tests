"""Mid-surface triangulation (gmsh 2D) and the one-layer prism extrusion."""
import numpy as np
import gmsh
from shapely.geometry import LineString, Point


def _resample(coords, h):
    c = np.asarray(coords)[:-1]
    seg = np.r_[np.linalg.norm(np.diff(np.r_[c, c[:1]], axis=0), axis=1)]
    s = np.r_[0, np.cumsum(seg)]; n = max(int(round(s[-1] / h)), 8)
    t = np.linspace(0, s[-1], n, endpoint=False)
    cc = np.r_[c, c[:1]]
    return np.c_[np.interp(t, s, cc[:, 0]), np.interp(t, s, cc[:, 1])]


def mesh_domain(poly, h):
    """Triangulate a shapely polygon (with holes) with element size ~h."""
    gmsh.model.add('sheet')
    loops = []
    for ring in [poly.exterior] + list(poly.interiors):
        pts = _resample(ring.coords, h)
        tags = [gmsh.model.geo.addPoint(x, y, 0, h) for x, y in pts]
        lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
        loops.append(gmsh.model.geo.addCurveLoop(lines))
    gmsh.model.geo.addPlaneSurface(loops)
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 1)
    gmsh.model.mesh.generate(2)
    nt, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)[:, :2]
    et, en = gmsh.model.mesh.getElementsByType(2)
    idx = {t: i for i, t in enumerate(nt)}
    tri = np.array([idx[t] for t in en]).reshape(-1, 3)
    gmsh.model.remove()
    used = np.unique(tri)
    remap = -np.ones(len(xyz), int); remap[used] = np.arange(len(used))
    return xyz[used], remap[tri]


def vertex_normals(X, tri):
    fn = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    N = np.zeros_like(X)
    for k in range(3):
        np.add.at(N, tri[:, k], fn)
    return N / np.linalg.norm(N, axis=1, keepdims=True)


def extrude(sheet, h, L):
    """Prism layer of one sheet.  Returns dict with bottom/top coordinates
    (top = bottom + offset), triangles, boundary edges, on-face flags."""
    uv, tri = mesh_domain(sheet.poly, h)
    X = sheet.to3d(uv)
    # orient triangles so their normal follows sheet.n
    fn = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
    flip = fn @ sheet.n < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    N = vertex_normals(X, tri)
    # distance to the free edge (edges clipped by a lateral face are not free)
    bnd = sheet.poly.boundary
    if sheet.lim is not None:
        bnd = bnd.difference(sheet.lim.exterior.buffer(1e-3))
    d = np.array([bnd.distance(Point(p)) for p in uv]) if not bnd.is_empty else np.full(len(uv), 99.)
    t = sheet.thickness(uv, d)
    # vertices lying on a box face: keep the offset inside that face
    face = -np.ones(len(X), int)
    # only vertices on the clipped part of the boundary go onto a box face
    clip_line = None
    if sheet.lim is not None:
        clip_line = sheet.poly.boundary.intersection(sheet.lim.exterior.buffer(1e-3))
    near = np.zeros(len(X), bool)
    for ax in range(3):
        for side, val in ((0, 0.0), (1, L[ax])):
            dist = np.abs(X[:, ax] - val)
            on = dist < 0.35
            if clip_line is not None and not clip_line.is_empty:
                on &= np.array([clip_line.distance(Point(p)) < 1e-2 for p in uv])
            else:
                on[:] = False
            X[on, ax] = val
            face[on] = 2 * ax + side
            N[dist < 2.5, ax] = 0.0                # offsets stay parallel to a near face
            near |= (dist < 1.2) & ~on
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    B = X - 0.5 * t[:, None] * N
    T = X + 0.5 * t[:, None] * N
    # boundary edges of the mid-surface triangulation
    e = np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    bedge = u[cnt == 1]
    return dict(mid=X, B=B, T=T, t=t, tri=tri, bedge=bedge, face=face, uv=uv, near=near)


def prism_tets(tri, nb, off):
    """3 tets per prism, diagonals from the smallest index -> conforming.
    Bottom node i, top node i+off (same local order).  nb: global base index."""
    out = []
    for t in tri:
        a, b, c = sorted(t)
        a, b, c = a + nb, b + nb, c + nb
        A, Bt, C = a + off, b + off, c + off
        out += [(a, b, c, C), (a, b, Bt, C), (a, A, Bt, C)]
    return np.array(out)


def side_tris(bedge, nb, off):
    out = []
    for i, j in bedge:
        i, j = sorted((i, j)); i += nb; j += nb
        out += [(i, j, j + off), (i, j + off, i + off)]
    return np.array(out)


def prism_ok(X, tri, off):
    """Every tet of every prism keeps the orientation of its own prism."""
    T = prism_tets(tri, 0, off)
    v = tet_vol(X, T).reshape(-1, 3)
    a = np.sort(tri, axis=1)
    s0 = np.sign(np.einsum('ij,ij->i', np.cross(X[a[:, 1]] - X[a[:, 0]], X[a[:, 2]] - X[a[:, 0]]),
                           X[a[:, 0] + off] - X[a[:, 0]]))
    q = v * s0[:, None] * np.array([1.0, -1.0, 1.0])   # tet 2 is listed with the other handedness
    return q.min(), np.abs(v).sum()


def tet_vol(X, T):
    a, b, c, d = (X[T[:, k]] for k in range(4))
    return np.einsum('ij,ij->i', b - a, np.cross(c - a, d - a)) / 6.0
