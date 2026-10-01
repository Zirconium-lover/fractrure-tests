"""Hydride packets as torn sheets: geometry only (units: micrometres).

A packet is a mid-surface over a ragged planar domain with holes, lifted out of
its plane by a smooth waviness plus a softened zigzag, and given a smooth
thickness field t(u, v) in [T_MIN, T_MAX] that thins towards every edge.  It is
later extruded into ONE layer of prisms (one element through the thickness).

Frames (load along x):
    circumferential packet  normal y, in-plane u = x, v = z  (zigzag along x, +-19 deg)
    radial packet           normal x, in-plane u = y, v = z  (zigzag along y, +-35 deg,
                            the smoothed 'deck of cards' of the reoriented packets)
Morphology numbers: Zircaloy-4 FIB-SEM thesis, tables 3.1-3.3 (packet length
~25-100 um, packet thickness 2-4 um, platelets 0.6-0.8 um thick and 4-5 um long,
packet spacing 57-71 um).
"""
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

T_MIN, T_MAX = 0.5, 2.0          # sheet thickness range, um


def ragged_loop(rng, a, b, n=256, rough=0.18, kmax=28):
    """Closed ragged outline: an ellipse a x b with a power-law radial noise."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    for k in range(2, kmax):
        r += rng.normal(0, rough * k ** -1.15) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    r = np.clip(r, 0.45, 1.6)
    return np.c_[a * r * np.cos(th), b * r * np.sin(th)]


class Field2D:
    """Smooth random field on the (u, v) plane: a sum of random plane waves."""

    def __init__(self, rng, lam, n=12):
        k = 2 * np.pi / (lam * rng.uniform(0.7, 1.4, n))
        ang = rng.uniform(0, np.pi, n)
        self.k = np.c_[k * np.cos(ang), k * np.sin(ang)]
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.n = n

    def __call__(self, uv):
        return np.sqrt(2.0 / self.n) * np.cos(uv @ self.k.T + self.ph).sum(1)


def soft_zigzag(s, half, slope, soft):
    """Triangle wave of half-period `half` and flank slope `slope`, its tips
    rounded over a width `soft` (curvature radius ~ soft / slope)."""
    p = np.mod(s / half, 2.0) - 1.0                      # -1..1
    tri = half * (np.sqrt(p * p + (soft / half) ** 2) - 1.0)  # rounded |p|-1
    return slope * (tri + half / 2.0)


class Sheet:
    """One leaf: frame (c, e1, e2, n), domain polygon, out-of-plane w, thickness t."""

    def __init__(self, rng, kind, c, e1, e2, n, poly, zz, wav, branch=False):
        self.kind, self.c, self.e1, self.e2, self.n = kind, np.asarray(c, float), e1, e2, n
        self.poly, self.branch = poly, branch
        self.zz_half, self.zz_slope = zz
        self.zz_dir = rng.uniform(-0.25, 0.25)           # zigzag axis meanders a little
        self.meander = Field2D(rng, 40.0, 6)
        self.wave = Field2D(rng, rng.uniform(25, 45), 10)
        self.wav = wav
        self.tfield = Field2D(rng, rng.uniform(8, 16), 12)

    def w(self, uv):
        s = uv[:, 0] * np.cos(self.zz_dir) + uv[:, 1] * np.sin(self.zz_dir) + 2.0 * self.meander(uv)
        w = self.wav * self.wave(uv) + soft_zigzag(s, self.zz_half, self.zz_slope, 1.2)
        if self.branch:                                   # a petal starts flat at its root
            w = w - (self.wav * self.wave(uv[:1] * 0) + soft_zigzag(np.zeros(1), self.zz_half, self.zz_slope, 1.2))
            w = w * np.clip(uv[:, 0] / 6.0, 0.0, 1.0)
        return w

    def thickness(self, uv, dist_edge):
        q = 0.5 + 0.5 * np.tanh(0.9 * self.tfield(uv))   # 0..1, smooth
        t = T_MIN + (T_MAX - T_MIN) * q
        taper = np.clip(dist_edge / 4.0, 0.0, 1.0)       # thin out over the last 4 um
        return T_MIN + (t - T_MIN) * taper

    def to3d(self, uv):
        return self.c + uv[:, :1] * self.e1 + uv[:, 1:2] * self.e2 + self.w(uv)[:, None] * self.n


FRAMES = {'circ': (np.array([1., 0, 0]), np.array([0, 0, 1.]), np.array([0, 1., 0])),
          'rad': (np.array([0, 1., 0]), np.array([0, 0, 1.]), np.array([1., 0, 0]))}


def _rot(xy, a):
    c, s_ = np.cos(a), np.sin(a)
    return xy @ np.array([[c, s_], [-s_, c]])


def torn_domain(rng, A, B, long_axis_jitter=0.35):
    """Ragged multi-lobe outline A x B (half sizes), torn from the edge and
    perforated by holes elongated along the long (u) axis."""
    from shapely.geometry import Point
    lobes = []
    for _ in range(rng.integers(2, 5)):
        a = A * rng.uniform(0.45, 0.8); b = B * rng.uniform(0.35, 0.75)
        c = np.array([rng.uniform(-A + a, A - a), rng.uniform(-B + b, B - b)])
        lob = _rot(ragged_loop(rng, a, b, n=360, rough=0.24, kmax=44), rng.uniform(-0.4, 0.4)) + c
        lobes.append(Polygon(lob).buffer(0))
    P = unary_union(lobes)
    if P.geom_type != 'Polygon':
        P = max(P.geoms, key=lambda g: g.area)
    cuts = []
    # tears: thin wedges cut from the edge towards the inside
    ext = np.asarray(P.exterior.coords)
    for _ in range(rng.integers(2, 6)):
        p0 = ext[rng.integers(len(ext))]
        inward = -p0 / (np.linalg.norm(p0) + 1e-9)
        d = _rot(inward[None], rng.uniform(-0.6, 0.6))[0]
        ln = rng.uniform(6, 18); w = rng.uniform(2.0, 3.5)
        nrm = np.array([-d[1], d[0]])
        cuts.append(Polygon([p0 - d * 2 + nrm * w / 2, p0 + d * ln, p0 - d * 2 - nrm * w / 2]))
    # holes: gaps between platelets, elongated roughly along u
    for _ in range(rng.integers(3, 9)):
        hl = rng.uniform(3, 10); hw = hl * rng.uniform(0.25, 0.6)
        for _try in range(30):
            hc = np.array([rng.uniform(-A, A), rng.uniform(-B, B)])
            H = Polygon(_rot(ragged_loop(rng, hl, hw, n=72, rough=0.25, kmax=12), rng.uniform(-long_axis_jitter, long_axis_jitter)) + hc).buffer(0)
            if P.buffer(-3.0).contains(H) and all(H.distance(h) > 3.0 for h in cuts):
                cuts.append(H); break
    P = P.difference(unary_union(cuts)) if cuts else P
    if P.geom_type != 'Polygon':
        P = max(P.geoms, key=lambda g: g.area)
    return P


def make_packet(rng, kind, L, margin_x):
    """Random packet of one kind inside the box L = (Lx, Ly, Lz), clipped by the
    lateral faces it crosses, never reaching the grips (x faces)."""
    e1, e2, n = FRAMES[kind]
    if kind == 'circ':                       # u = x (TD) long, v = z (RD)
        A = rng.uniform(28, 50); B = rng.uniform(20, 40)
        c = np.array([rng.uniform(margin_x + A, L[0] - margin_x - A), rng.uniform(10, L[1] - 10),
                      rng.uniform(0.1 * L[2], 0.9 * L[2])])
        lim = box(margin_x - c[0], -c[2], L[0] - margin_x - c[0], L[2] - c[2])
        zz = (rng.uniform(3.0, 5.5), np.tan(np.radians(19)))
    else:                                    # u = y (ND), v = z (RD): ribbons along RD
        A = rng.uniform(22, 45); B = rng.uniform(25, 45)
        c = np.array([rng.uniform(margin_x + 8, L[0] - margin_x - 8), rng.uniform(0.05 * L[1], 0.95 * L[1]),
                      rng.uniform(0.1 * L[2], 0.9 * L[2])])
        lim = box(-c[1], -c[2], L[1] - c[1], L[2] - c[2])
        zz = (rng.uniform(2.5, 4.0), np.tan(np.radians(35)))
    P = torn_domain(rng, A, B)
    P = P.intersection(lim)
    if P.geom_type != 'Polygon':
        P = max(P.geoms, key=lambda g: g.area)
    P = P.buffer(0.7).buffer(-0.7).simplify(0.25).intersection(lim)   # no slivers < ~1.4 um
    if P.geom_type != 'Polygon':
        P = max(P.geoms, key=lambda g: g.area)
    sh = Sheet(rng, kind, c, e1, e2, n, P, zz, wav=rng.uniform(1.5, 3.0))
    sh.lim = lim
    return sh


def make_branch(rng, parent):
    """A petal leaving the parent sheet at 20-35 deg from a point inside it; its
    root stays one gap (~1.5 um) off the parent surface."""
    P = parent.poly
    for _ in range(50):
        u0 = rng.uniform(*P.bounds[0::2]); v0 = rng.uniform(*P.bounds[1::2])
        from shapely.geometry import Point
        if P.buffer(-6.0).contains(Point(u0, v0)):
            break
    else:
        return None
    ang = rng.uniform(np.radians(20), np.radians(35)) * rng.choice([-1, 1])
    side = rng.choice([-1, 1])
    phi = rng.uniform(0, 2 * np.pi)                      # in-plane direction of the petal
    d = np.cos(phi) * parent.e1 + np.sin(phi) * parent.e2
    nn = parent.n * side
    e1 = np.cos(ang) * d + np.sin(abs(ang)) * nn       # tilted out of the parent plane
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(parent.n, d); e2 /= np.linalg.norm(e2)
    n = np.cross(e1, e2); n /= np.linalg.norm(n)
    root = parent.to3d(np.array([[u0, v0]]))[0] + nn * 2.6
    a = rng.uniform(9, 18); b = rng.uniform(2.5, 6)
    outline = ragged_loop(rng, a, b, n=160, rough=0.2, kmax=20) + np.array([a, 0.0])
    poly = Polygon(outline).buffer(0).buffer(0.6).buffer(-0.6).simplify(0.25)
    zz = (parent.zz_half, parent.zz_slope * 0.6)
    s = Sheet(rng, parent.kind, root, e1, e2, n, poly, zz, wav=0.6, branch=True)
    s.lim = None
    return s
