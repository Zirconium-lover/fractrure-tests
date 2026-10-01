"""3D picture of the hydride leaves (pyvista, off-screen)."""
import sys, pickle, numpy as np, pyvista as pv
pv.OFF_SCREEN = True


def leaf_surface(m):
    n = len(m['B']); X = np.r_[m['B'], m['T']]
    tri = m['tri']
    faces = [tri, tri[:, ::-1] + n]
    side = []
    for i, j in m['bedge']:
        side += [(i, j, j + n), (i, j + n, i + n)]
    F = np.vstack(faces + [np.array(side)])
    return pv.PolyData(X, np.c_[np.full(len(F), 3), F].ravel())


def render(G, out, cam='iso', size=(1600, 1100), title=None):
    L = G['L']
    p = pv.Plotter(window_size=size)
    p.set_background('white')
    p.add_mesh(pv.Box(bounds=(0, L[0], 0, L[1], 0, L[2])), style='wireframe', color='#888888', line_width=1.2)
    for s, m in zip(G['sheets'], G['meshes']):
        col = '#D9534F' if s.kind == 'rad' else '#2E7FB8'
        if s.branch: col = '#F08A84' if s.kind == 'rad' else '#6FB1DE'
        p.add_mesh(leaf_surface(m), color=col, smooth_shading=True, specular=0.3)
    p.add_axes(xlabel='x (TD, load)', ylabel='y (ND)', zlabel='z (RD)')
    if cam == 'iso':
        p.camera_position = [(L[0] * 2.2, -L[1] * 1.3, L[2] * 2.6), (L[0] / 2, L[1] / 2, L[2] / 2), (0, 0, 1)]
    if title: p.add_text(title, font_size=12, color='black')
    p.screenshot(out)
    p.close()


if __name__ == '__main__':
    G = pickle.load(open(sys.argv[1], 'rb'))
    render(G, sys.argv[2], title=sys.argv[3] if len(sys.argv) > 3 else None)
