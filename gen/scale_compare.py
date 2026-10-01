"""Model box against the thesis micrographs, at one common um scale."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from PIL import Image
from sections import cut
FIB = '/tmp/claude-0/-home-user/7e4d2794-1d9f-5b4f-9d02-74b6594c40ca/scratchpad/fib'
G = '/home/user/fractrure-tests/geom'
L = (150., 150., 75.)


def micro(ax, f, crop, ppu, title, box_at=None, fib=None):
    im = np.asarray(Image.open(f'{FIB}/{f}').convert('L').crop(crop))
    w, h = im.shape[1] / ppu, im.shape[0] / ppu
    ax.imshow(im, cmap='gray', extent=(0, w, 0, h), vmin=0, vmax=255)
    if box_at is not None:
        ax.add_patch(Rectangle(box_at, 150, 150, fill=False, ec='#FFB000', lw=2.4))
        ax.text(box_at[0] + 75, box_at[1] + 155, 'модель 150×150 мкм', color='#FFB000', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    if fib is not None:
        ax.add_patch(Rectangle(fib, 20, 5, fill=True, fc='#36A3FF', ec='#36A3FF'))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    return w, h


def model(ax, p, z=37.5, title=None):
    d = np.load(f'{G}/fn{p}/mesh.npz')
    ph, _ = cut(d['X'], d['htets'], 2, z)
    ax.add_patch(Rectangle((0, 0), 150, 150, fc='#D9D9D9', ec='#FFB000', lw=2.4))
    ax.add_collection(PolyCollection(ph, facecolors='#2b2b2b', edgecolors='#2b2b2b', linewidths=0.3))
    ax.set_xlim(0, 150); ax.set_ylim(0, 150); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f'модель Fn = {p} %, сечение TD–ND, z = {z:g} мкм', fontsize=10)


def bar(ax, x, y, l, txt, col='white'):
    ax.plot([x, x + l], [y, y], color=col, lw=3, solid_capstyle='butt')
    ax.text(x + l / 2, y + 3, txt, color=col, ha='center', va='bottom', fontsize=9, fontweight='bold')


# ---- 1. overview: mesoscale SEM (fig. 3.1, 3.3), 100 um bars = 138 / 156 px
fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 2, width_ratios=[812, 150], wspace=0.04, hspace=0.18)
for r, (f, ppu, p, ttl, box, fib) in enumerate([
        ('img-000.jpg', 1.38, 0, 'диссертация, рис. 3.1: окружные гидриды (0 МПа), SEM', (330, 220), (520, 300)),
        ('img-002.jpg', 1.56, 100, 'диссертация, рис. 3.3: радиальные гидриды (250 МПа), SEM', (300, 200), (430, 280))]):
    a = fig.add_subplot(gs[r, 0])
    w, h = micro(a, f, (0, 0, 1272, 790), ppu, ttl, box_at=box, fib=fib)
    bar(a, w - 140, 18, 100, '100 мкм', col='#FFB000')
    b = fig.add_subplot(gs[r, 1]); model(b, p, title=f'модель Fn = {p} %\n(та же шкала)')
    bar(b, 20, 10, 50, '50 мкм', col='black')
fig.suptitle('Масштаб расчётной области на фоне мезомасштабной микроструктуры\n'
             'жёлтая рамка – наша область 150 × 150 мкм в плоскости TD–ND (глубина 75 мкм по RD);  '
             'голубой – объём FIB-SEM ~20 × 5 мкм', fontsize=12)
fig.savefig(f'{G}/scale_overview.png', dpi=120, bbox_inches='tight'); plt.close(fig)

# ---- 2. same scale, detail: etched micrographs (fig. 3.9); 20 um bar = 66 px -> 3.3 px/um
PPU = 3.3
fig, axs = plt.subplots(2, 2, figsize=(12.5, 13.0), gridspec_kw=dict(wspace=0.05, hspace=0.12))
side = int(round(150 * PPU))
for r, (x0, y0, p, ttl) in enumerate([(120, 10, 0, 'диссертация, рис. 3.9а: окружные пакеты (травление)'),
                                      (805 + 150, 10, 100, 'диссертация, рис. 3.9б: радиальные пакеты (травление)')]):
    micro(axs[r, 0], 'img-008.jpg', (x0, y0, x0 + side, y0 + side), PPU, ttl)
    bar(axs[r, 0], 120, 6, 20, '20 мкм', col='black')
    axs[r, 0].add_patch(Rectangle((0, 0), 150, 150, fill=False, ec='#FFB000', lw=2.4))
    model(axs[r, 1], p)
    bar(axs[r, 1], 120, 6, 20, '20 мкм', col='black')
fig.suptitle('Одинаковый масштаб: поле 150 × 150 мкм на шлифе (слева) и сечение модели (справа)\n'
             'тёмные прямоугольники на фото – объёмы FIB-SEM (~5 × 20 мкм)', fontsize=12)
fig.savefig(f'{G}/scale_detail.png', dpi=110, bbox_inches='tight'); plt.close(fig)
print('ok')
