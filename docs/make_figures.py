#!/usr/bin/env python3
# Copyright 2025-2026 Abdolvahab Khalili Sadaghiani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the README figures of object_detector_husky.

Usage::

    python3 docs/make_figures.py               # writes docs/figures/*.png
    python3 docs/make_figures.py --out /tmp/f  # somewhere else

Every figure is written twice, ``<name>_light.png`` and ``<name>_dark.png``,
for GitHub's light and dark themes.  Only NumPy and Matplotlib are needed.

The scenes in the figures are small synthetic examples.  Their geometry is
computed with the same rules as ``object_detector_husky/object_detector_husky.py``
(IoU, edge weights, kNN selection, greedy matching, Smooth-L1), so every edge,
weight and match drawn is what the package computes for those boxes.  No
figure shows a measured detection result.
"""

import argparse
import math
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

# ----------------------------------------------------------------------------
# Defaults of object_detector_husky.py that the figures depict
# ----------------------------------------------------------------------------
ALPHA_IOU = 0.7        # --alpha_iou
IOU_THR_GRAPH = 0.25   # --iou_thr_graph
KNN_DEFAULT = 5        # --knn
CAP_NODES = 20         # --cap_nodes
MATCH_IOU = 0.5        # greedy_match(..., iou_thresh=0.5)
EMB_DIM = 16           # --emb_dim
CHANNELS = 128         # --in/hidden/out_channels
LAYERS = 3             # --layers
NUM_CLASSES = 80       # COCO

# ----------------------------------------------------------------------------
# Themes (GitHub light / dark surfaces; categorical hues validated for CVD)
# ----------------------------------------------------------------------------
THEMES = {
    'light': dict(
        bg='#ffffff', panel='#f6f8fa', ink='#1f2328', ink2='#59636e',
        muted='#818b98', line='#d1d9e0', grid='#eaeef2',
        blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', tint=0.13,
        seq=['#eef5fd', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95'],
    ),
    'dark': dict(
        bg='#0d1117', panel='#151b23', ink='#f0f6fc', ink2='#9198a1',
        muted='#6e7681', line='#3d444d', grid='#21262d',
        blue='#3987e5', orange='#d95926', aqua='#199e70', tint=0.24,
        seq=['#151b23', '#13284a', '#104281', '#1c5cab', '#2a78d6', '#5598e7', '#9ec5f4'],
    ),
}


def mix(c1, c2, t):
    """Blend colour ``c1`` into ``c2`` with weight ``t`` (0..1)."""
    a = np.array(mcolors.to_rgb(c1))
    b = np.array(mcolors.to_rgb(c2))
    return tuple(t * a + (1.0 - t) * b)


def kind_colors(T, kind):
    """Edge and fill colour of a block of the given kind."""
    hue = {'frozen': T['blue'], 'train': T['orange'], 'super': T['aqua']}.get(kind)
    if hue is None:  # data / fixed operation
        return T['ink2'], T['panel']
    return hue, mix(hue, T['bg'], T['tint'])


def setup_fonts():
    """Use Inter when installed (matching mathtext), otherwise DejaVu Sans."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    if 'Inter' in names:
        plt.rcParams.update({
            'font.family': ['Inter', 'DejaVu Sans'],
            'mathtext.fontset': 'custom', 'mathtext.rm': 'Inter',
            'mathtext.it': 'Inter:italic', 'mathtext.bf': 'Inter:bold',
            'mathtext.fallback': 'stixsans',
        })
    else:
        plt.rcParams.update({'font.family': ['DejaVu Sans'],
                             'mathtext.fontset': 'dejavusans'})
    plt.rcParams.update({'axes.titlesize': 10, 'axes.titleweight': 'semibold',
                         'font.size': 9, 'svg.fonttype': 'path'})


# ----------------------------------------------------------------------------
# Geometry: the same rules as object_detector_husky.py
# ----------------------------------------------------------------------------
def iou_xyxy(a, b):
    """IoU of two XYXY boxes (compute_iou_xyxy)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def centers(boxes):
    """Box centres."""
    b = np.asarray(boxes, dtype=float)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def graph_edges(boxes, H, W, iou_thr=IOU_THR_GRAPH, knn=KNN_DEFAULT, alpha=ALPHA_IOU):
    """Replicate build_weighted_graph and summarise it per unordered pair.

    Returns {(i, j): dict(w, iou_rule, knn_ij, knn_ji, mult)} for i < j, where
    ``mult`` is how many times the directed edge i->j occurs in edge_index.
    """
    bx = np.asarray(boxes, dtype=float)
    n = len(bx)
    c = centers(bx)
    sigma = max(10.0, 0.10 * max(H, W))
    directed = []  # list of (i, j, w) exactly as appended by the package
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                continue
            dist = math.hypot(c[i, 0] - c[j, 0], c[i, 1] - c[j, 1])
            iou = iou_xyxy(bx[i], bx[j])
            w = alpha * iou + (1.0 - alpha) * math.exp(-dist / sigma)
            if iou >= iou_thr:
                directed += [(i, j, w, 'iou'), (j, i, w, 'iou')]
            row.append((dist, j, w))
        row.sort(key=lambda t: t[0])
        for k in range(min(knn, len(row))):
            _, j, w = row[k]
            directed += [(i, j, w, ('knn', i)), (j, i, w, ('knn', i))]
    pairs = {}
    for i, j, w, rule in directed:
        key = (min(i, j), max(i, j))
        p = pairs.setdefault(key, dict(w=w, iou_rule=False, knn_ij=False,
                                       knn_ji=False, mult=0))
        if rule == 'iou':
            p['iou_rule'] = True
        elif rule[1] == key[0]:
            p['knn_ij'] = True
        else:
            p['knn_ji'] = True
        if (i, j) == key:
            p['mult'] += 1
    return pairs, sigma


def greedy_match(pred, gt, thr=MATCH_IOU):
    """Replicate greedy_match and also return the IoU matrix."""
    P, G = len(pred), len(gt)
    m = np.array([[iou_xyxy(pred[i], gt[j]) for j in range(G)] for i in range(P)])
    work = m.copy()
    matches, used_p, used_g = [], set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(work), work.shape)
        if work[i, j] < thr:
            break
        if i in used_p or j in used_g:
            work[i, j] = -1.0
            continue
        matches.append((int(i), int(j)))
        used_p.add(i)
        used_g.add(j)
        work[i, :] = -1.0
        work[:, j] = -1.0
    return m, matches


# ----------------------------------------------------------------------------
# Drawing helpers for block diagrams (data units, equal aspect)
# ----------------------------------------------------------------------------
class Board:
    """A blank canvas measured in abstract units, 1 unit = width_in / w_units inch."""

    def __init__(self, T, w_units, h_units, width_in):
        self.T = T
        self.fig = plt.figure(figsize=(width_in, width_in * h_units / w_units),
                              facecolor=T['bg'])
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w_units)
        self.ax.set_ylim(0, h_units)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.pt = width_in * 72.0 / w_units  # points per unit

    def u(self, points):
        """Convert points to canvas units."""
        return points / self.pt

    def block(self, cx, cy, w, h, title, sub=None, kind='data', tfs=9.6, sfs=8.3,
              r=1.1, lw=1.3, ls='-', fill=None, edge=None, title_color=None,
              weight='semibold', align='center'):
        """Rounded block with a title and optional sub-lines; returns its anchors."""
        T = self.T
        ec, fc = kind_colors(T, kind)
        ec = edge or ec
        fc = fill or fc
        self.ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
            fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
        n_sub = 0 if not sub else sub.count('\n') + 1
        th = self.u(tfs * 1.22)
        sh = self.u(sfs * 1.32) * n_sub
        gap = self.u(3.0) if sub else 0
        top = cy + (th + gap + sh) / 2
        x = cx if align == 'center' else cx - w / 2 + self.u(7)
        self.ax.text(x, top, title, ha=align, va='top', fontsize=tfs,
                     fontweight=weight, color=title_color or T['ink'], zorder=5)
        if sub:
            self.ax.text(x, top - th - gap, sub, ha=align, va='top', fontsize=sfs,
                         color=T['ink2'], linespacing=1.32, zorder=5)
        return dict(l=cx - w / 2, r=cx + w / 2, t=cy + h / 2, b=cy - h / 2, cx=cx, cy=cy)

    def container(self, x0, y0, x1, y1, label, kind='train', r=1.6, fs=8.8):
        """Large rounded group frame with a caption on its top edge."""
        T = self.T
        ec, _ = kind_colors(T, kind)
        fc = mix(ec, T['bg'], T['tint'] * 0.45)
        self.ax.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0, boxstyle=f'round,pad=0,rounding_size={r}',
            fc=fc, ec=ec, lw=1.2, ls=(0, (4, 2.5)), zorder=1))
        self.ax.text(x0 + self.u(8), y1 - self.u(5), label, ha='left', va='top',
                     fontsize=fs, fontweight='semibold', color=T['ink'], zorder=5)

    def arrow(self, pts, color=None, lw=1.25, dashed=False, head=True, z=4):
        """Polyline arrow through ``pts``; dashed lines get a solid head."""
        T = self.T
        color = color or T['ink2']
        xs, ys = zip(*pts)
        ls = (0, (3.2, 2.2)) if dashed else '-'
        self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle='butt',
                     dash_capstyle='butt', zorder=z)
        if head:
            (xa, ya), (xb, yb) = pts[-2], pts[-1]
            seg = math.hypot(xb - xa, yb - ya)
            k = min(1.0, self.u(9) / max(seg, 1e-9))
            start = (xb - (xb - xa) * k, yb - (yb - ya) * k)
            self.ax.add_patch(FancyArrowPatch(
                start, (xb, yb), arrowstyle='-|>', mutation_scale=9.5, lw=lw,
                color=color, shrinkA=0, shrinkB=0, zorder=z + 0.1))

    def label(self, x, y, s, fs=7.8, color=None, ha='center', va='center', bg=True,
              weight='normal', style='normal'):
        """Small annotation; ``bg`` paints the surface behind it to cut lines."""
        T = self.T
        kw = {}
        if bg:
            kw['bbox'] = dict(boxstyle='round,pad=0.18', fc=T['bg'], ec='none')
        self.ax.text(x, y, s, fontsize=fs, color=color or T['ink2'], ha=ha, va=va,
                     fontweight=weight, fontstyle=style, zorder=6, **kw)

    def oplus(self, cx, cy, r=2.0):
        """Summation node."""
        T = self.T
        self.ax.add_patch(Circle((cx, cy), r, fc=T['bg'], ec=T['ink'], lw=1.3, zorder=5))
        self.ax.plot([cx - r * 0.55, cx + r * 0.55], [cy, cy], color=T['ink'], lw=1.3, zorder=6)
        self.ax.plot([cx, cx], [cy - r * 0.55, cy + r * 0.55], color=T['ink'], lw=1.3, zorder=6)
        return dict(l=cx - r, r=cx + r, t=cy + r, b=cy - r, cx=cx, cy=cy)

    def file_icon(self, x, y, w, h, title, sub=None, fs=8.0):
        """Document shape with a folded corner (an artefact on disk)."""
        T = self.T
        f = min(w, h) * 0.28
        poly = [(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h), (x, y + h)]
        self.ax.add_patch(Polygon(poly, closed=True, fc=T['panel'], ec=T['ink2'],
                                  lw=1.1, zorder=3))
        self.ax.plot([x + w - f, x + w - f, x + w], [y + h, y + h - f, y + h - f],
                     color=T['ink2'], lw=1.1, zorder=4)
        self.ax.text(x + self.u(6), y + h - self.u(5), title, ha='left', va='top',
                     fontsize=fs, fontweight='semibold', color=T['ink'], zorder=5,
                     family='DejaVu Sans Mono' if title.endswith(('.json', '.pt', '/', '.pkl'))
                     else None)
        if sub:
            self.ax.text(x + self.u(6), y + h - self.u(5 + fs * 1.35), sub, ha='left',
                         va='top', fontsize=fs - 0.6, color=T['ink2'], zorder=5,
                         linespacing=1.3)

    def legend(self, x, y, items, fs=8.0, sw=3.2, sh=2.2, gap=2.6):
        """Row of swatches: items = [(kind or ('line', dashed), text), ...]."""
        T = self.T
        for kind, text in items:
            if isinstance(kind, tuple):  # arrow sample
                _, dashed, color = kind
                self.arrow([(x, y), (x + sw + 1.2, y)], color=color or T['ink2'],
                           dashed=dashed)
                x += sw + 1.2
            else:
                ec, fc = kind_colors(T, kind)
                self.ax.add_patch(FancyBboxPatch(
                    (x, y - sh / 2), sw, sh, boxstyle='round,pad=0,rounding_size=0.5',
                    fc=fc, ec=ec, lw=1.2, zorder=3))
                x += sw
            t = self.ax.text(x + 1.0, y, text, fontsize=fs, color=T['ink2'],
                             ha='left', va='center')
            self.fig.canvas.draw()
            bb = t.get_window_extent().transformed(self.ax.transData.inverted())
            x = bb.x1 + gap

    def save(self, path, dpi):
        self.fig.savefig(path, dpi=dpi, facecolor=self.T['bg'])
        plt.close(self.fig)


def style_axes(ax, T, grid_axis='y'):
    """Recessive hairline axes for the chart panels."""
    ax.set_facecolor(T['bg'])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(T['line'])
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=T['ink2'], labelsize=8, length=3, width=0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=T['grid'], lw=0.8)
        ax.set_axisbelow(True)
    ax.xaxis.label.set_color(T['ink2'])
    ax.yaxis.label.set_color(T['ink2'])


def panel_title(ax, T, s, x=0.0, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, ha='left', va='bottom', fontsize=9.6,
            fontweight='semibold', color=T['ink'])


# ----------------------------------------------------------------------------
# Figure 1: system overview
# ----------------------------------------------------------------------------
def fig_overview(T, path, dpi):
    B = Board(T, 140, 63, 10.0)
    ax = B.ax
    yt, yb, h = 40.0, 13.0, 13.0   # lane centres, block height
    sfs = 8.0

    B.legend(1.5, 60.2, [('frozen', 'frozen'), ('train', 'trainable'),
                         ('super', 'supervision'), ('data', 'data / fixed operation'),
                         (('a', True, T['orange']), 'gradient')], fs=7.8)
    B.label(1.5, 50.6, 'INFERENCE', fs=7.6, ha='left', color=T['muted'], bg=False,
            weight='semibold')
    ax.plot([1.5, 138.5], [27.4, 27.4], color=T['line'], lw=0.8)
    B.label(1.5, 25.2, 'TRAINING', fs=7.6, ha='left', color=T['muted'], bg=False,
            weight='semibold')

    img = B.block(6.5, yt, 10, h, 'Image', 'H × W × 3', sfs=sfs)
    yolo = B.block(22.5, yt, 15, h, 'YOLOv8n', 'COCO-pretrained\nfrozen · NMS',
                   kind='frozen', sfs=sfs)
    det = B.block(42.0, yt, 17, h, 'Top-N detections',
                  f'N ≤ {CAP_NODES} by score\nboxes, scores,\nclass ids', sfs=sfs)
    gra = B.block(62.0, yt, 16, h, 'Proposal graph',
                  'IoU ∪ kNN edges\n' r'weights $w_{ij}$', sfs=sfs)
    pc = param_counts()
    B.container(73.0, yt - 8.4, 115.5, yt + 12.2,
                f'GNN refiner · trainable · {pc["total_gcn"] / 1000:.1f} k params')
    enc = B.block(81.5, yt, 13, 12.0, 'Encoder', 'geometry,\nscore, class\nMLP → 128',
                  kind='train', tfs=9.2, sfs=7.8)
    gnn = B.block(95.6, yt, 11, 12.0, f'GNN × {LAYERS}', 'GCNConv\n128 → 128\nReLU',
                  kind='train', tfs=9.2, sfs=7.8)
    head = B.block(108.3, yt, 10.6, 12.0, 'Box head', 'Linear\n128 → 4\nΔ per box',
                   kind='train', tfs=9.2, sfs=7.8)
    add = B.oplus(119.6, yt, 2.0)
    out = B.block(131.2, yt, 14.6, h, 'Refined boxes', 'B′ = B + Δ\ndrawn & saved', sfs=sfs)

    chain = [(img, yolo), (yolo, det), (det, gra), (gra, enc), (enc, gnn), (gnn, head),
             (head, add), (add, out)]
    for a, b in chain:
        B.arrow([(a['r'], yt), (b['l'], yt)])
    B.label(116.0, yt + 2.3, 'Δ', fs=8.8, color=T['ink'], bg=False)
    ytop = 55.2
    B.arrow([(det['cx'], det['t']), (det['cx'], ytop), (add['cx'], ytop), (add['cx'], add['t'])])
    B.label(86.0, ytop, 'B  —  identity path: the refiner only predicts a correction Δ',
            fs=7.8)

    # training lane
    cache = B.block(22.5, yb + 1.0, 15, 10.5, 'YOLO cache', 'cache/*.pkl\ntrain / val',
                    tfs=9.2, sfs=7.8)
    B.arrow([(yolo['cx'], yolo['b']), (yolo['cx'], cache['t'])])
    B.arrow([(cache['r'], cache['cy']), (39.0, cache['cy']), (39.0, det['b'])])
    B.label(39.0, 22.6, 'read when\ntraining', fs=7.2)
    B.label(22.5, 4.4, 'requires_grad = False:\nno gradient reaches YOLO', fs=7.4,
            color=T['muted'], bg=False)

    gt = B.block(131.2, yb, 14.6, 12.0, 'Ground truth G', 'YOLO .txt\ncls cx cy w h',
                 kind='super', tfs=9.2, sfs=7.8)
    match = B.block(111.0, yb, 16.0, 12.0, 'Greedy matching',
                    'one-to-one\nIoU(B′, G) ≥ 0.5\nclass-agnostic', kind='super', tfs=9.2, sfs=7.8)
    loss = B.block(90.5, yb, 16.0, 12.0, 'Smooth-L1 loss', 'β = 1 px\nmatched boxes',
                   kind='super', tfs=9.2, sfs=7.8)
    B.arrow([(gt['l'], yb), (match['r'], yb)])
    B.arrow([(add['cx'], add['b']), (add['cx'], 23.2), (114.0, 23.2), (114.0, match['t'])])
    B.label(117.0, 23.2, 'B′', fs=8.2, color=T['ink'], bg=True)
    B.arrow([(match['l'], yb), (loss['r'], yb)])
    B.arrow([(loss['l'], yb), (77.0, yb), (77.0, yt - 8.4)], color=T['orange'], dashed=True)
    B.label(75.6, 18.6, 'gradient → encoder,\nGNN and head\n(Adam, lr 1e-3)', fs=7.3,
            ha='right', bg=False)
    B.save(path, dpi)


# ----------------------------------------------------------------------------
# Figure 2: graph construction
# ----------------------------------------------------------------------------
SCENE_W, SCENE_H = 640, 480
SCENE = [  # (x1, y1, x2, y2), score  -- a synthetic street scene
    ((40, 170, 130, 420), 0.91),   # 0 pedestrian
    ((88, 150, 183, 410), 0.86),   # 1 pedestrian (occludes 0)
    ((150, 165, 240, 415), 0.78),  # 2 pedestrian
    ((200, 175, 285, 420), 0.64),  # 3 pedestrian
    ((360, 250, 540, 410), 0.88),  # 4 bicycle
    ((395, 115, 505, 355), 0.83),  # 5 rider
    ((545, 45, 625, 125), 0.55),   # 6 traffic sign
    ((15, 25, 115, 105), 0.47),    # 7 small object
    ((290, 35, 375, 110), 0.52),   # 8 small object
]
FIG2_K = 2
NODE_R = 14


def draw_scene(ax, T, boxes, scores=None, box_alpha=1.0):
    ax.set_xlim(-6, SCENE_W + 6)
    ax.set_ylim(SCENE_H + 6, -6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0, 0), SCENE_W, SCENE_H,
                                boxstyle='round,pad=0,rounding_size=10',
                                fc=T['panel'], ec=T['line'], lw=0.9, zorder=0))
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc='none', ec=T['ink2'],
                               lw=1.0, alpha=box_alpha, zorder=1))
        if scores is not None:
            ax.text(x2 - 3, y2 - 4, f'{scores[k]:.2f}', fontsize=6.6, color=T['ink2'],
                    ha='right', va='bottom', zorder=2)


def draw_nodes(ax, T, c):
    for k, (x, y) in enumerate(c):
        ax.add_patch(Circle((x, y), NODE_R, fc=T['bg'], ec=T['ink'], lw=1.1, zorder=6))
        ax.text(x, y + 0.6, str(k), fontsize=7.0, color=T['ink'], ha='center', va='center',
                zorder=7, fontweight='semibold')


def edge_label(ax, T, x, y, s):
    ax.text(x, y, s, fontsize=6.8, color=T['ink'], ha='center', va='center', zorder=8,
            bbox=dict(boxstyle='round,pad=0.16', fc=T['bg'], ec='none'))


def fig_graph(T, path, dpi):
    boxes = [b for b, _ in SCENE]
    scores = [s for _, s in SCENE]
    c = centers(boxes)
    pairs, sigma = graph_edges(boxes, SCENE_H, SCENE_W, knn=FIG2_K)

    fig = plt.figure(figsize=(10.0, 6.9), facecolor=T['bg'])
    gs = fig.add_gridspec(2, 3, left=0.02, right=0.985, top=0.945, bottom=0.115,
                          wspace=0.06, hspace=0.36)
    axa, axb, axc = (fig.add_subplot(gs[0, k]) for k in range(3))
    axd = fig.add_subplot(gs[1, 0])
    axe = fig.add_subplot(gs[1, 1:])

    def caption(ax, s):
        ax.text(SCENE_W / 2, SCENE_H + 22, s, fontsize=7.6, color=T['ink2'], ha='center',
                va='top')

    # (a) proposals
    draw_scene(axa, T, boxes, scores)
    draw_nodes(axa, T, c)
    panel_title(axa, T, '(a) Top-N YOLO detections = nodes')
    caption(axa, 'number = YOLO score; features: geometry, score, class')

    # (b) IoU rule
    draw_scene(axb, T, boxes, box_alpha=0.5)
    for (i, j), p in pairs.items():
        if p['iou_rule']:
            axb.plot(*zip(c[i], c[j]), color=T['blue'], lw=2.4, zorder=4,
                     solid_capstyle='round')
            mx, my = (c[i] + c[j]) / 2
            dy = -34 if abs(c[i, 1] - c[j, 1]) < 40 else 0
            dx = 0 if dy else 46
            edge_label(axb, T, mx + dx, my + dy, f'{iou_xyxy(boxes[i], boxes[j]):.2f}')
    draw_nodes(axb, T, c)
    panel_title(axb, T, f'(b) IoU rule: IoU ≥ τ = {IOU_THR_GRAPH}')
    caption(axb, 'number = IoU; overlapping boxes are linked, any class')

    # (c) kNN rule
    draw_scene(axc, T, boxes, box_alpha=0.5)
    for (i, j), p in pairs.items():
        if p['knn_ij'] or p['knn_ji']:
            axc.plot(*zip(c[i], c[j]), color=T['orange'], lw=1.9, zorder=4,
                     ls=(0, (4, 2.4)), dash_capstyle='round')
    draw_nodes(axc, T, c)
    panel_title(axc, T, f'(c) kNN rule: k = {FIG2_K} nearest centres')
    caption(axc, f'no node is left isolated (package default k = {KNN_DEFAULT})')

    # (d) weighted graph
    draw_scene(axd, T, boxes, box_alpha=0.32)
    order = sorted(pairs.items(), key=lambda kv: kv[1]['w'])
    for (i, j), p in order:
        col = T['blue'] if p['iou_rule'] else T['orange']
        axd.plot(*zip(c[i], c[j]), color=col, lw=0.7 + 7.0 * p['w'], zorder=4,
                 solid_capstyle='round')
    for (i, j), p in order:
        if not p['iou_rule']:
            continue
        mx, my = (c[i] + c[j]) / 2
        if abs(c[i, 1] - c[j, 1]) < 40:  # near-horizontal edge: label below it
            edge_label(axd, T, mx, my + 36, f'{p["w"]:.2f}')
        else:  # near-vertical edge: label to its right
            edge_label(axd, T, mx + 46, my, f'{p["w"]:.2f}')
    (i, j), p = order[0]
    mx, my = (c[i] + c[j]) / 2
    edge_label(axd, T, mx, my - 18, f'{p["w"]:.2f}')
    draw_nodes(axd, T, c)
    panel_title(axd, T, r'(d) Union; line width $\propto w_{ij}$')
    caption(axd, 'blue: IoU rule · orange: kNN only')

    # (e) edge weight as a function of the centre offset of two equal boxes
    s = 100.0
    d = np.linspace(0, 330, 661)
    iou = np.where(d < s, (s - d) / (s + d), 0.0)
    t_iou = ALPHA_IOU * iou
    t_dist = (1 - ALPHA_IOU) * np.exp(-d / sigma)
    w = t_iou + t_dist
    style_axes(axe, T)
    d_tau = s * (1 - IOU_THR_GRAPH) / (1 + IOU_THR_GRAPH)
    axe.axvspan(0, d_tau, color=T['blue'], alpha=0.10, lw=0, zorder=0)
    axe.text(d_tau / 2, 0.985, 'IoU ≥ τ:\nIoU edge', fontsize=7.4, color=T['ink2'],
             ha='center', va='top')
    axe.plot(d, t_iou, color=T['blue'], lw=2, label=r'$\alpha\,$IoU')
    axe.plot(d, t_dist, color=T['orange'], lw=2, ls=(0, (4, 2.4)),
             label=r'$(1-\alpha)\,e^{-d/\sigma}$')
    axe.plot(d, w, color=T['ink'], lw=2.2, label=r'$w_{ij}$ (sum)')
    axe.axvline(sigma, color=T['line'], lw=1.0, zorder=0)
    axe.text(sigma + 4, 0.62, f'd = σ = {sigma:.0f} px', fontsize=7.4, color=T['ink2'],
             ha='left', va='center')
    axe.axvline(s, color=T['line'], lw=1.0, zorder=0)
    axe.text(s + 4, 0.49, 'd = s: boxes no\nlonger overlap', fontsize=7.4,
             color=T['ink2'], ha='left', va='center')
    axe.text(42, w[84] + 0.03, r'$w_{ij}$', fontsize=8.6, color=T['ink'], ha='left',
             va='bottom')
    axe.text(1.5, 0.47, r'$\alpha\,$IoU', fontsize=8.2, color=T['ink'],
             ha='left', va='top')
    axe.text(150, t_dist[300] + 0.03, r'$(1-\alpha)\,e^{-d/\sigma}$: only distance is left',
             fontsize=8.0, color=T['ink'], ha='left', va='bottom')
    axe.set_xlim(0, 330)
    axe.set_ylim(0, 1.0)
    axe.set_xlabel('centre offset d between two equal 100 × 100 px boxes (px)', fontsize=8)
    axe.set_ylabel('edge weight', fontsize=8)
    leg = axe.legend(loc='upper right', fontsize=7.8, frameon=False, handlelength=2.6)
    for txt in leg.get_texts():
        txt.set_color(T['ink2'])
    panel_title(axe, T, rf'(e) $w_{{ij}} = \alpha\,\mathrm{{IoU}}_{{ij}} + (1-\alpha)\,'
                rf'e^{{-d_{{ij}}/\sigma}}$,   α = {ALPHA_IOU},   σ = 0.1 · max(H, W)')
    fig.text(0.02, 0.012, f'Synthetic {SCENE_W} × {SCENE_H} scene. Edges, IoUs and weights '
             'are computed with the package rules; k = 2 here for legibility.',
             fontsize=7.4, color=T['muted'], ha='left', va='bottom')
    fig.savefig(path, dpi=dpi, facecolor=T['bg'])
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: node encoder and GNN refiner
# ----------------------------------------------------------------------------
def param_counts():
    """Exact parameter counts of GNNRefiner with the default sizes."""
    raw = 7 + EMB_DIM
    emb = NUM_CLASSES * EMB_DIM
    mlp = (raw * CHANNELS + CHANNELS) + (CHANNELS * CHANNELS + CHANNELS)
    gcn = LAYERS * (CHANNELS * CHANNELS + CHANNELS)           # lin (no bias) + bias
    graph = LAYERS * (2 * CHANNELS * CHANNELS + CHANNELS)     # lin_rel (+bias) + lin_root
    head = CHANNELS * 4 + 4
    return dict(raw=raw, emb=emb, mlp=mlp, gcn=gcn, graph=graph, head=head,
                total_gcn=emb + mlp + gcn + head, total_graph=emb + mlp + graph + head)


def fig_refiner(T, path, dpi):
    pc = param_counts()
    B = Board(T, 150, 62, 10.0)
    ax = B.ax

    # --- feature vector as a segmented strip ------------------------------
    names = ['cx/W', 'cy/H', 'w/W', 'h/H', 'area', 'w/h', 'score']
    x0, y0, cw, ch = 2.5, 44.0, 4.3, 6.6
    ec_d, fc_d = kind_colors(T, 'data')
    for k, nme in enumerate(names):
        ax.add_patch(FancyBboxPatch((x0 + k * cw + 0.15, y0), cw - 0.3, ch,
                                    boxstyle='round,pad=0,rounding_size=0.5',
                                    fc=fc_d, ec=ec_d, lw=1.0, zorder=3))
        ax.text(x0 + k * cw + cw / 2, y0 + ch / 2, nme, rotation=90, fontsize=6.9,
                ha='center', va='center', color=T['ink'], zorder=5)
    xe = x0 + 7 * cw + 2.6
    ew = 1.3
    ec_t, fc_t = kind_colors(T, 'train')
    for k in range(EMB_DIM):
        ax.add_patch(FancyBboxPatch((xe + k * ew + 0.1, y0), ew - 0.2, ch,
                                    boxstyle='round,pad=0,rounding_size=0.25',
                                    fc=fc_t, ec=ec_t, lw=0.8, zorder=3))
    xend = xe + EMB_DIM * ew

    def brace(xa, xb, y, text):
        ax.plot([xa, xa, xb, xb], [y + 0.9, y, y, y + 0.9], color=T['ink2'], lw=0.9)
        ax.plot([(xa + xb) / 2] * 2, [y, y - 0.9], color=T['ink2'], lw=0.9)
        ax.text((xa + xb) / 2, y - 1.4, text, fontsize=7.5, color=T['ink2'], ha='center',
                va='top', linespacing=1.3)
    brace(x0 + 0.2, x0 + 6 * cw - 0.2, y0 - 1.1, 'geometry, normalised\nby image W, H')
    brace(x0 + 6 * cw + 0.2, x0 + 7 * cw - 0.2, y0 - 1.1, 'YOLO\nscore')
    brace(xe + 0.1, xend - 0.1, y0 - 1.1,
          f'class embedding, learned\n{NUM_CLASSES} × {EMB_DIM} = {pc["emb"]:,} params')
    ax.text(x0, y0 + ch + 1.5, rf'Node feature  $x_i \in \mathbb{{R}}^{{{pc["raw"]}}}$'
            f'   (7 + {EMB_DIM})', fontsize=9.4, fontweight='semibold', color=T['ink'],
            ha='left', va='bottom')

    # --- message-passing equations ----------------------------------------
    xr = 80.0
    ax.text(xr, 58.6, 'One GNN layer (message passing)', fontsize=9.4,
            fontweight='semibold', color=T['ink'], ha='left', va='top')
    ax.text(xr, 54.4, r'GCNConv:   $H^{(\ell+1)} = \hat D^{-1/2}\,\hat A\,\hat D^{-1/2}\,'
            r'H^{(\ell)}\,\Theta^{(\ell)} + b$', fontsize=8.6, color=T['ink'], ha='left',
            va='top')
    ax.text(xr + 13.2, 50.0, r'$\hat A = A + I$,   $A_{ij}$ = sum of the weights of all '
            r'parallel edges i → j', fontsize=7.7, color=T['ink2'], ha='left', va='top')
    ax.text(xr, 45.6, r'GraphConv:   $h_i^{(\ell+1)} = \Theta_1 h_i^{(\ell)} + \Theta_2 '
            r'\sum_{j \in \mathcal{N}(i)} w_{ji}\, h_j^{(\ell)}$', fontsize=8.6,
            color=T['ink'], ha='left', va='top')
    ax.text(xr, 39.4, r'then ReLU; residual $h^{(\ell)}$ added with --use_residual; '
            'dropout between layers', fontsize=7.7, color=T['ink2'], ha='left', va='top')

    # --- pipeline ------------------------------------------------------------
    yp, hb = 19.5, 17.0
    sf = 7.9
    mlp = B.block(14.5, yp, 25, hb, 'Feature MLP',
                  f'Linear {pc["raw"]} → {CHANNELS}, ReLU\nLinear {CHANNELS} → {CHANNELS}, ReLU\n'
                  f'{pc["mlp"]:,} params', kind='train', tfs=9.4, sfs=sf)
    gx, gw = 48.5, 31.0
    for k in (2, 1):
        ax.add_patch(FancyBboxPatch((gx - gw / 2 + 1.1 * k, yp - hb / 2 + 1.1 * k), gw, hb,
                                    boxstyle='round,pad=0,rounding_size=1.1', fc=fc_t, ec=ec_t,
                                    lw=1.0, zorder=2 - k * 0.1))
    gnn = B.block(gx, yp, gw, hb, f'GNN layers ℓ = 1 … {LAYERS}',
                  f'GCNConv {CHANNELS} → {CHANNELS}, ReLU\n+ residual if --use_residual\n'
                  f'dropout 0.1 between layers\n{pc["gcn"]:,} params ({pc["graph"]:,} GraphConv)',
                  kind='train', tfs=9.4, sfs=sf)
    head = B.block(79.5, yp, 16, hb, 'Box head', f'Linear {CHANNELS} → 4\n{pc["head"]} params',
                   kind='train', tfs=9.4, sfs=sf)
    dlt = B.block(99.5, yp, 14, hb, 'Δᵢ', '(dx₁, dy₁,\n dx₂, dy₂)\nin pixels', tfs=10.5, sfs=sf)
    add = B.oplus(113.0, yp, 2.1)
    out = B.block(132.0, yp, 22, hb, 'B′ᵢ = Bᵢ + Δᵢ', 'refined box\n(x₁, y₁, x₂, y₂)',
                  tfs=9.4, sfs=sf)
    xin = (x0 + xend) / 2
    B.arrow([(xin, 36.4), (xin, 32.0), (mlp['cx'], 32.0), (mlp['cx'], mlp['t'])])
    B.arrow([(mlp['r'], yp), (gnn['l'], yp)])
    B.label((mlp['r'] + gnn['l']) / 2, yp + 2.7, r'$h^{(0)}$', fs=8.6, bg=False,
            color=T['ink'])
    B.arrow([(gnn['r'] + 2.2, yp), (head['l'], yp)])
    B.label((gnn['r'] + 2.2 + head['l']) / 2, yp + 2.7, r'$h^{(L)}$', fs=8.6, bg=False,
            color=T['ink'])
    B.arrow([(head['r'], yp), (dlt['l'], yp)])
    B.arrow([(dlt['r'], yp), (add['l'], yp)])
    B.arrow([(add['r'], yp), (out['l'], yp)])
    B.arrow([(add['cx'], 4.6), (add['cx'], add['b'])])
    B.label(add['cx'] + 1.5, 3.0, 'YOLO box Bᵢ (identity path)', fs=7.7, bg=False, ha='left')

    # graph input to the GNN
    B.arrow([(gx + 8.0, 33.0), (gx + 8.0, gnn['t'] + 2.2)])
    B.label(gx + 8.0, 35.0, r'graph: edge_index, $w_{ij}$', fs=7.8, color=T['ink'], bg=False)

    ax.text(2.5, 3.0, f'Trainable: {pc["total_gcn"]:,} parameters with GCNConv (default), '
            f'{pc["total_graph"]:,} with GraphConv. YOLOv8n stays frozen.',
            fontsize=7.7, color=T['ink2'], ha='left', va='center')
    B.save(path, dpi)


# ----------------------------------------------------------------------------
# Figure 4: training objective
# ----------------------------------------------------------------------------
MATCH_PRED = [(55, 64, 153, 196), (204, 88, 301, 226), (222, 70, 326, 236),
              (360, 118, 446, 216), (80, 180, 160, 292)]
MATCH_GT = [(50, 60, 150, 200), (200, 80, 300, 220), (350, 100, 430, 210),
            (470, 40, 522, 92)]


def fig_training(T, path, dpi):
    fig = plt.figure(figsize=(10.0, 3.95), facecolor=T['bg'])
    ty = 0.925  # common baseline of the panel titles

    # (a) corner offsets ------------------------------------------------------
    ax = fig.add_axes([0.01, 0.10, 0.30, 0.76])
    ax.set_xlim(0, 100)
    ax.set_ylim(84, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    b = (16, 20, 64, 56)
    g = (28, 9, 84, 65)
    r = (26, 11, 81, 63)
    ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fc='none', ec=T['ink2'],
                           lw=1.6, zorder=2))
    ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fc='none', ec=T['aqua'],
                           lw=1.8, ls=(0, (3, 2)), zorder=2))
    ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc='none',
                           ec=T['orange'], lw=2.0, zorder=3))

    def arr(p, q):
        ax.annotate('', xy=q, xytext=p,
                    arrowprops=dict(arrowstyle='-|>', color=T['ink'], lw=1.1,
                                    mutation_scale=8, shrinkA=0, shrinkB=0), zorder=6)
    # top-left corner: (b0,b1) -> (r0,r1)
    arr((b[0], b[1]), (r[0], b[1]))
    arr((r[0], b[1]), (r[0], r[1]))
    ax.text((b[0] + r[0]) / 2, b[1] + 3.4, r'$dx_1$', fontsize=8.6, ha='center', va='center',
            color=T['ink'])
    ax.text(r[0] - 1.4, (b[1] + r[1]) / 2, r'$dy_1$', fontsize=8.6, ha='right', va='center',
            color=T['ink'])
    # bottom-right corner
    arr((b[2], b[3]), (r[2], b[3]))
    arr((r[2], b[3]), (r[2], r[3]))
    ax.text((b[2] + r[2]) / 2, b[3] - 3.0, r'$dx_2$', fontsize=8.6, ha='center',
            va='center', color=T['ink'])
    ax.text(g[2] + 1.4, (b[3] + r[3]) / 2, r'$dy_2$', fontsize=8.6, ha='left', va='center',
            color=T['ink'])
    for (x, y) in [(b[0], b[1]), (b[2], b[3])]:
        ax.add_patch(Circle((x, y), 1.1, fc=T['ink2'], ec='none', zorder=4))
    items = [(T['ink2'], '-', 'B   YOLO box'), (T['orange'], '-', 'B′ = B + Δ   refined'),
             (T['aqua'], (0, (3, 2)), 'G   ground truth')]
    for k, (col, ls, txt) in enumerate(items):
        yy = 71.5 + k * 5.2
        ax.plot([6, 15], [yy, yy], color=col, lw=1.9, ls=ls)
        ax.text(17.5, yy, txt, fontsize=7.7, color=T['ink2'], ha='left', va='center')
    fig.text(0.015, ty, '(a) Corner offsets Δ, in pixels', fontsize=9.6,
             fontweight='semibold', color=T['ink'], ha='left', va='bottom')

    # (b) greedy matching on an IoU matrix ----------------------------------
    ax = fig.add_axes([0.385, 0.27, 0.24, 0.56])
    m, matches = greedy_match(MATCH_PRED, MATCH_GT)
    cmap = mcolors.LinearSegmentedColormap.from_list('seq', T['seq'])
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    P, G = m.shape
    for i in range(P):
        for j in range(G):
            v = m[i, j]
            rgb = cmap(v)[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            col = '#ffffff' if lum < 0.45 else '#0b0b0b'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.6, color=col)
    for k, (i, j) in enumerate(matches, 1):
        ax.add_patch(Rectangle((j - 0.5 + 0.07, i - 0.5 + 0.07), 0.86, 0.86, fc='none',
                               ec=T['orange'], lw=2.2, zorder=5))
        ax.text(j + 0.43, i - 0.40, str(k), ha='right', va='top', fontsize=6.6,
                color=T['ink'], fontweight='bold', zorder=6,
                bbox=dict(boxstyle='circle,pad=0.14', fc=T['bg'], ec=T['orange'], lw=1.0))
    ax.set_xticks(range(G), [f'G{j + 1}' for j in range(G)])
    ax.set_yticks(range(P), [f'B′{i + 1}' for i in range(P)])
    ax.tick_params(colors=T['ink2'], labelsize=7.8, length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, G, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, P, 1), minor=True)
    ax.grid(which='minor', color=T['bg'], lw=2)
    ax.tick_params(which='minor', length=0)
    ax.xaxis.tick_top()
    matched_p = {i for i, _ in matches}
    matched_g = {j for _, j in matches}
    un_p = ', '.join(f'B′{i + 1}' for i in range(P) if i not in matched_p)
    un_g = ', '.join(f'G{j + 1}' for j in range(G) if j not in matched_g)
    fig.text(0.345, 0.205, f'Picks ①②③ in descending IoU; each row and column is used once;\n'
             f'stop below IoU {MATCH_IOU}. Unmatched {un_p}: no loss.  {un_g}: no partner.',
             fontsize=7.3, color=T['ink2'], ha='left', va='top', linespacing=1.45)
    fig.text(0.345, ty, '(b) Greedy one-to-one matching', fontsize=9.6,
             fontweight='semibold', color=T['ink'], ha='left', va='bottom')

    # (c) Smooth-L1 ------------------------------------------------------------
    ax = fig.add_axes([0.725, 0.20, 0.26, 0.63])
    style_axes(ax, T, grid_axis='y')
    x = np.linspace(-3, 3, 601)
    beta = 1.0
    sl1 = np.where(np.abs(x) < beta, 0.5 * x ** 2 / beta, np.abs(x) - 0.5 * beta)
    ax.axvspan(-beta, beta, color=T['blue'], alpha=0.10, lw=0)
    ax.plot(x, 0.5 * x ** 2, color=T['aqua'], lw=2, ls=(0, (4, 2.4)), label='½ r²')
    ax.plot(x, np.abs(x), color=T['orange'], lw=2, ls=(0, (1.2, 2.0)),
            dash_capstyle='round', label='|r|')
    ax.plot(x, sl1, color=T['blue'], lw=2.4, label='Smooth-L1, β = 1 px')
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, 3.0)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel('corner residual r = B′ − G  (px)', fontsize=8)
    ax.set_ylabel('loss', fontsize=8)
    ax.text(0, 1.75, 'quadratic\nonly for\n|r| < 1 px', fontsize=7.2, color=T['ink2'],
            ha='center', va='center')
    ax.text(2.95, 1.52, 'Smooth-L1', fontsize=7.8, color=T['ink'], ha='right', va='top')
    leg = ax.legend(loc='upper center', fontsize=7.2, frameon=False, handlelength=2.2,
                    borderaxespad=0.0, bbox_to_anchor=(0.5, 1.02))
    for txt in leg.get_texts():
        txt.set_color(T['ink2'])
    fig.text(0.70, ty, '(c) Loss on the matched corners', fontsize=9.6,
             fontweight='semibold', color=T['ink'], ha='left', va='bottom')

    fig.text(0.015, 0.015, 'Synthetic boxes. IoUs and the matching order are computed '
             'with the package rules.', fontsize=7.4, color=T['muted'], ha='left',
             va='bottom')
    fig.savefig(path, dpi=dpi, facecolor=T['bg'])
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: ROS 2 workflow, modes and artefacts
# ----------------------------------------------------------------------------
def fig_workflow(T, path, dpi):
    B = Board(T, 150, 68, 10.0)
    ax = B.ax
    sf = 7.6
    # entry points ------------------------------------------------------------
    l1 = B.block(9.0, 58.5, 14.0, 11.0, 'ros2 launch', 'launch file,\n8 arguments',
                 tfs=9.0, sfs=sf)
    l2 = B.block(24.0, 58.5, 14.0, 11.0, 'ros2 run', 'any CLI\nflags', tfs=9.0, sfs=sf)
    cs = B.block(16.5, 26.5, 29.0, 25.0, 'main()',
                 'console-script entry point\n'
                 'drop the --ros-args block\nparse CLI flags\n'
                 'resolve --model weights\nwrite config.json', tfs=9.4, sfs=sf)
    B.arrow([(l1['cx'], l1['b']), (l1['cx'], cs['t'])])
    B.label(l1['cx'], 46.2, 'adds\n--ros-args', fs=7.0, ha='center')
    B.arrow([(l2['cx'], l2['b']), (l2['cx'], cs['t'])])

    # lanes -------------------------------------------------------------------
    ax.plot([33.0, 33.0], [2.5, 66.0], color=T['line'], lw=0.8)
    ax.plot([34.5, 148.5], [29.0, 29.0], color=T['line'], lw=0.8)
    B.label(36.0, 65.0, 'mode = train  or  train_detect', fs=7.6, ha='left',
            color=T['muted'], bg=False, weight='semibold')
    B.label(148.5, 26.6, 'mode = detect  or  train_detect', fs=7.6, ha='right',
            color=T['muted'], bg=False, weight='semibold')
    yt, yd = 47.0, 14.0
    s1 = B.block(46.5, yt, 20.0, 12.5, 'Split train_dir',
                 'shuffle (--seed)\n--val_split 0.05', tfs=9.2, sfs=sf)
    s2 = B.block(72.0, yt, 23.0, 12.5, 'YOLO pass',
                 'top --cap_nodes boxes\nper image, cached', kind='frozen', tfs=9.2, sfs=sf)
    s3 = B.block(99.5, yt, 24.0, 12.5, 'Epoch loop',
                 'train_one_epoch, then\nevaluate: val loss, IoU gain', kind='train',
                 tfs=9.2, sfs=sf)
    s4 = B.block(132.0, yt, 25.0, 12.5, 'Keep the best',
                 'lowest validation loss\nsaved as GNN checkpoint', tfs=9.2, sfs=sf)
    d1 = B.block(46.5, yd, 20.0, 12.5, 'Load checkpoint',
                 '--gnn_ckpt, default\nout_dir/best_weights.pt', tfs=9.2, sfs=sf)
    d2 = B.block(78.0, yd, 31.0, 12.5, 'For every test image',
                 'YOLO → top-N → graph → refiner\nB′ = B + Δ, clipped to the image',
                 tfs=9.2, sfs=sf)
    d3 = B.block(114.5, yd, 24.0, 12.5, 'Draw and save',
                 'class name: YOLO score\nscore ≥ --score_thr_draw', tfs=9.2, sfs=sf)
    B.arrow([(cs['r'], 34.0), (35.0, 34.0), (35.0, yt), (s1['l'], yt)])
    B.arrow([(cs['r'], yd), (d1['l'], yd)])
    for a, b2 in [(s1, s2), (s2, s3), (s3, s4), (d1, d2), (d2, d3)]:
        B.arrow([(a['r'], a['cy']), (b2['l'], b2['cy'])])
    B.arrow([(s4['cx'], s4['b']), (s4['cx'], 32.0), (d1['cx'], 32.0), (d1['cx'], d1['t'])],
            dashed=True)
    B.label(97.0, 32.0, 'train_detect: the new checkpoint is used for detection', fs=7.2)

    # artefacts ----------------------------------------------------------------
    fy = 56.6
    B.file_icon(61.0, fy, 23.0, 7.8, 'cache/', 'yolo_{train,val}_<tag>.pkl', fs=7.4)
    B.file_icon(120.5, fy, 23.0, 7.8, 'best_weights.pt', 'GNN state_dict', fs=7.4)
    B.file_icon(127.0, 1.4, 21.5, 7.2, 'detections_refined/', 'annotated images', fs=7.4)
    B.file_icon(5.5, 1.4, 22.0, 7.2, 'config.json', 'arguments + weights path', fs=7.4)
    dash = dict(color=T['ink2'], lw=0.9, ls=(0, (2, 2)), zorder=1)
    ax.plot([s2['cx'], s2['cx']], [s2['t'], fy], **dash)
    ax.plot([s4['cx'], s4['cx']], [s4['t'], fy], **dash)
    ax.plot([d3['r'], 137.5, 137.5], [d3['cy'], d3['cy'], 8.6], **dash)
    ax.plot([cs['cx'], cs['cx']], [cs['b'], 8.6], **dash)
    B.label(103.0, 66.2, 'every artefact is written below --out_dir', fs=7.2,
            color=T['muted'], bg=False)
    B.save(path, dpi)


FIGURES = [
    ('fig1_overview', fig_overview),
    ('fig2_graph_construction', fig_graph),
    ('fig3_refiner_architecture', fig_refiner),
    ('fig4_training_objective', fig_training),
    ('fig5_ros2_workflow', fig_workflow),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', default=os.path.join(here, 'figures'))
    ap.add_argument('--dpi', type=int, default=200)
    ap.add_argument('--only', default='', help='comma-separated figure names')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup_fonts()
    only = {s for s in args.only.split(',') if s}
    for name, fn in FIGURES:
        if only and name not in only:
            continue
        for theme, T in THEMES.items():
            path = os.path.join(args.out, f'{name}_{theme}.png')
            fn(T, path, args.dpi)
            print('wrote', path)


if __name__ == '__main__':
    main()
