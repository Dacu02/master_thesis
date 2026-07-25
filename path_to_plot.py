from typing import Optional
from math import comb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import os


# ============================================================
# Utils
# ============================================================
Path = list[tuple[float, float, float, float]|tuple[float, float, float]]
_POLY  = dict(linestyle='--', linewidth=1.0, color='gray', alpha=0.55)
_ANC2D = dict(marker='o', s=90, zorder=6, edgecolors='black', linewidths=0.8)
_HDL2D = dict(marker='D', s=55, zorder=5, edgecolors='gray',  linewidths=0.7)

# Dominio normalizzato: tutti gli assi vanno da 0 a 1, cosi' i plot di path
# diversi restano confrontabili tra loro. NORM_MARGIN e' solo un piccolo
# respiro visivo per non tagliare a meta' i marker che cadono esattamente
# sul bordo (es. punto iniziale in (0,0)); non altera il dominio dei dati.
NORM_MIN, NORM_MAX = 0, 1
NORM_MARGIN = 0.05

CMAP = LinearSegmentedColormap.from_list(
    'red_black_blue', [(1, 0, 0), (0, 0, 0), (0, 0, 1)], N=256
)
Z_NORM = plt.Normalize(vmin=NORM_MIN, vmax=NORM_MAX)


def bezier_curve(ctrl_pts: list[np.ndarray], weights: Optional[np.ndarray] = None,
                  num_points: int = 400) -> np.ndarray:
    """
    Valuta la curva di Bézier (razionale) di grado len(ctrl_pts)-1
    usando i polinomi di Bernstein.
    """
    pts = np.array(ctrl_pts, dtype=float)   # (n+1, 3)
    n = len(pts) - 1
    if weights is None:
        weights = np.ones(n + 1)
    else:
        weights = np.asarray(weights, dtype=float)

    t = np.linspace(0.0, 1.0, num_points)
    numerator = np.zeros((num_points, 3))
    denominator = np.zeros(num_points)
    for i, p in enumerate(pts):
        B = comb(n, i) * (t ** i) * ((1.0 - t) ** (n - i)) * weights[i]
        numerator += B[:, None] * p
        denominator += B
    return numerator / denominator[:, None]


def draw_ctrl_numbered(ax, proj: np.ndarray, colors: np.ndarray) -> None:
    """Poligono di controllo + punti colorati in base a Z, con numero di sequenza accanto a ciascuno."""
    ax.plot(proj[:, 0], proj[:, 1], **_POLY)
    ax.scatter(proj[[0, -1], 0], proj[[0, -1], 1], color=colors[[0, -1]], **_ANC2D)
    if len(proj) > 2:
        ax.scatter(proj[1:-1, 0], proj[1:-1, 1], color=colors[1:-1], **_HDL2D)

    # Numero di sequenza: 1 = punto iniziale, ..., n = punto finale
    for i, (x, y) in enumerate(proj, start=1):
        ax.annotate(
            str(i), (x, y), textcoords="offset points", xytext=(7, 7),
            fontsize=9, fontweight='bold', color='black',
            bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='none', alpha=0.8),
            zorder=7,
        )

def specularize_path(path: Path) -> Path:
    """
        Adopts a specularization of the path across the xy plane, effectively mirroring the path for left-handed users
    """
    resulting_path: Path = []
    for i in range(len(path)):
        if len(path[i]) == 4:
            resulting_path.append((1 - path[i][0], path[i][1], path[i][2], path[i][3])) # type: ignore
        else:
            resulting_path.append((1 - path[i][0], path[i][1], path[i][2]))
    return resulting_path

def counter_clock_wise_rotation(path: Path) -> Path:
    """
        Rotates the path counter-clockwise by 90 degrees across the xy plane
    """
    resulting_path: Path = []
    for i in range(len(path)):
        if len(path[i]) == 4:
            resulting_path.append((1 - path[i][1], path[i][0], path[i][2], path[i][3])) # type: ignore
        else:
            resulting_path.append((1 - path[i][1], path[i][0], path[i][2]))
    return resulting_path

def invert_order(path: Path) -> Path:
    """
        Inverts the order of the path
    """
    resulting_path:Path = []
    resulting_path.append((path[len(path)-1][0], path[len(path)-1][1], path[len(path)-1][2]))
    for i in range(len(path)-2, -1, -1):
        if len(path[i]) == 4:
            resulting_path.append((path[i][0], path[i][1], path[i][2], path[i][3])) # type: ignore
        else:
            resulting_path.append((path[i][0], path[i][1], path[i][2]))
    return resulting_path

# ============================================================
# Input
# ============================================================

SIMPLE_CURVE:Path = [
    (0.25, 0, 0),
    (0.75, 0.25, .5, 5),
    (1, 1, .5, 1)
]

DEEP_SIMPLE_CURVE:Path = [
    (0.5, 0, 0),
    (0.75, 0.125, .5, 5),
    (0.75, .25, 1, 1)
]

INWARD_SIMPLE_CURVE:Path = [
    (0.5, 0, 0),
    (0.35, 0.33, 0.0, 5),
    (0.75, 0.66, 0.5, 5),
    (0.75, 1, 0.5, 1)
]

SPIRAL_CURVE:Path = [
    (0.2, 0.2, 0.5),
    (0.8, 0.4, 0.5, 35),
    (0.5, .8, 0.5, 45),
    (0.25, .55, 0.5, 15),
    (0.55, .4, 0.5, 1)
]

CIRCLE:Path = [
    (0.425, 0.2, 0.5),
    (0.7, 0.5, 0.5, 80),
    (0.4, 0.8, 0.5, 250),
    (0.1, 0.5, 0.5, 80),
    (0.375, 0.2, 0.5),
]

L:Path = [
    (0.3, 0.2, 0.5),
    (0.525, 0.4, 0.5, 100),
    (0.7, 0.8, 0.5, 1000),
    (0.5, 1, 0.5, 1000),
    (0.3, 0.8, 0.5, 1000),
    (0.475, 0.4, 0.5, 100),
    (0.7, 0.2, 0.5),
]

L_REDUCED:Path = [
    (0.3, 0.2, 0.5),
    (0.7, 0.8, 0.5, 50),
    (0.5, 1, 0.5, 100),
    (0.3, 0.8, 0.5, 50),
    (0.7, 0.2, 0.5),
]

N:Path = [
    (0.2, 0.2, 0.2),
    (0.3, 0.6, 0.6), 
    (0.5, 0.2, 0.6),
    (0.6, 0.6, 0.2),
]


PATHS: dict[str, Path] = {

    'Simple_Curve' : SIMPLE_CURVE,
    'Simple_Curve_I': invert_order(SIMPLE_CURVE),
    'Simple_Curve_CC': counter_clock_wise_rotation(SIMPLE_CURVE),
    'Simple_Curve_DCC': counter_clock_wise_rotation(counter_clock_wise_rotation(SIMPLE_CURVE)),
    'Simple_Curve_S': specularize_path(SIMPLE_CURVE),
    'Deep_Simple_Curve': DEEP_SIMPLE_CURVE,
    'Inward_Simple_Curve': INWARD_SIMPLE_CURVE,
    'Spiral_Curve': specularize_path(SPIRAL_CURVE), # Mirrored for right-handed users
    'Circle': CIRCLE,
    'L': L,
    'L_Reduced': L_REDUCED,
    'N': N,
    'S': [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1)
    ]
}

CONTROL_RATIO = 100
CONTROL_WEIGHT = 50
END_CONTROL_WEIGHT = CONTROL_WEIGHT / CONTROL_RATIO

OUTPUT_FOLDER = os.path.join(os.getcwd(), 'images', 'paths')

if not os.path.exists(os.path.join(os.getcwd(), 'images')):
    raise ValueError('Path not found')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for name, path in PATHS.items():

    initial_points = [
        np.array([p for p in path[0]]),
    ]

    control_points = []
    control_point_weights = []
    for p in path[1:]:
        control_points.append(np.array([p for p in p[:3]]))
        if len(p) == 4:
            control_point_weights.append(p[3])
        else:
            control_point_weights.append(None)

    control_point_weights[-1] = END_CONTROL_WEIGHT

    resolved_control_weights = [
        CONTROL_WEIGHT if w is None else w
        for w in control_point_weights
    ]

    bezier_points = initial_points + control_points
    point_weights = np.array([1.0] * len(initial_points) + resolved_control_weights)

    degree = len(bezier_points) - 1
    print(f"Bézier: grado {degree}  ({len(initial_points)} punti iniziali + {len(control_points)} di controllo)")
    print(f"Pesi effettivi dei control_points: {resolved_control_weights}  "
          f"(peso comune di base = {CONTROL_WEIGHT}, dove non sovrascritto)")

    curve_pts = bezier_curve(bezier_points, point_weights)
    ctrl_arr  = np.array(bezier_points, dtype=float)

    # Colore = quota Z (non piu' il "tempo"/ordine): stesso Z_NORM (0-1) per
    # tutti i path, cosi' anche il colore resta confrontabile tra plot diversi.
    seg_colors  = CMAP(Z_NORM(curve_pts[:-1, 2]))
    ctrl_colors = CMAP(Z_NORM(ctrl_arr[:, 2]))

    c_xy = curve_pts[:, [0, 1]]
    k_xy = ctrl_arr[:, [0, 1]]
    segs_xy = np.stack([c_xy[:-1], c_xy[1:]], axis=1)

    xlim = (NORM_MIN - NORM_MARGIN, NORM_MAX + NORM_MARGIN)
    ylim = (NORM_MIN - NORM_MARGIN, NORM_MAX + NORM_MARGIN)

    # ---------- Front view (XY), unico plot ----------
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(LineCollection(segs_xy, colors=seg_colors, linewidths=2))
    draw_ctrl_numbered(ax, k_xy, ctrl_colors)
    ax.set(xlabel='X', ylabel='Y', title=f'{name}')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.45)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=Z_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('Z')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)