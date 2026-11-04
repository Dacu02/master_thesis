import math
from yaml import load, dump
from typing import Optional
from math import comb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra il proiettore '3d')
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
import os

PATHS_LENGTH: float = 0.9
PATH_CENTER:  float = 0.5
# ============================================================
# Utils
# ============================================================
Path = list[tuple[float, float, float, float]|tuple[float, float, float]]
_POLY  = dict(linestyle='--', linewidth=1.0, color='gray', alpha=0.55)
_ANC   = dict(marker='o', s=90, zorder=6, edgecolors='black', linewidths=0.8)
_HDL   = dict(marker='D', s=55, zorder=5, edgecolors='gray',  linewidths=0.7)

# Dominio normalizzato: tutti gli assi (X, Y, Z) vanno da 0 a 1, cosi' i plot
# di path diversi restano confrontabili tra loro. NORM_MARGIN e' solo un
# piccolo respiro visivo per non tagliare a meta' i marker che cadono
# esattamente sul bordo (es. punto iniziale in (0,0,0)); non altera il
# dominio dei dati.
NORM_MIN, NORM_MAX = 0, 1
NORM_MARGIN = 0.05

CMAP = LinearSegmentedColormap.from_list(
    'red_black_blue', [(1, 0, 0), (0, 0, 0), (0, 0, 1)], N=256
)
# Il colore ora rappresenta l'ORDINE lungo il percorso (0 = waypoint 1 /
# rosso, 1 = waypoint N / blu), non piu' la quota Z.
ORDER_NORM = plt.Normalize(vmin=0.0, vmax=1.0) # type: ignore

# ------------------------------------------------------------
# Angolo di vista del plot 3D: yaw (azim), pitch (elev), roll.
# Valori consigliati (vista quasi-isometrica, mostra bene curva + le tre
# proiezioni sulle pareti): elev=28, azim=-55, roll=0.
# Alternative utili:
#   - vista "dall'alto" (per leggere bene la forma su XY, es. lettere):
#     elev=65, azim=-90, roll=0
#   - vista laterale (per leggere bene il profilo su Z):
#     elev=8,  azim=-90, roll=0
# ------------------------------------------------------------
VIEW_YAW, VIEW_PITCH, VIEW_ROLL = -55, 28, 0

# Se True, scambia il RUOLO degli assi Y e Z: cio' che prima compariva
# sull'asse verticale Z viene disegnato sull'asse orizzontale "Y" e
# viceversa (non e' un semplice specchio/inversione di direzione, e' uno
# scambio di colonne dei dati + delle etichette).
SWAP_Y_Z = True

# Inverti la direzione di un asse (mantiene il ruolo dell'asse, ma lo fa
# crescere nel verso opposto). Si applica DOPO l'eventuale SWAP_Y_Z, quindi
# si riferisce sempre agli assi come appaiono nel plot finale.
INVERT_Y_AXIS = False
INVERT_Z_AXIS = False


def maybe_swap_yz(arr: np.ndarray) -> np.ndarray:
    """Se SWAP_Y_Z e' attivo, scambia le colonne Y (indice 1) e Z (indice 2)
    di un array di punti (N, 3). Usata solo ai fini del plot, non tocca il
    calcolo della curva di Bezier ne' i colori (basati sull'ordine)."""
    if not SWAP_Y_Z:
        return arr
    swapped = arr.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    return swapped


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


def draw_ctrl_numbered_3d(ax, arr: np.ndarray, colors: np.ndarray,
                           label_offset: float = 0.02) -> None:
    """Poligono di controllo + punti colorati in base all'ordine, con numero
    di sequenza accanto a ciascuno (versione 3D)."""
    ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], **_POLY)
    ax.scatter(arr[[0, -1], 0], arr[[0, -1], 1], arr[[0, -1], 2],
               color=colors[[0, -1]], **_ANC)
    if len(arr) > 2:
        ax.scatter(arr[1:-1, 0], arr[1:-1, 1], arr[1:-1, 2],
                   color=colors[1:-1], **_HDL)

    # Numero di sequenza: 1 = punto iniziale, ..., n = punto finale
    for i, (x, y, z) in enumerate(arr, start=1):
        ax.text(
            x + label_offset, y + label_offset, z + label_offset, str(i),
            fontsize=9, fontweight='bold', color='black',
            bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='none', alpha=0.8),
            zorder=10,
        )


def add_plane_projections(ax, curve_pts: np.ndarray, seg_colors: np.ndarray,
                           ctrl_arr: np.ndarray, ctrl_colors: np.ndarray,
                           xlim: tuple[float, float], ylim: tuple[float, float],
                           zlim: tuple[float, float],
                           alpha: float = 0.35, linewidth: float = 1.5,
                           point_size: float = 22, degenerate_tol: float = 1e-9) -> None:
    """Disegna l'ombra 2D della traiettoria (e dei punti di controllo) su
    ciascuna delle tre pareti del box 3D (piano XY sul pavimento, XZ e YZ
    sulle pareti), oltre alla curva vera e propria nello spazio.

    Se su una parete la traiettoria collassa in un solo punto (es. un
    segmento dritto allineato con l'asse perpendicolare a quella parete,
    tipo un BURST1_X visto sul piano YZ), una Line3DCollection con segmenti a
    lunghezza zero non disegnerebbe nulla: in quel caso viene disegnato un
    marker al suo posto, cosi' il punto resta visibile.

    I punti di controllo (ancore = cerchio, handle = diamante) vengono
    proiettati sulle stesse tre pareti con lo stesso colore/alpha usati per
    l'ombra della curva.
    """
    x, y, z = curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2]
    cx, cy, cz = ctrl_arr[:, 0], ctrl_arr[:, 1], ctrl_arr[:, 2]

    shadow_colors = seg_colors.copy()
    shadow_colors[:, 3] = alpha  # riduce l'opacita' mantenendo il gradiente colore

    ctrl_shadow_colors = ctrl_colors.copy()
    ctrl_shadow_colors[:, 3] = alpha  # stessa opacita' usata per l'ombra della curva

    def draw_wall_line(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        pts = np.column_stack([xs, ys, zs])
        extent = pts.max(axis=0) - pts.min(axis=0)
        if np.all(extent < degenerate_tol):
            # Traiettoria degenere su questa parete -> un solo punto: uso un
            # marker (colore a meta' gradiente, piu' opaco) invece della linea.
            mid_color = seg_colors[len(seg_colors) // 2].copy()
            mid_color[3] = min(1.0, alpha * 2.5)
            ax.scatter([xs[0]], [ys[0]], [zs[0]], color=[mid_color],
                       s=point_size, zorder=2, edgecolors='none')
        else:
            segs = np.stack([
                np.column_stack([xs[:-1], ys[:-1], zs[:-1]]),
                np.column_stack([xs[1:],  ys[1:],  zs[1:]]),
            ], axis=1)
            ax.add_collection3d(Line3DCollection(segs, colors=shadow_colors, linewidths=linewidth, zorder=1))

    def draw_wall_points(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        # Ancore (primo e ultimo punto di controllo) -> cerchio
        anchors = [0, -1]
        ax.scatter(xs[anchors], ys[anchors], zs[anchors],
                   color=ctrl_shadow_colors[anchors], marker=_ANC['marker'],
                   s=_ANC['s'], zorder=1, edgecolors='none')
        # Handle (punti di controllo intermedi) -> diamante
        if len(xs) > 2:
            ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1],
                       color=ctrl_shadow_colors[1:-1], marker=_HDL['marker'],
                       s=_HDL['s'], zorder=1, edgecolors='none')

    n, nc = len(x), len(cx)

    # XY -> pavimento (z = z_min)
    draw_wall_line(x, y, np.full(n, zlim[0]))
    draw_wall_points(cx, cy, np.full(nc, zlim[0]))

    # XZ -> parete di fondo (y = y_max)
    draw_wall_line(x, np.full(n, ylim[1]), z)
    draw_wall_points(cx, np.full(nc, ylim[1]), cz)

    # YZ -> parete laterale (x = x_min)
    draw_wall_line(np.full(n, xlim[0]), y, z)
    draw_wall_points(np.full(nc, xlim[0]), cy, cz)


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
        Rotates the path counter-clockwise by 90 degrees across the Z axis
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
    resulting_path: Path = []
    resulting_path.append((path[len(path)-1][0], path[len(path)-1][1], path[len(path)-1][2]))
    for i in range(len(path)-2, -1, -1):
        if len(path[i]) == 4:
            resulting_path.append((path[i][0], path[i][1], path[i][2], path[i][3])) # type: ignore
        else:
            resulting_path.append((path[i][0], path[i][1], path[i][2]))
    return resulting_path


import numpy as np
from typing import List, Tuple

# ── Costanti del cubo di normalizzazione ─────────────────────────────
MINIMUM_X = MINIMUM_Y = -0.063997
MINIMUM_Z = -0.067524
MAXIMUM_X = MAXIMUM_Y = -MINIMUM_X       # 0.063997
MAXIMUM_Z = 0.06047

CUBE_MIN = np.array([MINIMUM_X, MINIMUM_Y, MINIMUM_Z])
CUBE_MAX = np.array([MAXIMUM_X, MAXIMUM_Y, MAXIMUM_Z])
CUBE_SPAN = CUBE_MAX - CUBE_MIN          # [0.127994, 0.127994, 0.127994]

# Centro del cubo: (0, 0, -0.003527)
CUBE_CENTER = (
    0.0,
    0.0,
    (MAXIMUM_Z - MINIMUM_Z) / 2.0 + MINIMUM_Z
)


def normalize_path(
    base_path: Path,
    size: float = 1.0,
    center: tuple[float, float, float] | None = None,
    uniform: bool = True
) -> Path:
    """
    Centra e scala un path all'interno del cubo definito dalle costanti globali.

    - size: fattore di riempimento del cubo (1.0 = tocca i bordi,
      0.5 = occupa metà dello spazio disponibile).
    - center: dove piazzare il centro del bounding box.
      Se None, usa CUBE_CENTER.
    - uniform: se True, usa la STESSA scala su X/Y/Z (preserva le
      proporzioni, inscrive il path nel cubo). Se False, scala ogni
      asse indipendentemente per riempire esattamente il cubo.
    I pesi (4° elemento delle tuple) non vengono toccati.
    """
    if center is None:
        center = CUBE_CENTER

    arr = np.array([p[:3] for p in base_path], dtype=float)   # (n,3)
    mins, maxs = arr.min(axis=0), arr.max(axis=0)
    bbox_center = (mins + maxs) / 2.0
    spans = maxs - mins
    spans[spans == 0] = 1.0                                   # evita div/0

    # Spazio target = cubo × fattore di riempimento
    target = CUBE_SPAN * size

    if uniform:
        # Preserva proporzioni: la scala è limitata dall'asse più "stretto"
        scale_factor = np.min(target / spans)
        scale = np.full(3, scale_factor)
    else:
        # Riempi il cubo su ogni asse indipendentemente
        scale = target / spans

    center_arr = np.array(center, dtype=float)
    result: Path = []

    for p in base_path:
        xyz = (np.array(p[:3], dtype=float) - bbox_center) * scale + center_arr
        if len(p) == 4:
            result.append((float(xyz[0]), float(xyz[1]), float(xyz[2]), p[3]))
        else:
            result.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

    return result
# ============================================================
# Input
# ============================================================

SIMPLE_CURVE: Path = [
    (0.25, 0, 0),
    (0.75, 0.25, .5, 5),
    (1, 1, .5, 1)
]

DEEP_SIMPLE_CURVE: Path = [
    (0.5, 0, 0),
    (0.75, 0.125, .5, 5),
    (0.75, .25, 1, 1)
]

INWARD_SIMPLE_CURVE: Path = [
    (0.5, 0, 0),
    (0.35, 0.33, 0.0, 5),
    (0.75, 0.66, 0.5, 5),
    (0.75, 1, 0.5, 1)
]

SPIRAL_CURVE: Path = [
    (0.2, 0.2, 0.5),
    (0.8, 0.4, 0.5, 35),
    (0.5, .8, 0.5, 45),
    (0.25, .55, 0.5, 15),
    (0.55, .4, 0.5, 1)
]

CIRCLE: Path = [
    (0.425, 0.2, 0.5),
    (0.7, 0.5, 0.5, 80),
    (0.4, 0.8, 0.5, 250),
    (0.1, 0.5, 0.5, 80),
    (0.375, 0.2, 0.5),
]

L: Path = [
    (0.3, 0.2, 0.5),
    (0.525, 0.4, 0.5, 100),
    (0.7, 0.8, 0.5, 1000),
    (0.5, 1, 0.5, 1000),
    (0.3, 0.8, 0.5, 1000),
    (0.475, 0.4, 0.5, 100),
    (0.7, 0.2, 0.5),
]

L_REDUCED: Path = [
    (0.3, 0.2, 0.5),
    (0.7, 0.8, 0.5, 50),
    (0.5, 1, 0.5, 100),
    (0.3, 0.8, 0.5, 50),
    (0.7, 0.2, 0.5),
]

N: Path = [
    (0.2, 0.2, 0.2),
    (0.3, 0.6, 0.6), 
    (0.5, 0.2, 0.6),
    (0.6, 0.6, 0.2),
]


# PATHS: dict[str, Path] = {

#     'Simple_Curve' : SIMPLE_CURVE,
#     # 'Simple_Curve_I': invert_order(SIMPLE_CURVE),
#     # 'Simple_Curve_CC': counter_clock_wise_rotation(SIMPLE_CURVE),
#     # 'Simple_Curve_DCC': counter_clock_wise_rotation(counter_clock_wise_rotation(SIMPLE_CURVE)),
#     'Simple_Curve_S': specularize_path(SIMPLE_CURVE),
#     # 'Deep_Simple_Curve': DEEP_SIMPLE_CURVE,
#     # 'Inward_Simple_Curve': INWARD_SIMPLE_CURVE,
#     'Spiral_Curve': specularize_path(SPIRAL_CURVE), # Mirrored for right-handed users
#     'Circle': CIRCLE,
#     'L': L,
#     'L_Reduced': L_REDUCED,
#     'N': N,
#     'S': [
#         (0, 0, 0),
#         (1, 0, 1),
#         (0, 1, 1),
#         (1, 1, 1)
#     ]
# }

PATHS_LENGTH: float = 0.8
PATH_CENTER:  float = 0.5
#============================================================
# BURST: Short movements
#============================================================
BURST_LENGTH: float = PATHS_LENGTH
CURVE_RADIUS: float = 0.9
CURVE_WEIGHT: float = 2
BURST: dict[str, Path] = {
    'BURST1_XY': [
        (1 - BURST_LENGTH, 1 - BURST_LENGTH, PATH_CENTER),
        (BURST_LENGTH, BURST_LENGTH, PATH_CENTER)
    ],
    'BURST2_XZ': [
        (1 - BURST_LENGTH, PATH_CENTER, 1 - BURST_LENGTH),
        (BURST_LENGTH, PATH_CENTER, BURST_LENGTH)
    ],
    'BURST3_YZ': [
        (PATH_CENTER, 1 - BURST_LENGTH, 1 - BURST_LENGTH),
        (PATH_CENTER, BURST_LENGTH, BURST_LENGTH)
    ],
    'BURST4_XY': [
        (1 - CURVE_RADIUS, 1 - BURST_LENGTH, PATH_CENTER),
        (CURVE_RADIUS, 0.5, PATH_CENTER, CURVE_WEIGHT),
        (1- CURVE_RADIUS, BURST_LENGTH, PATH_CENTER)
    ],
    'BURST5_XZ': [
        (1 - BURST_LENGTH, PATH_CENTER, 1 - CURVE_RADIUS),
        (PATH_CENTER, PATH_CENTER, CURVE_RADIUS, CURVE_WEIGHT),
        (BURST_LENGTH, PATH_CENTER, 1 - CURVE_RADIUS)
    ],
    'BURST6_YZ': [
        (PATH_CENTER, 1 - CURVE_RADIUS, 1 - BURST_LENGTH),
        (PATH_CENTER, CURVE_RADIUS, PATH_CENTER, CURVE_WEIGHT),
        (PATH_CENTER, 1 - CURVE_RADIUS, BURST_LENGTH)
    ],
}

# ============================================================
# CURVE: Semicircumference with diameter BURST_LENGTH
# ============================================================


# CURVE: dict[str, Path] = {
#     'CURVE1_XY': CURVE_XY,
#     'CURVE2_XZ': CURVE_XZ,
#     'CURVE3_YZ': CURVE_YZ,
#     'CURVE4_SPIRALXY': SPIRAL_CURVE,
# }


# ============================================================
# LETTER: Letters in 3D space
# ============================================================
LETTER_SIZE = 0.8
L: Path = [
    (0.3, 0.2, 0.5),
    (0.7, 0.8, 0.5, 50),
    (0.5, 1, 0.5, 100),
    (0.3, 0.8, 0.5, 50),
    (0.7, 0.2, 0.5),
]

N: Path = [
    (0.2, 0.2, 0.2),
    (0.3, 0.6, 0.6), 
    (0.5, 0.2, 0.6),
    (0.6, 0.6, 0.2),
]

S: Path = [
    (1 - LETTER_SIZE, 1 - LETTER_SIZE, 1 - LETTER_SIZE),
    (LETTER_SIZE, 1 - LETTER_SIZE, 1 - LETTER_SIZE),
    (1 - LETTER_SIZE, LETTER_SIZE, LETTER_SIZE),
    (LETTER_SIZE, LETTER_SIZE, LETTER_SIZE)
]

PHI_WIDTH = 0.25
PHI: Path = [
    (PATH_CENTER - PHI_WIDTH, PATH_CENTER, PATH_CENTER),
    (PATH_CENTER, PATH_CENTER - PHI_WIDTH, PATH_CENTER, 30),
    (PATH_CENTER + PHI_WIDTH, PATH_CENTER, PATH_CENTER, 250),
    (PATH_CENTER, PATH_CENTER + PHI_WIDTH, PATH_CENTER, 500),
    (PATH_CENTER, PATH_CENTER, PATH_CENTER, 1000),
    (PATH_CENTER, PATH_CENTER, PATH_CENTER - PHI_WIDTH, 200),
]

LETTER: dict[str, Path] = {
    'LETTER1_L': L,
    'LETTER2_PHI': PHI,
    'LETTER3_S': S,
    'LETTER4_N': N,
}
# ============================================================
# GEOMETRIC: Geometric shapes
# ============================================================
SPIRAL_LENGTH = .8
SPIRAL_WIDTH = 0.3
SPIRAL_WAYPOINTS = 5



initial_angle = -math.pi
SPIRAL_Y: Path = []
for i in range(SPIRAL_WAYPOINTS):
    angle = initial_angle + (i / (SPIRAL_WAYPOINTS - 1)) * 2 * math.pi
    x = 0.5 + SPIRAL_WIDTH * math.cos(angle)
    y = (i + 1) / (SPIRAL_WAYPOINTS) * SPIRAL_LENGTH + (1 - SPIRAL_LENGTH) / 2
    z = 0.5 + SPIRAL_WIDTH * math.sin(angle)
    SPIRAL_Y.append((x, y, z))


# SPIRAL_CURVE: Path = [
#     (0.2, 0.2, PATH_CENTER),
#     (0.8, 0.4, PATH_CENTER, 35),
#     (0.5, .8, PATH_CENTER, 45),
#     (0.25, .55, PATH_CENTER, 15),
#     (0.55, .4, PATH_CENTER, 1)
# ]

GOLDEN_RATIO_SIZE = 0.5
GOLDEN_RATIO_START_X = 0.1
GOLDEN_RATIO_START_Y = 0.3

GOLDEN_RATIO_CORRECTION = 0.3
GOLDEN_RATIO = 1.61803398875 - GOLDEN_RATIO_CORRECTION

GOLDEN_CURVE: Path = [
    (GOLDEN_RATIO_START_X, GOLDEN_RATIO_START_Y, PATH_CENTER)
]
dx = GOLDEN_RATIO_SIZE
dy = GOLDEN_RATIO_SIZE
WAYPOINTS_GOLDEN = 5
for i in range(WAYPOINTS_GOLDEN):
    if i%4 == 0:
        x = GOLDEN_CURVE[-1][0] + dx
        y = GOLDEN_CURVE[-1][1] + dy
    elif i%4 == 1:
        x = GOLDEN_CURVE[-1][0] + dx
        y = GOLDEN_CURVE[-1][1] - dy
    elif i%4 == 2:
        x = GOLDEN_CURVE[-1][0] - dx
        y = GOLDEN_CURVE[-1][1] - dy
    else:
        x = GOLDEN_CURVE[-1][0] - dx
        y = GOLDEN_CURVE[-1][1] + dy

    z = PATH_CENTER
    weight = 100 if i < WAYPOINTS_GOLDEN - 1 else 1

    GOLDEN_CURVE.append((x, y, z, weight))

    dx /= GOLDEN_RATIO
    dy /= GOLDEN_RATIO

for point in GOLDEN_CURVE:
    for coordinate in point[:3]:
        if coordinate < 0 or coordinate > 1:
            raise ValueError(f"Coordinate {coordinate} out of bounds for point {point} in GOLDEN_CURVE")

OFFSET = 0.01
CIRCLE_RADIUS = 0.33

CIRCLE_XY = [
    (PATH_CENTER + OFFSET, PATH_CENTER - CIRCLE_RADIUS, PATH_CENTER),
    (PATH_CENTER + CIRCLE_RADIUS, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER, PATH_CENTER + CIRCLE_RADIUS, PATH_CENTER, 300),
    (PATH_CENTER - CIRCLE_RADIUS, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER - OFFSET, PATH_CENTER - CIRCLE_RADIUS, PATH_CENTER),
]

CIRCLE_ZX = [
    (PATH_CENTER + OFFSET, PATH_CENTER, PATH_CENTER - CIRCLE_RADIUS),
    (PATH_CENTER + CIRCLE_RADIUS, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER, PATH_CENTER, PATH_CENTER + CIRCLE_RADIUS, 300),
    (PATH_CENTER - CIRCLE_RADIUS, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER - OFFSET, PATH_CENTER, PATH_CENTER - CIRCLE_RADIUS),
]

ELLIPSE_XY_RADIUS_X = 0.4
ELLIPSE_XY_RADIUS_Y = 0.15
ELLIPSE_XY = [
    (PATH_CENTER + OFFSET, PATH_CENTER - ELLIPSE_XY_RADIUS_Y, PATH_CENTER),
    (PATH_CENTER + ELLIPSE_XY_RADIUS_X, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER, PATH_CENTER + ELLIPSE_XY_RADIUS_Y, PATH_CENTER, 300),
    (PATH_CENTER - ELLIPSE_XY_RADIUS_X, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER - OFFSET, PATH_CENTER - ELLIPSE_XY_RADIUS_Y, PATH_CENTER),
]

ELLIPSE_XZ_RADIUS_X = 0.1
ELLIPSE_XZ_RADIUS_Z = 0.45
ELLIPSE_XZ = [
    (PATH_CENTER + OFFSET, PATH_CENTER, PATH_CENTER - ELLIPSE_XZ_RADIUS_Z),
    (PATH_CENTER + ELLIPSE_XZ_RADIUS_X, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER, PATH_CENTER, PATH_CENTER + ELLIPSE_XZ_RADIUS_Z, 300),
    (PATH_CENTER - ELLIPSE_XZ_RADIUS_X, PATH_CENTER, PATH_CENTER, 100),
    (PATH_CENTER - OFFSET, PATH_CENTER,PATH_CENTER - ELLIPSE_XZ_RADIUS_Z),
]

GEOMETRIC: dict[str, Path] = {
    'GEOMETRIC1_SPIRAL_3DY': SPIRAL_Y,
    'GEOMETRIC2_SPIRAL_3DX': counter_clock_wise_rotation(SPIRAL_Y),
    'GEOMETRIC3_SPIRAL_XY': GOLDEN_CURVE,
    'GEOMETRIC4_CIRCLE_XY': CIRCLE_XY,
    'GEOMETRIC5_CIRCLE_ZX': CIRCLE_ZX,
    'GEOMETRIC6_ELLIPSE_XY': ELLIPSE_XY,
    'GEOMETRIC7_ELLIPSE_XZ': ELLIPSE_XZ,
}

PATHS: dict[str, Path] = {
    **BURST,
    # **CURVE,
    **GEOMETRIC,
    **LETTER,
}

for name, path in PATHS.items():
    PATHS[name] = normalize_path(path, size=PATHS_LENGTH, center=CUBE_CENTER, uniform=True)

CONTROL_RATIO = 100
CONTROL_WEIGHT = 50
END_CONTROL_WEIGHT = CONTROL_WEIGHT / CONTROL_RATIO

OUTPUT_FOLDER = os.path.join(os.getcwd(), 'images', 'paths')
OUTPUT_YAML = os.path.join(os.getcwd(), 'images', 'paths.yaml')

yaml_dict = {'paths': []}
for path_name, path_value in PATHS.items():
    yaml_dict['paths'].append({
        'name': path_name,
        'waypoints': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in path_value]
    })

with open(OUTPUT_YAML, 'w') as yaml_file:
    dump(yaml_dict, yaml_file)
    
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

    # Scambio Y/Z solo ai fini della visualizzazione (vedi SWAP_Y_Z sopra).
    curve_pts = maybe_swap_yz(curve_pts)
    ctrl_arr  = maybe_swap_yz(ctrl_arr)

    # Colore = ordine lungo il percorso (0 = waypoint 1 / rosso, 1 = waypoint
    # N / blu). Per la curva uso il parametro t (gia' 0->1 lungo il
    # tragitto); per i punti di controllo uso l'indice di sequenza
    # normalizzato. Stesso ORDER_NORM (0-1) per tutti i path, cosi' anche il
    # colore resta confrontabile tra plot diversi.
    t_curve = np.linspace(0.0, 1.0, len(curve_pts))
    order_ctrl = np.linspace(0.0, 1.0, len(ctrl_arr))

    seg_colors  = CMAP(ORDER_NORM(t_curve[:-1]))
    ctrl_colors = CMAP(ORDER_NORM(order_ctrl))

    xlim = (NORM_MIN - NORM_MARGIN, NORM_MAX + NORM_MARGIN)
    ylim = (NORM_MIN - NORM_MARGIN, NORM_MAX + NORM_MARGIN)
    zlim = (NORM_MIN - NORM_MARGIN, NORM_MAX + NORM_MARGIN)
    if SWAP_Y_Z:
        ylim, zlim = zlim, ylim  # per ora identici, ma corretto se in futuro NORM_MIN/MAX differissero per asse

    # ---------- Plot 3D con proiezioni sui tre piani ----------
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    segs_xyz = np.stack([curve_pts[:-1], curve_pts[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segs_xyz, colors=seg_colors, linewidths=2.2, zorder=6))
    draw_ctrl_numbered_3d(ax, ctrl_arr, ctrl_colors)
    add_plane_projections(ax, curve_pts, seg_colors, ctrl_arr, ctrl_colors, xlim, ylim, zlim)

    ax.set(xlabel='X', ylabel=('Z' if SWAP_Y_Z else 'Y'),
           zlabel=('Y' if SWAP_Y_Z else 'Z'), title=f'{name}')
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.view_init(elev=VIEW_PITCH, azim=VIEW_YAW, roll=VIEW_ROLL)
    if INVERT_Y_AXIS:
        ax.invert_yaxis()
    if INVERT_Z_AXIS:
        ax.invert_zaxis()

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=ORDER_NORM)
    sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
    # cbar.set_label('Ordine (1 -> N)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)