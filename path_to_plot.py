from yaml import safe_load
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registra il proiettore '3d')
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter
import os

INPUT_YAML    = os.path.join(os.getcwd(), 'paths.yaml')
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'images', 'paths')

# ------------------------------------------------------------
# Impostazioni per la stampa nella tesi
# ------------------------------------------------------------
# Le figure sono pensate per essere incluse a circa un terzo della larghezza
# del testo (0.31\linewidth, circa 5 cm) senza essere ridotte: la figura ha
# gia' le dimensioni finali e i caratteri la loro dimensione finale in punti.
FIG_SIZE_IN   = 2.3                 # lato della figura in pollici, prima del ritaglio
FONT_SIZE_PT  = 8                   # etichette degli assi
TICK_SIZE_PT  = 7                   # numeri sulle tacche
LABEL_SIZE_PT = 7                   # numeri dei waypoint
SHOW_TITLE    = False               # il nome del task va nella sotto-didascalia LaTeX
AXIS_TICKS_M  = (-0.1, 0.0, 0.1)    # tacche in metri, mostrate in millimetri
BOX_ZOOM      = 1.0                 # ingrandimento del cubo nella figura
SAVE_FORMATS  = ('pdf', 'png')      # PDF vettoriale per la tesi, PNG per l'anteprima
MAIN_LINEWIDTH   = 1.3
SHADOW_LINEWIDTH = 0.8
TICK_PAD  = 0                       # distanza tra tacche ed etichette (punti); se negativa
                                    # matplotlib non disegna il nome dell'asse verticale
LABEL_PAD = -3                      # distanza tra asse e nome dell'asse (punti)
AXES_RECT = (0.0, 0.06, 0.86, 0.94) # posizione degli assi nella figura (frazioni)

# Niente trasparenze: la tesi e' compilata in PDF/A-1b (pdfx a-1b), che non
# ammette trasparenze. Le ombre sono quindi colori mescolati con il colore
# delle pareti, che a vista equivalgono a un'opacita' SHADOW_KEEP.
SHADOW_KEEP = 0.35
PANE_COLOR  = (0.95, 0.95, 0.95, 1.0)
GRID_COLOR  = '#d0d0d0'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': FONT_SIZE_PT,
    'axes.linewidth': 0.6,
    'grid.color': GRID_COLOR,
    'grid.linewidth': 0.4,
    'grid.linestyle': '-',
    'pdf.fonttype': 42,             # font TrueType incorporati, non Type 3
})

# ============================================================
# Utils
# ============================================================
Path = list[tuple[float, float, float]]
_ANC   = dict(marker='o', s=20, zorder=6, edgecolors='black', linewidths=0.5, depthshade=False)
_HDL   = dict(marker='D', s=13, zorder=5, edgecolors='gray',  linewidths=0.4, depthshade=False)


def lighten(colors: np.ndarray, keep: float) -> np.ndarray:
    """Mescola i colori RGBA con il colore delle pareti, mantenendo una
    frazione `keep` del colore originale, e li rende opachi. Sostituisce
    l'alpha, che il PDF/A-1b non ammette."""
    c = np.array(colors, dtype=float, copy=True)
    c[..., :3] = keep * c[..., :3] + (1.0 - keep) * np.asarray(PANE_COLOR[:3])
    c[..., 3] = 1.0
    return c


_MM = FuncFormatter(lambda v, _: f'{v * 1000:.0f}'.replace('-', '−'))

# Limiti degli assi: per ciascun asse si prendono minimo e massimo assoluti
# su TUTTI i waypoint di TUTTI i path letti da INPUT_YAML (vedi
# global_axis_limits), cosi' ogni plot ha gli stessi valori sugli assi e i
# path restano confrontabili tra loro. NORM_MARGIN (stessa unita' dei dati,
# metri) e' solo un respiro visivo per non tagliare a meta' i marker che
# cadono esattamente sul bordo; non altera il dominio dei dati.
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

# Inverti il verso di crescita di un asse: la vista (VIEW_*) resta la stessa,
# ma i percorsi appaiono specchiati rispetto al piano perpendicolare a
# quell'asse. Il flag si riferisce all'asse con quella ETICHETTA nel plot
# (cioe' alla coordinata dei dati), indipendentemente da SWAP_Y_Z: con
# SWAP_Y_Z attivo, INVERT_Z_AXIS agisce sull'asse orizzontale etichettato 'Z'.
INVERT_Y_AXIS = False
INVERT_Z_AXIS = True


def maybe_swap_yz(arr: np.ndarray) -> np.ndarray:
    """Se SWAP_Y_Z e' attivo, scambia le colonne Y (indice 1) e Z (indice 2)
    di un array di punti (N, 3). Usata solo ai fini del plot, non tocca il
    calcolo della curva di Bezier ne' i colori (basati sull'ordine)."""
    if not SWAP_Y_Z:
        return arr
    swapped = arr.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    return swapped


def polyline(points: np.ndarray, samples_per_segment: int = 100
             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Spezzata che unisce ogni coppia di waypoint consecutivi con un segmento
    rettilineo. Ogni segmento e' campionato fittamente solo per poter
    sfumare il colore lungo il percorso.

    Restituisce i punti campionati (M, 3) e, per ciascuno, l'ordine
    normalizzato in [0, 1]: sul waypoint i vale esattamente i / (N-1), cosi'
    il colore della linea coincide con quello del marker del waypoint.
    """
    n_seg = len(points) - 1
    s = np.linspace(0.0, n_seg, n_seg * samples_per_segment + 1)
    seg = np.minimum(s.astype(int), n_seg - 1)    # indice del segmento
    frac = (s - seg)[:, None]                      # posizione nel segmento
    pts = points[seg] * (1.0 - frac) + points[seg + 1] * frac
    return pts, s / n_seg


def draw_ctrl_numbered_3d(ax, arr: np.ndarray, colors: np.ndarray,
                           label_offset: float = 0.005,
                           label_dir: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
    """Waypoint colorati in base all'ordine, con numero di sequenza accanto
    a ciascuno (versione 3D). label_offset e' in unita' dei dati (metri);
    label_dir da' il segno dello spostamento per asse (-1 sugli assi
    invertiti, cosi' sullo schermo l'etichetta finisce sempre dalla stessa
    parte del punto)."""
    offset = label_offset * np.asarray(label_dir, dtype=float)
    ax.scatter(arr[[0, -1], 0], arr[[0, -1], 1], arr[[0, -1], 2],
               color=colors[[0, -1]], **_ANC)
    if len(arr) > 2:
        ax.scatter(arr[1:-1, 0], arr[1:-1, 1], arr[1:-1, 2],
                   color=colors[1:-1], **_HDL)

    # Numero di sequenza: 1 = punto iniziale, ..., n = punto finale
    for i, (x, y, z) in enumerate(arr, start=1):
        ax.text(
            x + offset[0], y + offset[1], z + offset[2], str(i),
            fontsize=LABEL_SIZE_PT, fontweight='bold', color='black',
            bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none'),
            zorder=10,
        )


def add_plane_projections(ax, curve_pts: np.ndarray, seg_colors: np.ndarray,
                           ctrl_arr: np.ndarray, ctrl_colors: np.ndarray,
                           xlim: tuple[float, float], ylim: tuple[float, float],
                           zlim: tuple[float, float],
                           keep: float = SHADOW_KEEP, linewidth: float = SHADOW_LINEWIDTH,
                           point_size: float = 8, degenerate_tol: float = 1e-9) -> None:
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

    # Ombre schiarite (non trasparenti), mantenendo il gradiente di colore.
    shadow_colors = lighten(seg_colors, keep)
    ctrl_shadow_colors = lighten(ctrl_colors, keep)

    def draw_wall_line(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> None:
        pts = np.column_stack([xs, ys, zs])
        extent = pts.max(axis=0) - pts.min(axis=0)
        if np.all(extent < degenerate_tol):
            # Traiettoria degenere su questa parete -> un solo punto: uso un
            # marker (colore a meta' gradiente, piu' opaco) invece della linea.
            mid_color = lighten(seg_colors[len(seg_colors) // 2], min(1.0, keep * 2.5))
            ax.scatter([xs[0]], [ys[0]], [zs[0]], color=[mid_color],
                       s=point_size, zorder=2, edgecolors='none', depthshade=False)
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
                   s=_ANC['s'], zorder=1, edgecolors='none', depthshade=False)
        # Handle (punti di controllo intermedi) -> diamante
        if len(xs) > 2:
            ax.scatter(xs[1:-1], ys[1:-1], zs[1:-1],
                       color=ctrl_shadow_colors[1:-1], marker=_HDL['marker'],
                       s=_HDL['s'], zorder=1, edgecolors='none', depthshade=False)

    n, nc = len(x), len(cx)

    # I limiti sono (inizio, fine) dell'asse: se un asse e' invertito la
    # tupla e' rovesciata, quindi gli indici qui sotto puntano comunque alle
    # pareti di fondo del box.

    # XY -> pavimento (z = inizio dell'asse z)
    draw_wall_line(x, y, np.full(n, zlim[0]))
    draw_wall_points(cx, cy, np.full(nc, zlim[0]))

    # XZ -> parete di fondo (y = fine dell'asse y)
    draw_wall_line(x, np.full(n, ylim[1]), z)
    draw_wall_points(cx, np.full(nc, ylim[1]), cz)

    # YZ -> parete laterale (x = inizio dell'asse x)
    draw_wall_line(np.full(n, xlim[0]), y, z)
    draw_wall_points(np.full(nc, xlim[0]), cy, cz)


# ============================================================
# Input
# ============================================================

def load_paths(yaml_path: str) -> dict[str, Path]:
    """
    Legge i percorsi da un file YAML con struttura

        paths:
        - name: <nome>
          waypoints:
          - {x: ..., y: ..., z: ...}

    e li restituisce come {nome: [(x, y, z), ...]}, nell'ordine in cui
    compaiono nel file.
    """
    with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
        data = safe_load(yaml_file)

    paths: dict[str, Path] = {}
    for entry in data['paths']:
        name = entry['name']
        if name in paths:
            raise ValueError(f"Path '{name}' duplicato in {yaml_path}")

        waypoints: Path = []
        for wp in entry['waypoints']:
            waypoints.append((float(wp['x']), float(wp['y']), float(wp['z'])))

        if len(waypoints) < 2:
            raise ValueError(f"Path '{name}' in {yaml_path} ha meno di 2 waypoint")
        paths[name] = waypoints
    return paths


def global_axis_limits(paths: dict[str, Path], margin: float = 0.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimo e massimo assoluti, per ciascun asse (X, Y, Z), su tutti i
    waypoint di tutti i path, allargati di `margin`. Basta guardare i
    waypoint: la spezzata che li unisce non esce da questi limiti.
    """
    all_pts = np.array([p for path in paths.values() for p in path], dtype=float)
    return all_pts.min(axis=0) - margin, all_pts.max(axis=0) + margin


PATHS: dict[str, Path] = load_paths(INPUT_YAML)

# Limiti comuni a tutti i plot, nello stesso ordine di colonne usato nel plot
# (quindi gia' scambiati se SWAP_Y_Z e' attivo).
_axis_min, _axis_max = global_axis_limits(PATHS, NORM_MARGIN)
_plot_lims = maybe_swap_yz(np.stack([_axis_min, _axis_max]))   # (2, 3): righe = min, max
XLIM = (float(_plot_lims[0, 0]), float(_plot_lims[1, 0]))
YLIM = (float(_plot_lims[0, 1]), float(_plot_lims[1, 1]))
ZLIM = (float(_plot_lims[0, 2]), float(_plot_lims[1, 2]))

# Inversione degli assi: i limiti sono tuple (inizio, fine) dell'asse cosi'
# come vanno passate a set_*lim, quindi rovesciarle inverte l'asse. Anche
# add_plane_projections ragiona per inizio/fine, cosi' le ombre restano
# sulle pareti di fondo del box anche ad asse invertito.
_invert_plot_y = INVERT_Z_AXIS if SWAP_Y_Z else INVERT_Y_AXIS
_invert_plot_z = INVERT_Y_AXIS if SWAP_Y_Z else INVERT_Z_AXIS
if _invert_plot_y:
    YLIM = YLIM[::-1]
if _invert_plot_z:
    ZLIM = ZLIM[::-1]
print(f"Limiti assi del plot (comuni a tutti i path): x={XLIM}  y={YLIM}  z={ZLIM}")

if not os.path.exists(os.path.join(os.getcwd(), 'images')):
    raise ValueError('Path not found')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

for name, path in PATHS.items():
    print(f"{name}: spezzata con {len(path)} waypoint ({len(path) - 1} segmenti)")

    ctrl_arr = np.array(path, dtype=float)       # waypoint (N, 3)
    curve_pts, t_curve = polyline(ctrl_arr)

    # Scambio Y/Z solo ai fini della visualizzazione (vedi SWAP_Y_Z sopra).
    curve_pts = maybe_swap_yz(curve_pts)
    ctrl_arr  = maybe_swap_yz(ctrl_arr)

    # Colore = ordine lungo il percorso (0 = waypoint 1 / rosso, 1 = waypoint
    # N / blu). Per la spezzata uso l'ordine restituito da polyline (i/(N-1)
    # sul waypoint i, interpolato lungo ogni segmento); per i waypoint uso
    # l'indice di sequenza normalizzato. Stesso ORDER_NORM (0-1) per tutti i
    # path, cosi' anche il colore resta confrontabile tra plot diversi.
    order_ctrl = np.linspace(0.0, 1.0, len(ctrl_arr))

    seg_colors  = CMAP(ORDER_NORM(t_curve[:-1]))
    ctrl_colors = CMAP(ORDER_NORM(order_ctrl))

    # Limiti globali (uguali per tutti i path, gia' scambiati se SWAP_Y_Z e
    # gia' rovesciati per gli assi invertiti).
    xlim, ylim, zlim = XLIM, YLIM, ZLIM

    # ---------- Plot 3D con proiezioni sui tre piani ----------
    fig = plt.figure(figsize=(FIG_SIZE_IN, FIG_SIZE_IN))
    # Assi quasi su tutta la figura: con i margini predefiniti del subplot il
    # cubo verrebbe ritagliato. Il margine a destra e in basso tiene dentro la
    # figura i nomi degli assi, che bbox_inches='tight' non considera quando
    # cadono fuori; lo spazio vuoto lo toglie poi il ritaglio al salvataggio.
    ax = fig.add_axes(AXES_RECT, projection='3d')

    segs_xyz = np.stack([curve_pts[:-1], curve_pts[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segs_xyz, colors=seg_colors, linewidths=MAIN_LINEWIDTH, zorder=6))
    label_dir = tuple(float(np.sign(lim[1] - lim[0])) for lim in (xlim, ylim, zlim))
    draw_ctrl_numbered_3d(ax, ctrl_arr, ctrl_colors, label_dir=label_dir)  # type: ignore[arg-type]
    add_plane_projections(ax, curve_pts, seg_colors, ctrl_arr, ctrl_colors, xlim, ylim, zlim)

    ax.set_xlabel('X', labelpad=LABEL_PAD)
    ax.set_ylabel('Z' if SWAP_Y_Z else 'Y', labelpad=LABEL_PAD)
    ax.set_zlabel('Y' if SWAP_Y_Z else 'Z', labelpad=LABEL_PAD)
    if SHOW_TITLE:
        ax.set_title(name)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    # Tacche uguali su tutti gli assi, mostrate in millimetri.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(FixedLocator(AXIS_TICKS_M))
        axis.set_major_formatter(_MM)
        axis.set_pane_color(PANE_COLOR)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE_PT, pad=TICK_PAD)
    # Box proporzionale all'estensione di ciascun asse: stessa scala su X, Y
    # e Z, cosi' lunghezze e angoli dei percorsi non vengono deformati.
    ax.set_box_aspect((abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0]), abs(zlim[1] - zlim[0])),
                      zoom=BOX_ZOOM)
    ax.grid(True)
    ax.view_init(elev=VIEW_PITCH, azim=VIEW_YAW, roll=VIEW_ROLL)

    # I nomi degli assi vanno passati esplicitamente: il riquadro calcolato da
    # bbox_inches='tight' per gli assi 3D esclude quello dell'asse verticale.
    axis_labels = [ax.xaxis.label, ax.yaxis.label, ax.zaxis.label]
    for fmt in SAVE_FORMATS:
        plt.savefig(os.path.join(OUTPUT_FOLDER, f'{name}.{fmt}'), dpi=300,
                    bbox_inches='tight', pad_inches=0.02, bbox_extra_artists=axis_labels)
    plt.close(fig)