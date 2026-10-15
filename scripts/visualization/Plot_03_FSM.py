"""
FIGURE: Flood Susceptibility Maps — Natural Breaks (Jenks) Classification
===========================================================================
Produces one composed figure showing susceptibility maps for all 5 models
side-by-side with a shared 5-class Natural Breaks legend.

WHY NATURAL BREAKS (JENKS) over Quantile?
  - Natural Breaks finds statistically natural groupings in the actual data
    distribution, meaning class boundaries are placed where the data has
    the largest jumps — perfect for skewed susceptibility distributions.
  - Quantile forces equal pixel counts per class, which can lump together
    genuinely different susceptibility zones just to equalise class sizes.
  - For publication & policy use, Natural Breaks is the GIS standard for
    hazard/susceptibility maps (e.g., USGS, UN-SPIDER guidelines).

Classes produced (Jenks):
    Very Low | Low | Moderate | High | Very High
"""

import os
import glob
import numpy as np
import rasterio
import rasterio.plot as rplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar   # pip install matplotlib-scalebar
import geopandas as gpd
import jenkspy                                        # pip install jenkspy

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
PREDICTION_DIR = "Khagaria_FSM/Predictions"
BOUNDARY_SHP   = "Khagaria_FSM/Shapefiles/Khagaria_Boundary_UTM.shp"
OUT_DIR        = "Khagaria_FSM/Manuscript_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

# 5-class flood susceptibility colour ramp (white → deep red)
SUSC_COLORS = ["#1A9641", "#A6D96A", "#FFFFBF", "#FDAE61", "#D7191C"]
CLASS_LABELS = ["Very Low", "Low", "Moderate", "High", "Very High"]
N_CLASSES    = 5

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def compute_jenks_breaks(data_flat, n_classes=5, sample_n=100_000):
    """
    Computes Natural Breaks (Jenks) on a flat valid array.
    Subsamples for speed on large rasters.
    Returns list of (n_classes+1) break values including min and max.
    """
    valid = data_flat[~np.isnan(data_flat)]
    valid = valid[valid >= 0]                    # exclude nodata artifacts
    if len(valid) > sample_n:
        np.random.seed(42)
        valid = np.random.choice(valid, sample_n, replace=False)
    breaks = jenkspy.jenks_breaks(valid.tolist(), n_classes=n_classes)
    return breaks                                # [min, b1, b2, b3, b4, max]

def classify_with_breaks(arr, breaks):
    """Map float probabilities to integer class IDs 0–(n-1)."""
    classified = np.zeros_like(arr, dtype=np.uint8)
    for i in range(1, len(breaks)):
        if i == 1:
            classified[(arr >= breaks[i-1]) & (arr <= breaks[i])] = i - 1
        else:
            classified[(arr > breaks[i-1]) & (arr <= breaks[i])]  = i - 1
    classified[np.isnan(arr)] = 255             # nodata
    return classified

def build_colormap_and_norm(breaks, colors):
    """Returns a ListedColormap + BoundaryNorm for exact class boundaries."""
    cmap  = mcolors.ListedColormap(colors)
    norm  = mcolors.BoundaryNorm(breaks, ncolors=len(colors))
    return cmap, norm

def add_north_arrow(ax, x=0.92, y=0.88):
    ax.annotate("N", xy=(x, y + 0.06), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

# ─── LOAD BOUNDARY ───────────────────────────────────────────────────────────
boundary = gpd.read_file(BOUNDARY_SHP)

# ─── LOAD PREDICTION FILES ───────────────────────────────────────────────────
pred_files = sorted(glob.glob(os.path.join(PREDICTION_DIR, "*.tif")))
# Use a preferred model order for the figure
order = ["XGBoost", "RandomForest", "CatBoost", "ANN", "DNN"]
ordered = []
for name in order:
    match = [f for f in pred_files if name in f]
    if match:
        ordered.append((name, match[0]))
# Add any remaining not in order list
for f in pred_files:
    nm = os.path.basename(f).replace("Map_","").replace(".tif","")
    if nm not in [o[0] for o in ordered]:
        ordered.append((nm, f))

n_models = len(ordered)
print(f"Found {n_models} prediction maps.")

# ─── COMPUTE GLOBAL BREAKS (shared across all models) ────────────────────────
# Option A: Global breaks based on first model (recommended for comparison)
# Option B: Per-model breaks (shows relative within-model spread better)
# We use GLOBAL here so all panels share the same legend — essential for comparison

print("Computing global Jenks breaks from first model...")
with rasterio.open(ordered[0][1]) as src:
    sample_data = src.read(1).flatten().astype(float)
    sample_data[sample_data < -9000] = np.nan
global_breaks = compute_jenks_breaks(sample_data, N_CLASSES)
print(f"  Global Breaks: {[f'{b:.3f}' for b in global_breaks]}")

cmap, norm = build_colormap_and_norm(global_breaks, SUSC_COLORS)

# ─── FIGURE LAYOUT ───────────────────────────────────────────────────────────
ncols = min(3, n_models)
nrows = int(np.ceil(n_models / ncols))

fig = plt.figure(figsize=(6.5 * ncols, 7 * nrows + 1.2), facecolor="white")
gs  = gridspec.GridSpec(nrows, ncols, figure=fig,
                        hspace=0.02, wspace=0.06,
                        left=0.03, right=0.97,
                        top=0.93, bottom=0.10)

panel_labels = "abcdefghij"

for idx, (model_name, fpath) in enumerate(ordered):
    row, col = divmod(idx, ncols)
    ax = fig.add_subplot(gs[row, col])

    with rasterio.open(fpath) as src:
        data     = src.read(1).astype(float)
        extent   = rplot.plotting_extent(src)
        data[data < -9000] = np.nan

    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm,
                   origin="upper", interpolation="nearest")

    # Overlay boundary
    boundary_proj = boundary.to_crs(src.crs)
    boundary_proj.boundary.plot(ax=ax, color="black", linewidth=0.9)

    # Set extent from boundary
    b = boundary_proj.total_bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])

    ax.set_title(f"({panel_labels[idx]}) {model_name}",
                 fontsize=16, fontweight="bold", pad=5)
    ax.set_axis_off()
    add_north_arrow(ax)

    # Scale bar on every panel
    try:
        scalebar = ScaleBar(1, units="m", dimension="si-length",
                            length_fraction=0.25, location="lower right",
                            box_alpha=0.7, font_properties={"size": 12})
        ax.add_artist(scalebar)
    except Exception:
        pass  # matplotlib-scalebar not installed

# ── Hide empty axes ───────────────────────────────────────────────────────────
total_cells = nrows * ncols
for idx in range(n_models, total_cells):
    row, col = divmod(idx, ncols)
    fig.add_subplot(gs[row, col]).set_axis_off()

# ── Shared Colour Legend (bottom) ─────────────────────────────────────────────
legend_ax = fig.add_axes([0.15, 0.02, 0.70, 0.045])   # [left, bottom, w, h]
patches   = [mpatches.Patch(facecolor=SUSC_COLORS[i], edgecolor="black",
                             linewidth=0.6, label=f"{CLASS_LABELS[i]}\n({global_breaks[i]:.2f}–{global_breaks[i+1]:.2f})")
             for i in range(N_CLASSES)]
legend_ax.legend(handles=patches, loc="center", ncol=N_CLASSES,
                 fontsize=9.5, frameon=True, framealpha=0.9,
                 title="Flood Susceptibility Classes (Natural Breaks / Jenks)",
                 title_fontsize=16, edgecolor="grey")
legend_ax.set_axis_off()

fig.suptitle("Flood Susceptibility Maps — Khagaria District\n(Probability classified using Natural Breaks/Jenks into 5 classes)",
             fontsize=16, fontweight="bold", y=0.97)

out_path = os.path.join(OUT_DIR, "Fig_Flood_Susceptibility_Maps.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"✅ Saved: {out_path}")
plt.close()

# ─── ALSO SAVE GeoTIFF of classified maps ─────────────────────────────────────
print("\nSaving classified GeoTIFFs...")
CLASSIFIED_DIR = os.path.join(OUT_DIR, "Classified_Maps")
os.makedirs(CLASSIFIED_DIR, exist_ok=True)

for model_name, fpath in ordered:
    with rasterio.open(fpath) as src:
        data  = src.read(1).astype(float)
        meta  = src.meta.copy()
        data[data < -9000] = np.nan
    classified = classify_with_breaks(data, global_breaks)
    meta.update(dtype="uint8", count=1, nodata=255)
    out = os.path.join(CLASSIFIED_DIR, f"Classified_{model_name}_NaturalBreaks.tif")
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(classified, 1)
    print(f"  Saved: {os.path.basename(out)}")

# ─── BREAK VALUES REPORT ──────────────────────────────────────────────────────
import pandas as pd
breaks_df = pd.DataFrame({
    "Class"      : CLASS_LABELS,
    "Min_Prob"   : global_breaks[:-1],
    "Max_Prob"   : global_breaks[1:],
})
breaks_df.to_csv(os.path.join(OUT_DIR, "NaturalBreaks_ClassBreaks.csv"), index=False)
print("\nClass break values:")
print(breaks_df.to_string(index=False))
print("\n✅ All done.")