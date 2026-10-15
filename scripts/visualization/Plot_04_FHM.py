"""
FIGURE: Flood Hazard Maps + Ensemble Agreement Map
====================================================
Hazard differs from Susceptibility:
  Susceptibility = likelihood of flooding (model output probability)
  Hazard         = susceptibility × exposure (population, land use, assets)

This script produces:
  1. Flood Hazard Maps  — susceptibility × normalised exposure layer
     (uses population density raster as proxy for exposure)
  2. Ensemble Agreement Map — pixel-level count of how many models
     classify a pixel as "High/Very High" susceptibility (0–5 models agree)

The Ensemble Agreement Map is a powerful validation & consensus figure.
"""

import os
import glob
import numpy as np
import rasterio
import rasterio.plot as rplot
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import geopandas as gpd
import pandas as pd

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE           = "Khagaria_FSM"
PREDICTION_DIR = os.path.join(BASE, "Predictions")
BOUNDARY_SHP   = os.path.join(BASE, "Shapefiles/Khagaria_Boundary_UTM.shp")
# Exposure layer: population density (WorldPop 100 m) OR LULC risk weights
# If you don't have one, set to None and script will skip hazard maps
EXPOSURE_RASTER = os.path.join(BASE, "Raster/Aligned/Population_Density.tif")   # or None
GROUND_TRUTH    = os.path.join(BASE, "Inventory/F_nF_Khagaria_binary.tif")
OUT_DIR         = os.path.join(BASE, "Manuscript_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Max-CSI thresholds (paste from your Final_Validation_MaxCSI.csv)
# The script will auto-load these if the CSV exists, otherwise uses 0.5
MAX_CSI_CSV = os.path.join(BASE, "Final_Binary_Maps_MaxCSI/Final_Validation_MaxCSI.csv")

# Hazard colour ramp
HAZARD_COLORS = ["#1A9641", "#A6D96A", "#FFFFBF", "#FDAE61", "#D7191C"]
CLASS_LABELS  = ["Very Low", "Low", "Moderate", "High", "Very High"]

# Ensemble agreement colour ramp (0 to 5 models)
AGREE_COLORS  = ["#f7f7f7", "#cccccc", "#fc8d59", "#d7301f", "#7f0000", "#49000d"]

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def read_and_align(target_path, ref_meta):
    with rasterio.open(target_path) as src:
        if src.width == ref_meta["width"] and src.height == ref_meta["height"]:
            return src.read(1).astype(np.float32)
        dest = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dest,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.nearest)
        return dest

def normalise(arr, nodata_val=-9999):
    arr = arr.astype(float)
    arr[arr <= nodata_val] = np.nan
    vmin, vmax = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
    normed = (arr - vmin) / (vmax - vmin + 1e-10)
    normed = np.clip(normed, 0, 1)
    normed[np.isnan(arr)] = np.nan
    return normed

def add_north_arrow(ax, x=0.92, y=0.87):
    ax.annotate("N", xy=(x, y + 0.06), xytext=(x, y),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

# ─── LOAD THRESHOLDS ─────────────────────────────────────────────────────────
thresholds = {}
if os.path.exists(MAX_CSI_CSV):
    df_thresh = pd.read_csv(MAX_CSI_CSV)
    for _, row in df_thresh.iterrows():
        thresholds[row["Model"]] = row["Optimal_Threshold"]
    print(f"Loaded {len(thresholds)} thresholds from CSV.")
else:
    print("⚠️  Threshold CSV not found — using 0.5 for all models.")

# ─── LOAD REFERENCE ──────────────────────────────────────────────────────────
with rasterio.open(GROUND_TRUTH) as src:
    ref_meta = src.meta.copy()
    h, w     = src.shape

boundary = gpd.read_file(BOUNDARY_SHP)

# ─── LOAD EXPOSURE ───────────────────────────────────────────────────────────
has_exposure = EXPOSURE_RASTER and os.path.exists(EXPOSURE_RASTER)
if has_exposure:
    print("Loading exposure raster...")
    exposure_raw  = read_and_align(EXPOSURE_RASTER, ref_meta)
    exposure_norm = normalise(exposure_raw)
    print("  Exposure loaded and normalised.")
else:
    print("ℹ️  No exposure raster found — hazard maps will be skipped. Only ensemble map generated.")

# ─── LOAD ALL PREDICTION MAPS ────────────────────────────────────────────────
order      = ["XGBoost", "RandomForest", "CatBoost", "ANN", "DNN"]
pred_files = sorted(glob.glob(os.path.join(PREDICTION_DIR, "*.tif")))
ordered    = []
for nm in order:
    match = [f for f in pred_files if nm in f]
    if match:
        ordered.append((nm, match[0]))
for f in pred_files:
    nm = os.path.basename(f).replace("Map_","").replace(".tif","")
    if nm not in [o[0] for o in ordered]:
        ordered.append((nm, f))

prob_maps = {}
for model_name, fpath in ordered:
    print(f"  Loading {model_name}...")
    data = read_and_align(fpath, ref_meta).astype(float)
    data[data < -9000] = np.nan
    prob_maps[model_name] = data

# ─── ENSEMBLE AGREEMENT MAP ──────────────────────────────────────────────────
# For each pixel, count how many models predict HIGH or VERY HIGH susceptibility.
# "High" threshold = probability > 0.6 (conservative — change as needed).
# The resulting map (0–5) shows spatial consensus.

print("\nComputing ensemble agreement map...")
HIGH_PROB_THRESHOLD = 0.60   # adjust if needed
agreement = np.zeros((h, w), dtype=np.float32)
valid_count = np.zeros((h, w), dtype=np.float32)

for model_name, prob in prob_maps.items():
    thresh   = thresholds.get(model_name, 0.5)
    mask_nan = ~np.isnan(prob)
    valid_count += mask_nan.astype(float)
    # A pixel is "High" if probability > HIGH_PROB_THRESHOLD (absolute)
    # OR if probability > model-specific MaxCSI threshold (relative)
    # We use whichever is higher: max(0.6, MaxCSI_thresh)
    eff_thresh = max(HIGH_PROB_THRESHOLD, thresh)
    agreement += ((prob > eff_thresh) & mask_nan).astype(float)

agreement[valid_count == 0] = np.nan   # no-data pixels

# ─── FIGURE A: FLOOD HAZARD MAPS ─────────────────────────────────────────────
if has_exposure:
    n_models = len(ordered)
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))

    fig_h, axes = plt.subplots(nrows, ncols,
                                figsize=(6.5 * ncols, 7 * nrows + 1.5),
                                facecolor="white")
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    panel_labels = "abcdefghij"
    hazard_maps  = {}

    for idx, (model_name, _) in enumerate(ordered):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        prob = prob_maps[model_name]
        # Hazard = susceptibility (prob) × exposure (normalised)
        hazard = prob * exposure_norm
        hazard[np.isnan(prob) | np.isnan(exposure_norm)] = np.nan
        hazard_maps[model_name] = hazard

        im = ax.imshow(hazard,
                       extent=rplot.plotting_extent(rasterio.open(ordered[idx][1])),
                       cmap="RdYlGn_r", vmin=0, vmax=1,
                       origin="upper", interpolation="nearest")
        boundary_proj = boundary.to_crs(ref_meta["crs"])
        boundary_proj.boundary.plot(ax=ax, color="black", linewidth=0.9)
        b = boundary_proj.total_bounds
        ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
        ax.set_title(f"({panel_labels[idx]}) {model_name}", fontsize=11, fontweight="bold")
        ax.set_axis_off()
        add_north_arrow(ax)
        plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02).set_label("Hazard Index", fontsize=8)

    # Hide unused
    for idx in range(len(ordered), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_axis_off()

    fig_h.suptitle("Flood Hazard Maps — Khagaria District\n(Susceptibility × Population Exposure Index)",
                   fontsize=13, fontweight="bold", y=0.99)
    out_h = os.path.join(OUT_DIR, "Fig_Flood_Hazard_Maps.png")
    plt.savefig(out_h, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {out_h}")
    plt.close()

    # Save hazard GeoTIFFs
    HAZARD_DIR = os.path.join(OUT_DIR, "Hazard_Maps")
    os.makedirs(HAZARD_DIR, exist_ok=True)
    hazard_meta = ref_meta.copy()
    hazard_meta.update(dtype="float32", count=1, nodata=-9999)
    for model_name, hazard in hazard_maps.items():
        out = os.path.join(HAZARD_DIR, f"Hazard_{model_name}.tif")
        hazard_out = hazard.copy()
        hazard_out[np.isnan(hazard)] = -9999
        with rasterio.open(out, "w", **hazard_meta) as dst:
            dst.write(hazard_out.astype(np.float32), 1)
        print(f"  Saved hazard GeoTIFF: {os.path.basename(out)}")

# ─── FIGURE B: ENSEMBLE AGREEMENT MAP ────────────────────────────────────────
print("\nPlotting ensemble agreement map...")
n_agree_levels = len(ordered) + 1   # 0 to n_models
agree_cmap = mcolors.ListedColormap(AGREE_COLORS[:n_agree_levels])
agree_norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_agree_levels),
                                  ncolors=n_agree_levels)

fig2, ax2 = plt.subplots(1, 1, figsize=(9, 10), facecolor="white")

try:
    with rasterio.open(ordered[0][1]) as src0:
        extent_ref = rplot.plotting_extent(src0)
except:
    extent_ref = None

im2 = ax2.imshow(agreement, extent=extent_ref,
                 cmap=agree_cmap, norm=agree_norm,
                 origin="upper", interpolation="nearest")

boundary_proj = boundary.to_crs(ref_meta["crs"])
boundary_proj.boundary.plot(ax=ax2, color="black", linewidth=1.0)
b = boundary_proj.total_bounds
ax2.set_xlim(b[0], b[2]); ax2.set_ylim(b[1], b[3])
ax2.set_axis_off()
add_north_arrow(ax2)

# Legend
model_names = [o[0] for o in ordered]
n           = len(model_names)
agree_labels = [f"0 models agree (Not High)"] + \
               [f"{i} model{'s' if i>1 else ''} agree" for i in range(1, n)] + \
               [f"{n} models agree (All High)"]
patches_a = [mpatches.Patch(facecolor=AGREE_COLORS[i], edgecolor="black",
                             linewidth=0.5, label=agree_labels[i])
             for i in range(n_agree_levels)]
ax2.legend(handles=patches_a, loc="lower left", fontsize=9.5,
           framealpha=0.9, title="Ensemble Agreement\n(No. of models predicting High/Very High)",
           title_fontsize=9, edgecolor="grey")

models_str = ", ".join(model_names)
ax2.set_title(f"Ensemble Agreement Map — Khagaria District\nModels: {models_str}",
              fontsize=12, fontweight="bold", pad=8)


# ─── CALCULATE PIXEL AREA ────────────────────────────────────────────────────
# Extract pixel width and height from the affine transform
pixel_width = ref_meta["transform"][0]
pixel_height = abs(ref_meta["transform"][4]) # Often negative, so we take absolute value

# Calculate pixel area in square meters, then convert to sq. km. (1 sq km = 1,000,000 sq m)
pixel_area_sqm = pixel_width * pixel_height
pixel_area_sqkm = pixel_area_sqm / 1e6

print(f"ℹ️  Resolution: {pixel_width}m x {pixel_height}m | Pixel Area: {pixel_area_sqkm:.6f} sq. km.")

# ─── GROUND TRUTH AREA CALCULATION ───────────────────────────────────────────
with rasterio.open(GROUND_TRUTH) as src:
    gt_data = src.read(1)
    
    # Assuming standard binary map: 1 = Flooded, 0 = Non-Flooded
    # Update these values if your raster uses different integers
    flooded_pixels = np.sum(gt_data == 1)
    non_flooded_pixels = np.sum(gt_data == 0)

    flooded_area = flooded_pixels * pixel_area_sqkm
    non_flooded_area = non_flooded_pixels * pixel_area_sqkm

    print(f"\n--- Ground Truth Inventory Area ---")
    print(f"  Flooded Area     : {flooded_area:.2f} sq. km.")
    print(f"  Non-Flooded Area : {non_flooded_area:.2f} sq. km.")

# Percentage and Area annotation
print("\n--- Predicted Ensemble Area ---")
n_pixels = np.sum(~np.isnan(agreement))
for i in range(n_agree_levels):
    pixel_count = np.sum(agreement == i)
    pct = 100 * pixel_count / n_pixels
    area_sqkm = pixel_count * pixel_area_sqkm
    print(f"  {i} models agree: {pct:.1f}% of area | {area_sqkm:.2f} sq. km.")
    
    
# Percentage annotation
n_pixels = np.sum(~np.isnan(agreement))
for i in range(n_agree_levels):
    pct = 100 * np.sum(agreement == i) / n_pixels
    print(f"  {i} models agree: {pct:.1f}% of area")

out2 = os.path.join(OUT_DIR, "Fig_Ensemble_Agreement_Map.png")
plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white")
print(f"✅ Saved: {out2}")
plt.close()

# Save agreement GeoTIFF
agree_meta = ref_meta.copy()
agree_meta.update(dtype="uint8", count=1, nodata=255)
agree_out  = agreement.copy()
agree_out[np.isnan(agreement)] = 255
with rasterio.open(os.path.join(OUT_DIR, "Ensemble_Agreement.tif"), "w", **agree_meta) as dst:
    dst.write(agree_out.astype(np.uint8), 1)

print("\n✅ All hazard and ensemble figures complete.")