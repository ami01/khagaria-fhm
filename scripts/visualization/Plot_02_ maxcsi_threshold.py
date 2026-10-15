"""
FIGURE: Max-CSI Threshold Visualization
=========================================
Creates a multi-panel diagnostic figure showing:
  Panel A: CSI vs Threshold curves for all models (with optimal point marked)
  Panel B: Youden's J vs Threshold curves (side-by-side comparison)
  Panel C: Summary bar chart — optimal threshold value per model
  Panel D: Summary bar chart — CSI score at BOTH optimal thresholds (apples-to-apples)

This figure justifies WHY the Max-CSI method was chosen over Youden's J.
Panel D compares CSI@Max-CSI-threshold vs CSI@Youden-threshold (same metric, different thresholds).
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve
import pandas as pd

# ─── GLOBAL FONT / STYLE SETTINGS ────────────────────────────────────────────
FONT_FAMILY   = "DejaVu Sans"   # change to any installed font if desired
FONT_BASE     = 13              # tick labels, bar annotations
FONT_LABEL    = 14              # axis x/y labels
FONT_TITLE    = 14              # panel (a)(b)(c)(d) titles
FONT_LEGEND   = 13              # legend entries
FONT_SUPTITLE = 16              # figure super-title
BAR_ANNOT_GAP = 0.030           # vertical gap between bar top and annotation text

plt.rcParams.update({
    "font.family"        : FONT_FAMILY,
    "font.size"          : FONT_BASE,       # default for everything not overridden
    "axes.titlesize"     : FONT_TITLE,
    "axes.titleweight"   : "bold",
    "axes.labelsize"     : FONT_LABEL,
    "xtick.labelsize"    : FONT_BASE,
    "ytick.labelsize"    : FONT_BASE,
    "legend.fontsize"    : FONT_LEGEND,
    "figure.titlesize"   : FONT_SUPTITLE,
    "figure.titleweight" : "bold",
})

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
PREDICTION_DIR = "Khagaria_FSM/Predictions"
GROUND_TRUTH   = "Khagaria_FSM/Inventory/F_nF_Khagaria_binary.tif"
OUT_DIR        = "Khagaria_FSM/Manuscript_Figures"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_COLORS = {
    "XGBoost":      "#E63946",
    "RandomForest": "#2A9D8F",
    "CatBoost":     "#E9C46A",
    "ANN":          "#457B9D",
    "DNN":          "#8338EC",
}
MARKER_SIZE = 90

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def read_and_align(target_path, ref_meta):
    with rasterio.open(target_path) as src:
        if src.width == ref_meta["width"] and src.height == ref_meta["height"]:
            return src.read(1).flatten()
        dest = np.zeros((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dest,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                  resampling=Resampling.nearest)
        return dest.flatten()

# ─── LOAD GROUND TRUTH ───────────────────────────────────────────────────────
print("Loading ground truth...")
with rasterio.open(GROUND_TRUTH) as src:
    y_raw    = src.read(1).flatten()
    ref_meta = src.meta.copy()

valid_mask = (y_raw != -9999) & (~np.isnan(y_raw))
y_true     = (y_raw[valid_mask] > 0).astype(int)
P = y_true.sum()
N = (y_true == 0).sum()

# ─── COMPUTE CURVES ──────────────────────────────────────────────────────────
print("Computing CSI and Youden curves...")
pred_files = sorted(glob.glob(os.path.join(PREDICTION_DIR, "*.tif")))

curve_data   = {}
summary_rows = []

for fpath in pred_files:
    model  = os.path.basename(fpath).replace("Map_", "").replace(".tif", "")
    y_prob = read_and_align(fpath, ref_meta)[valid_mask]
    y_prob = np.nan_to_num(y_prob, nan=0.0)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # CSI
    numer   = tpr * P
    denom   = P + fpr * N
    csi_arr = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
    ix_csi  = np.argmax(csi_arr)

    # Youden's J
    J_arr = tpr - fpr
    ix_J  = np.argmax(J_arr)

    # CSI evaluated at Youden's optimal threshold (apples-to-apples comparison)
    csi_at_youden = csi_arr[ix_J]

    curve_data[model] = dict(
        thresh          = thresholds,
        csi             = csi_arr,
        J               = J_arr,
        best_thresh_csi = thresholds[ix_csi],
        best_csi        = csi_arr[ix_csi],
        best_thresh_J   = thresholds[ix_J],
        best_J          = J_arr[ix_J],
        csi_at_youden   = csi_at_youden,
    )

    summary_rows.append({
        "Model"         : model,
        "Thresh_CSI"    : thresholds[ix_csi],
        "Max_CSI"       : csi_arr[ix_csi],
        "Thresh_Youden" : thresholds[ix_J],
        "Max_J"         : J_arr[ix_J],
        "CSI_at_Youden" : csi_at_youden,
    })

    print(f"  {model:15s} | MaxCSI thresh={thresholds[ix_csi]:.3f}  CSI={csi_arr[ix_csi]:.4f}"
          f" | Youden thresh={thresholds[ix_J]:.3f}  CSI={csi_at_youden:.4f}  J={J_arr[ix_J]:.4f}")

df_summary = pd.DataFrame(summary_rows).sort_values("Max_CSI", ascending=False)

# ─── FIGURE ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16), facecolor="white")
gs  = gridspec.GridSpec(2, 2, figure=fig,
                        hspace=0.18,   # more vertical breathing room between rows
                        wspace=0.12,
                        left=0.07, right=0.97, top=0.92, bottom=0.08)

ax_csi = fig.add_subplot(gs[0, 0])
ax_J   = fig.add_subplot(gs[0, 1])
ax_thr = fig.add_subplot(gs[1, 0])
ax_sc  = fig.add_subplot(gs[1, 1])

# ── helper: annotate bars uniformly ──────────────────────────────────────────
def annotate_bars(ax, bars, text_color):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + BAR_ANNOT_GAP,
                f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=FONT_BASE, fontweight="bold",
                color=text_color)

# ── Panel A: CSI vs Threshold ─────────────────────────────────────────────────
for model, cd in curve_data.items():
    color = MODEL_COLORS.get(model, "grey")
    n    = len(cd["thresh"])
    step = max(1, n // 500)
    idx  = np.arange(0, n, step)
    ax_csi.plot(cd["thresh"][idx], cd["csi"][idx],
                color=color, lw=2.0, label=model, alpha=0.9)
    ax_csi.scatter(cd["best_thresh_csi"], cd["best_csi"],
                   color=color, s=MARKER_SIZE, marker="D",
                   edgecolors="black", lw=1.0, zorder=6)
    ax_csi.vlines(cd["best_thresh_csi"], 0, cd["best_csi"],
                  colors=color, linestyles=":", lw=1.0, alpha=0.6)

ax_csi.set_xlabel("Threshold Value")
ax_csi.set_ylabel("Critical Success Index (CSI)")
ax_csi.set_title("(a) CSI vs. Threshold\n(◆ = Max-CSI optimal point)")
ax_csi.set_xlim(0, 1); ax_csi.set_ylim(0, 1)
ax_csi.legend(loc="upper right", framealpha=0.85)
ax_csi.grid(alpha=0.25, linestyle="--")

# ── Panel B: Youden's J vs Threshold ─────────────────────────────────────────
for model, cd in curve_data.items():
    color = MODEL_COLORS.get(model, "grey")
    n    = len(cd["thresh"])
    step = max(1, n // 500)
    idx  = np.arange(0, n, step)
    ax_J.plot(cd["thresh"][idx], cd["J"][idx],
              color=color, lw=2.0, label=model, alpha=0.9)
    ax_J.scatter(cd["best_thresh_J"], cd["best_J"],
                 color=color, s=MARKER_SIZE, marker="s",
                 edgecolors="black", lw=1.0, zorder=6)
    ax_J.vlines(cd["best_thresh_J"], 0, cd["best_J"],
                colors=color, linestyles=":", lw=1.0, alpha=0.6)

ax_J.set_xlabel("Threshold Value")
ax_J.set_ylabel("Youden's J Statistic")
ax_J.set_title("(b) Youden's J vs. Threshold\n(■ = Youden optimal point)")
ax_J.set_xlim(0, 1); ax_J.set_ylim(0, 1)
ax_J.legend(loc="upper right", framealpha=0.85)
ax_J.grid(alpha=0.25, linestyle="--")

# ── Panel C: Threshold Comparison Grouped Bar ─────────────────────────────────
models = df_summary["Model"].tolist()
x      = np.arange(len(models))
width  = 0.35

bars1 = ax_thr.bar(x - width/2, df_summary["Thresh_CSI"],    width,
                   label="Max-CSI",    color="#2A9D8F", alpha=0.85, edgecolor="black", lw=0.8)
bars2 = ax_thr.bar(x + width/2, df_summary["Thresh_Youden"], width,
                   label="Youden's J", color="#E63946", alpha=0.85, edgecolor="black", lw=0.8)

annotate_bars(ax_thr, bars1, "#1a6b5c")
annotate_bars(ax_thr, bars2, "#9b1c1c")

ax_thr.set_xticks(x)
ax_thr.set_xticklabels(models)
ax_thr.set_ylim(0, 1.22)          # headroom for annotation text
ax_thr.set_ylabel("Optimal Threshold")
ax_thr.set_title("(c) Optimal Threshold: Max-CSI vs. Youden")
ax_thr.legend(framealpha=0.85)
ax_thr.grid(axis="y", alpha=0.25, linestyle="--")

# ── Panel D: CSI at Both Optimal Thresholds (apples-to-apples) ───────────────
bars3 = ax_sc.bar(x - width/2, df_summary["Max_CSI"],       width,
                  label="CSI @ Max-CSI threshold", color="#2A9D8F", alpha=0.85, edgecolor="black", lw=0.8)
bars4 = ax_sc.bar(x + width/2, df_summary["CSI_at_Youden"], width,
                  label="CSI @ Youden threshold",  color="#E63946", alpha=0.85, edgecolor="black", lw=0.8)

annotate_bars(ax_sc, bars3, "#1a6b5c")
annotate_bars(ax_sc, bars4, "#9b1c1c")

ax_sc.set_xticks(x)
ax_sc.set_xticklabels(models)
ax_sc.set_ylim(0, 1.22)
ax_sc.set_ylabel("CSI Score")
ax_sc.set_title("(d) CSI at Max-CSI Threshold vs. Youden Threshold\n"
                "(same metric — higher bar = better threshold choice for CSI)")
ax_sc.legend(framealpha=0.85)
ax_sc.grid(axis="y", alpha=0.25, linestyle="--")

# ── Super-title ───────────────────────────────────────────────────────────────
fig.suptitle("Max-CSI Threshold Optimization for Flood Susceptibility Classification", y=0.97)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "Fig_MaxCSI_Threshold_Analysis.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\n✅ Figure saved: {out_path}")

csv_path = os.path.join(OUT_DIR, "MaxCSI_vs_Youden_Summary.csv")
df_summary.to_csv(csv_path, index=False)
print(f"✅ CSV saved:    {csv_path}")
print("\nSummary table:")
print(df_summary.to_string(index=False))

plt.close()