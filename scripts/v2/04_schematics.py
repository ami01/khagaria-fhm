"""v2 methodology flowchart and graphical abstract (numbers read from results_v2/). Run from the repo root."""
import json

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

RES = "results_v2"
OUT = f"{RES}/figures"
cfg = json.load(open(f"{RES}/run_config.json"))
prep = json.load(open(f"{RES}/prepare_log.json"))
met = pd.read_csv(f"{RES}/metrics/test_metrics_maxcsi.csv").set_index("Model")
cmp_ = pd.read_csv(f"{RES}/metrics/maxcsi_vs_youden.csv").set_index("Model")
area = pd.read_csv(f"{RES}/metrics/area_statistics.csv").set_index("Model")
vif_max = pd.read_csv(f"{RES}/metrics/VIF.csv").VIF.max()
ORDER = ["XGBoost", "CatBoost", "RandomForest", "DNN", "ANN"]
C = {"blue": "#3A6BB0", "green": "#5FA03A", "mag": "#B0287F", "orange": "#C9765A", "red": "#A3262E", "purple": "#6A3D8F"}


def box(ax, x, y, w, h, text, color, fs=10, tc="white", bold=True, style="round,pad=0.012"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def group(ax, x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005", fc=color + "14", ec=color, ls="--", lw=1.4))
    ax.text(x + w / 2, y + h - 0.012, title, ha="center", va="top", color=color, fontsize=12, fontweight="bold")


def arrow(ax, a, b, text=None, rad=0.0):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k",
                                                    connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + 0.005, (a[1] + b[1]) / 2, text, fontsize=9.5, ha="left", va="center")


# ------------------------------------------------ flowchart
fig = plt.figure(figsize=(16, 8.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

group(ax, 0.01, 0.80, 0.50, 0.19, "① Data & inventory", C["blue"])
box(ax, 0.025, 0.825, 0.14, 0.11, "SAR flood extents\n2008 · 2020 · 2021\n(ENVISAT ASAR, S-1)", C["blue"], 9.5)
box(ax, 0.195, 0.825, 0.13, 0.11, "Union → binary\ninventory\nflood / non-flood", C["blue"], 9.5)
box(ax, 0.355, 0.825, 0.145, 0.11, f"All district pixels\n{prep['district_cells'] / 1e6:.2f} M · flood {prep['label']['flood_cells'] / prep['district_cells'] * 100:.1f}%\n70 / 10 / 20 split", C["blue"], 9.5)
arrow(ax, (0.165, 0.88), (0.195, 0.88))
arrow(ax, (0.325, 0.88), (0.355, 0.88))

group(ax, 0.01, 0.55, 0.50, 0.22, "② 14 CgFs, district mask, audit", C["green"])
box(ax, 0.025, 0.575, 0.21, 0.14, "Elevation · Slope · Aspect · Curvature\nTWI · SPI · Dist. river · Stream density\nLULC · Geomorphology · Soil\nLithology · Rainfall · Dist. roads", C["green"], 9)
box(ax, 0.265, 0.575, 0.235, 0.14, "30 m grid, UTM 45N, district mask\nSlope recomputed from DEM\n(supplied layer was a copy of TWI)\nsoil fill code → nearest class", C["green"], 9)
arrow(ax, (0.235, 0.645), (0.265, 0.645))

group(ax, 0.01, 0.30, 0.50, 0.22, "③ Collinearity & model training", C["mag"])
box(ax, 0.025, 0.37, 0.15, 0.10, f"VIF (50,000 px)\nmax VIF = {vif_max:.2f}\nall 14 CgFs kept", C["mag"], 9.5)
box(ax, 0.20, 0.42, 0.30, 0.06, "Trees on raw values: RF · XGBoost · CatBoost", C["orange"], 9.5)
box(ax, 0.20, 0.33, 0.30, 0.06, "Neural nets on z-scores: ANN · DNN", C["orange"], 9.5)
arrow(ax, (0.175, 0.44), (0.20, 0.45))
arrow(ax, (0.175, 0.40), (0.20, 0.36))

group(ax, 0.01, 0.03, 0.50, 0.24, "④ Validation split (10%)", C["purple"])
box(ax, 0.025, 0.06, 0.225, 0.15, "XGBoost early stopping\nMax-CSI threshold\nτ* = argmax CSI(τ)\nCSI = TP / (TP+FP+FN)", C["purple"], 9.5)
box(ax, 0.275, 0.06, 0.225, 0.15, "Youden threshold\nJ = TPR − FPR\n(comparison)", C["purple"], 9.5)

group(ax, 0.54, 0.55, 0.45, 0.44, "⑤ Independent test set (20%) + maps", C["red"])
box(ax, 0.56, 0.80, 0.19, 0.12, "AUC · CSI · sensitivity\nspecificity · accuracy\ncalibration", C["red"], 9.5)
box(ax, 0.78, 0.80, 0.19, 0.12, "SHAP: 5,000 balanced\ntest pixels, all models", C["red"], 9.5)
box(ax, 0.56, 0.60, 0.19, 0.14, "5 susceptibility maps\nP̂ ∈ [0,1] · 30 m\nnatural breaks (5 classes)", C["red"], 9.5)
box(ax, 0.78, 0.60, 0.19, 0.14, "5 binary hazard maps\nP̂ ≥ τ*\n+ model agreement", C["red"], 9.5)

group(ax, 0.54, 0.03, 0.45, 0.47, "Key v2 results (test set)", "#444444")
rows = [f"{m:<13s} AUC {met.loc[m, 'AUC']:.3f}   CSI {met.loc[m, 'CSI']:.3f}   τ* {met.loc[m, 'Threshold']:.3f}" for m in ORDER]
ax.text(0.765, 0.33, "\n".join(rows), ha="center", va="center", family="monospace", fontsize=10.5, linespacing=1.6)
ax.text(0.765, 0.12, f"Max-CSI vs Youden: ΔCSI +{cmp_.dCSI.min():.3f} to +{cmp_.dCSI.max():.3f}\n"
                     f"XGBoost hazard area {area.loc['XGBoost', 'Hazard_km2']:.0f} km² "
                     f"({area.loc['XGBoost', 'Hazard_pct']:.1f}% of district)",
        ha="center", va="center", fontsize=10.5)
arrow(ax, (0.43, 0.825), (0.43, 0.715))
arrow(ax, (0.38, 0.575), (0.30, 0.48))
arrow(ax, (0.35, 0.33), (0.35, 0.21))
arrow(ax, (0.51, 0.135), (0.655, 0.60), rad=0.35)
fig.savefig(f"{OUT}/FHM_Khagaria_Methodology_Flowchart.png", dpi=300, facecolor="white")
fig.savefig(f"{OUT}/FHM_Khagaria_Methodology_Flowchart.pdf", facecolor="white")
plt.close(fig)
print("wrote flowchart")

# ------------------------------------------------ graphical abstract
import rasterio

fig = plt.figure(figsize=(12, 5.2))
fig.text(0.5, 0.97, "Flood susceptibility and hazard zonation · Khagaria, North Bihar · five models, Max-CSI thresholds",
         ha="center", va="top", fontsize=13, fontweight="bold")
ax = fig.add_axes([0.0, 0.03, 0.30, 0.86])
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
steps = [("Multi-mission SAR inventory\n2008 · 2020 · 2021 (824 km² flooded)", C["blue"]),
         ("14 CgFs at 30 m · district mask\nreal slope from DEM · max VIF " + f"{vif_max:.1f}", C["green"]),
         ("RF · XGBoost · CatBoost · ANN · DNN\n70 / 10 / 20 split", C["orange"]),
         ("Max-CSI threshold on validation\nmetrics on independent test set", C["purple"]),
         ("Susceptibility + hazard maps\nSHAP for all five models", C["red"])]
for k, (t, c) in enumerate(steps):
    y = 0.82 - k * 0.19
    box(ax, 0.04, y, 0.92, 0.14, t, c, 9.5)
    if k:
        arrow(ax, (0.5, y + 0.19), (0.5, y + 0.14))

ax2 = fig.add_axes([0.31, 0.08, 0.31, 0.80])
with rasterio.open(f"{RES}/maps/binary/Binary_XGBoost_MaxCSI.tif") as r:
    a = r.read(1, masked=True)
    b = r.bounds
from matplotlib.colors import ListedColormap
import geopandas as gpd

ax2.imshow(a, cmap=ListedColormap(["#F4F4F4", "#D7191C"]), vmin=0, vmax=1, extent=(b.left, b.right, b.bottom, b.top))
gpd.read_file("data/vectors/Khagaria_Boundary_UTM.shp").boundary.plot(ax=ax2, color="k", lw=1)
ax2.set_axis_off()
ax2.set_title(f"XGBoost flood hazard map (τ* = {met.loc['XGBoost', 'Threshold']:.3f})\n"
              f"{area.loc['XGBoost', 'Hazard_km2']:.0f} km² = {area.loc['XGBoost', 'Hazard_pct']:.1f}% of the district",
              fontsize=10.5)

ax3 = fig.add_axes([0.745, 0.12, 0.235, 0.72])
y = range(len(ORDER))[::-1]
ax3.barh(list(y), met.loc[ORDER, "CSI"], color=["#D62728", "#E6A817", "#2A9D8F", "#8A3FFC", "#4E79A7"], edgecolor="k")
for yy, m in zip(y, ORDER):
    ax3.text(met.loc[m, "CSI"] + 0.005, yy, f"{met.loc[m, 'CSI']:.3f}  (AUC {met.loc[m, 'AUC']:.3f})", va="center", fontsize=9.5)
ax3.set_yticks(list(y), ORDER)
ax3.set_xlim(0.6, 1.08)
ax3.set_xlabel("CSI on independent test set")
ax3.set_title("Trees > neural networks\nSHAP top 4: Rainfall · Geomorphology · Soil · Elevation", fontsize=10.5)
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
fig.savefig(f"{OUT}/Graphical_Abstract.png", dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(f"{OUT}/Graphical_Abstract.pdf", facecolor="white", bbox_inches="tight")
print("wrote graphical abstract")
