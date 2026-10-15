"""v2 manuscript figures from results_v2/ (run after 02_train_evaluate.py, from the repo root).

Writes results_v2/figures/*.png with the same names the manuscript uses.
Map panels: district boundary from data/vectors, UTM 45N.
"""
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

RES = "results_v2"
OUT = f"{RES}/figures"
os.makedirs(OUT, exist_ok=True)
ORDER = ["XGBoost", "CatBoost", "RandomForest", "ANN", "DNN"]  # manuscript panel order (A)-(E)
COL = {"XGBoost": "#D62728", "CatBoost": "#E6A817", "RandomForest": "#2A9D8F", "ANN": "#4E79A7", "DNN": "#8A3FFC"}
CLASSES = ["Very Low", "Low", "Moderate", "High", "Very High"]
SUSC = ["#1A9641", "#A6D96A", "#FFFFBF", "#FDAE61", "#D7191C"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
cfg = json.load(open(f"{RES}/run_config.json"))
met = pd.read_csv(f"{RES}/metrics/test_metrics_all_thresholds.csv")
area = pd.read_csv(f"{RES}/metrics/area_statistics.csv").set_index("Model")
boundary = gpd.read_file("data/vectors/Khagaria_Boundary_UTM.shp")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------ Max-CSI threshold analysis (2x2, as in v1)
fig, ax = plt.subplots(2, 2, figsize=(15, 11.5))
for m in ORDER:
    c = pd.read_csv(f"{RES}/curves/threshold_curve_{m}_test.csv")
    j = c.Sensitivity + c.Specificity - 1
    ax[0, 0].plot(c.threshold, c.CSI, color=COL[m], lw=2, label=m)
    ax[0, 1].plot(c.threshold, j, color=COL[m], lw=2, label=m)
    t1, t2 = cfg["thresholds"][m]["MaxCSI"], cfg["thresholds"][m]["Youden"]
    k1, k2 = (c.threshold - t1).abs().idxmin(), (c.threshold - t2).abs().idxmin()
    ax[0, 0].scatter(t1, c.CSI[k1], marker="D", s=80, color=COL[m], edgecolor="k", zorder=5)
    ax[0, 1].scatter(t2, j[k2], marker="s", s=80, color=COL[m], edgecolor="k", zorder=5)
for a, yl, t in [(ax[0, 0], "Critical Success Index (CSI)", "(a) CSI vs. threshold (test set)\n◆ = Max-CSI threshold (chosen on validation set)"),
                 (ax[0, 1], "Youden's J", "(b) Youden's J vs. threshold (test set)\n■ = Youden threshold (chosen on validation set)")]:
    a.set(xlim=(0, 1), ylim=(0, 1), xlabel="Threshold", ylabel=yl)
    a.set_title(t, fontweight="bold")
    a.grid(alpha=0.3, ls="--")
    a.legend(loc="lower center", ncol=3, frameon=False)
x = np.arange(len(ORDER))
mc = met[met.Threshold_method == "MaxCSI"].set_index("Model").loc[ORDER]
yo = met[met.Threshold_method == "Youden"].set_index("Model").loc[ORDER]
for a, col, yl, t in [(ax[1, 0], "Threshold", "Threshold", "(c) Max-CSI vs. Youden threshold"),
                      (ax[1, 1], "CSI", "CSI (test set)", "(d) Test CSI at each threshold")]:
    b1 = a.bar(x - 0.18, mc[col], 0.36, color="#2A9D8F", edgecolor="k", label="Max-CSI")
    b2 = a.bar(x + 0.18, yo[col], 0.36, color="#E76F51", edgecolor="k", label="Youden's J")
    for bars in (b1, b2):
        for r in bars:
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015, f"{r.get_height():.3f}",
                   ha="center", va="bottom", rotation=90, fontsize=9.5)
    a.set_xticks(x, ORDER)
    a.set(ylim=(0, 1.2), ylabel=yl)
    a.set_title(t, fontweight="bold")
    a.legend(frameon=False, loc="upper right")
    a.grid(axis="y", alpha=0.3, ls="--")
save(fig, "Fig_MaxCSI_Threshold_Analysis.png")

# ------------------------------------------------ ROC train / test
for split, name, title in [("train", "Training__ROC.png", "Training set"), ("test", "Validation_ROC.png", "Independent test set")]:
    auc_tab = pd.read_csv(f"{RES}/metrics/auc_train_val_test.csv").set_index("Model")
    fig, a = plt.subplots(figsize=(7, 6.5))
    for m in ORDER:
        c = pd.read_csv(f"{RES}/curves/roc_{m}_{split}.csv")
        a.plot(c.fpr, c.tpr, color=COL[m], lw=2, label=f"{m} (AUC = {auc_tab.loc[m, f'{split}_AUC']:.4f})")
    a.plot([0, 1], [0, 1], "k--", lw=1)
    a.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate")
    a.set_title(f"ROC curves: {title}", fontweight="bold")
    a.legend(loc="lower right", frameon=False)
    a.grid(alpha=0.3, ls="--")
    save(fig, name)

# ------------------------------------------------ correlation matrix
C = pd.read_csv(f"{RES}/metrics/correlation_matrix.csv", index_col=0)
fig, a = plt.subplots(figsize=(10, 8.5))
im = a.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
a.set_xticks(range(len(C)), C.columns, rotation=45, ha="right")
a.set_yticks(range(len(C)), C.index)
for i in range(len(C)):
    for j in range(len(C)):
        a.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
               color="white" if abs(C.values[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=a, shrink=0.8, label="Pearson r")
a.set_title("Correlation of the 14 conditioning factors (50,000 training pixels)", fontweight="bold")
save(fig, "Correlation_Matrix_Robust.png")


# ------------------------------------------------ maps
def read(path):
    with rasterio.open(path) as r:
        a = r.read(1, masked=True)
        b = r.bounds
    return a, (b.left, b.right, b.bottom, b.top)


def panel_axes():
    fig = plt.figure(figsize=(16, 9.6))
    axes = [fig.add_axes([0.00 + 0.33 * k, 0.50, 0.32, 0.48]) for k in range(3)] + \
           [fig.add_axes([0.00 + 0.33 * k, 0.01, 0.32, 0.48]) for k in range(2)]
    leg = fig.add_axes([0.70, 0.05, 0.28, 0.40])
    leg.axis("off")
    return fig, axes, leg


def decorate(a, letter, text):
    boundary.boundary.plot(ax=a, color="k", lw=1.2)
    a.set_axis_off()
    a.text(0.02, 0.02, f"({letter})", transform=a.transAxes, fontsize=15, fontweight="bold")
    a.text(0.02, 0.17, text, transform=a.transAxes, fontsize=11, family="monospace")


def scalebar_north(leg):
    leg.annotate("N", xy=(0.82, 0.62), xytext=(0.82, 0.40), ha="center", fontsize=14, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", lw=2, color="k"), xycoords="axes fraction")
    # 30 km bar: map axes are ~ 58 km wide over 0.32 fig width; legend ax is 0.28 fig width
    w = 30 / 58.0 * 0.32 / 0.28
    for k in range(3):
        leg.add_patch(plt.Rectangle((0.05 + k * w / 3, 0.12), w / 3, 0.035, transform=leg.transAxes,
                                    facecolor="k" if k != 1 else "white", edgecolor="k"))
    for k, lbl in enumerate(["0", "10", "20", "30 km"]):
        leg.text(0.05 + k * w / 3, 0.18, lbl, transform=leg.transAxes, ha="center")


# FSM classified
fig, axes, leg = panel_axes()
cmap, norm = ListedColormap(SUSC), BoundaryNorm(np.arange(-0.5, 5.5), 5)
br = cfg["natural_breaks"]
for k, m in enumerate(ORDER):
    a, ext = read(f"{RES}/maps/classified/Classified_{m}_NaturalBreaks.tif")
    axes[k].imshow(a, cmap=cmap, norm=norm, extent=ext, interpolation="nearest")
    decorate(axes[k], "ABCDE"[k], f"Very High: {area.loc[m, 'Very High (%)']:.1f}%")
edges = [0.0] + br + [1.0]
leg.legend(handles=[Patch(facecolor=SUSC[i], edgecolor="k", label=f"{CLASSES[i]} ({edges[i]:.2f}–{edges[i + 1]:.2f})")
                    for i in range(5)] + [Patch(facecolor="none", edgecolor="k", lw=1.5, label="Khagaria district")],
           loc="upper left", title="Susceptibility (natural breaks)", frameon=False, fontsize=11, title_fontsize=12)
scalebar_north(leg)
save(fig, "Fig_Flood_Susceptibility_Maps.png")

# FHM binary
fig, axes, leg = panel_axes()
cmap2 = ListedColormap(["white", "#D7191C"])
for k, m in enumerate(ORDER):
    a, ext = read(f"{RES}/maps/binary/Binary_{m}_MaxCSI.tif")
    axes[k].imshow(a, cmap=cmap2, vmin=0, vmax=1, extent=ext, interpolation="nearest")
    decorate(axes[k], "ABCDE"[k], f"{area.loc[m, 'Hazard_km2']:.2f} sq.km.\nτ* = {area.loc[m, 'MaxCSI_threshold']:.3f}")
leg.legend(handles=[Patch(facecolor="white", edgecolor="k", label="No flood"),
                    Patch(facecolor="#D7191C", edgecolor="k", label="Flood hazard"),
                    Patch(facecolor="none", edgecolor="k", lw=1.5, label="Khagaria district")],
           loc="upper left", title="Classes", frameon=False, fontsize=12, title_fontsize=13)
scalebar_north(leg)
save(fig, "FHM.png")

# ------------------------------------------------ SHAP
shap_df = pd.read_csv(f"{RES}/shap/mean_abs_shap.csv")
FEAT = cfg["features"]
piv = shap_df.pivot(index="Feature", columns="Model", values="mean_abs_SHAP")[ORDER]
norm_ = (piv - piv.min()) / (piv.max() - piv.min())
order_f = norm_.mean(1).sort_values().index
fig, axs = plt.subplots(1, 5, figsize=(20, 6), sharey=True)
for a, m in zip(axs, ORDER):
    a.barh(order_f, norm_.loc[order_f, m], color=COL[m], edgecolor="k", lw=0.5)
    a.set_title(m, fontweight="bold")
    a.set_xlim(0, 1.05)
    a.set_xlabel("normalised mean |SHAP|")
    a.grid(axis="x", alpha=0.3, ls="--")
fig.suptitle("Feature importance (mean |SHAP|, 5,000 balanced test pixels)", fontweight="bold", y=1.02)
save(fig, "Fig_SHAP_All_Models.png")

rank = shap_df.pivot(index="Feature", columns="Model", values="rank")[ORDER]
rank = rank.loc[rank.mean(1).sort_values().index]
fig, a = plt.subplots(figsize=(8, 8))
im = a.imshow(rank.values, cmap="viridis_r", vmin=1, vmax=len(FEAT))
a.set_xticks(range(5), ORDER)
a.set_yticks(range(len(rank)), rank.index)
for i in range(len(rank)):
    for j in range(5):
        a.text(j, i, int(rank.values[i, j]), ha="center", va="center",
               color="white" if rank.values[i, j] < 8 else "black", fontsize=10)
fig.colorbar(im, ax=a, shrink=0.7, label="SHAP rank (1 = most important)")
a.set_title("SHAP rank by model", fontweight="bold")
save(fig, "Fig_SHAP_Rank_Heatmap.png")

Xs = pd.read_csv(f"{RES}/shap/shap_sample_features.csv")[FEAT]
for m in ORDER:
    sv = np.load(f"{RES}/shap/shap_values_{m}.npy")
    o = np.argsort(np.abs(sv).mean(0))
    fig, a = plt.subplots(figsize=(9, 7))
    r = np.random.default_rng(0)
    for row, j in enumerate(o):
        v = Xs.iloc[:, j].values
        lo, hi = np.nanpercentile(v, [5, 95])
        cval = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        a.scatter(sv[:, j], row + r.normal(0, 0.12, len(v)), c=cval, cmap="coolwarm", s=4, alpha=0.6, lw=0)
    a.set_yticks(range(len(o)), [FEAT[j] for j in o])
    a.axvline(0, color="grey", lw=0.8)
    a.set_xlabel("SHAP value (impact on model output)")
    a.set_title(f"{m}: SHAP summary (5,000 balanced test pixels)", fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    cb = fig.colorbar(sm, ax=a, shrink=0.6, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Feature value")
    save(fig, f"Fig_SHAP_Beeswarm_{m}.png")

# ------------------------------------------------ training curves (supplement / visualisation)
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
h = pd.read_csv(f"{RES}/training/history_XGBoost.csv")
axs[0].plot(h["round"], h.train_auc, label="train", color="k")
axs[0].plot(h["round"], h.val_auc, label="validation", color=COL["XGBoost"])
axs[0].set(title="XGBoost (early stopping on validation)", xlabel="boosting round", ylabel="AUC")
h = pd.read_csv(f"{RES}/training/history_CatBoost.csv")
axs[1].plot(h.iteration, h.val_auc, color=COL["CatBoost"])
axs[1].set(title="CatBoost (validation AUC)", xlabel="iteration", ylabel="AUC")
for m in ["ANN", "DNN"]:
    h = pd.read_csv(f"{RES}/training/history_{m}.csv")
    axs[2].plot(h.epoch, h.train_loss, color=COL[m], ls="--", label=f"{m} train")
    axs[2].plot(h.epoch, h.val_loss, color=COL[m], label=f"{m} validation")
axs[2].set(title="ANN / DNN (binary cross-entropy)", xlabel="epoch", ylabel="loss")
for a in axs:
    a.grid(alpha=0.3, ls="--")
    a.legend(frameon=False) if a.get_legend_handles_labels()[0] else None
save(fig, "Fig_Training_Curves.png")
