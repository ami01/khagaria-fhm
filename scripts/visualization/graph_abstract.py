import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

W_in, H_in = 190/25.4, 85/25.4
fig = plt.figure(figsize=(W_in, H_in), dpi=300)
fig.patch.set_facecolor("white")

C = {"blue":"#2166AC","green":"#4DAC26","mgnt":"#D01C8B",
     "salm":"#E08060","red":"#B2182B","purp":"#762A83","dg":"#333333"}

# Four inset axes: [left, bottom, width, height] in figure fraction
ax_a = fig.add_axes([0.01, 0.10, 0.20, 0.78])   # Study area
ax_b = fig.add_axes([0.23, 0.10, 0.18, 0.78])   # Pipeline
ax_c = fig.add_axes([0.44, 0.13, 0.28, 0.72])   # Bar chart
ax_d = fig.add_axes([0.74, 0.10, 0.25, 0.78])   # CSI curve + findings

for ax in [ax_a, ax_b, ax_d]:
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

# ── Title ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97,
    "Flood Hazard Mapping · Khagaria, Bihar · ML vs DL Comparison",
    ha="center", va="top", fontsize=7.5, fontweight="bold", color=C["dg"])

# ══════════════════════════════════════════
#  A — Study Area
# ══════════════════════════════════════════
ax_a.text(0.5, 0.97, "A  Study Area", ha="center", va="top",
          fontsize=7, fontweight="bold", color=C["blue"])
# Bihar
bihar = np.array([[0.05,0.28],[0.08,0.72],[0.25,0.80],[0.45,0.72],
                  [0.50,0.50],[0.40,0.28],[0.20,0.20],[0.05,0.28]])
ax_a.fill(bihar[:,0], bihar[:,1], color="#D6E8F7", zorder=1)
ax_a.plot(bihar[:,0], bihar[:,1], color=C["blue"], lw=0.7, zorder=2)
ax_a.text(0.28, 0.55, "Bihar", ha="center", fontsize=5.5, color=C["blue"])
# Khagaria
khag = np.array([[0.22,0.38],[0.28,0.47],[0.38,0.47],[0.40,0.38],[0.22,0.38]])
ax_a.fill(khag[:,0], khag[:,1], color=C["red"], alpha=0.75, zorder=3)
ax_a.text(0.31, 0.43, "Khagaria\n1 486 km²", ha="center",
          fontsize=5, color="white", fontweight="bold", zorder=4)
# Rivers
riv = [((0.15,0.78),(0.25,0.45)),((0.25,0.78),(0.30,0.47)),
       ((0.35,0.78),(0.36,0.47)),((0.08,0.65),(0.23,0.42)),
       ((0.42,0.62),(0.39,0.42))]
for (x0,y0),(x1,y1) in riv:
    ax_a.annotate("",xy=(x1,y1),xytext=(x0,y0),
        arrowprops=dict(arrowstyle="-",color="#2166AC",lw=0.6,
                        connectionstyle="arc3,rad=0.2"))
ax_a.text(0.28, 0.14, "7 rivers · alluvial plain",
          ha="center", fontsize=5, color=C["blue"])
# India inset
ax_a.add_patch(FancyBboxPatch((0.60,0.70),0.36,0.22,
    boxstyle="round,pad=0.02",linewidth=0.5,
    edgecolor=C["blue"],facecolor="#EBF3FB",zorder=2))
ax_a.text(0.78, 0.84, "India", ha="center", fontsize=4.5, color=C["dg"])
ax_a.add_patch(plt.Circle((0.78,0.77),0.04,color=C["blue"],alpha=0.3,zorder=3))
ax_a.text(0.78, 0.76, "Bihar", ha="center", fontsize=4, color=C["blue"],zorder=4)

# ══════════════════════════════════════════
#  B — Pipeline
# ══════════════════════════════════════════
ax_b.text(0.5, 0.97, "B  Pipeline", ha="center", va="top",
          fontsize=7, fontweight="bold", color=C["mgnt"])

steps = [
    ("Sentinel-1 SAR\nFlood Inventory",   C["blue"],  0.80),
    ("14 CgFs + VIF\nAlignment",          C["green"], 0.63),
    ("Trees: 14 raw\nDL: 12 scaled",      C["mgnt"],  0.46),
    ("RF · XGB · CB\nANN · DNN",          C["salm"],  0.29),
    ("5 FSMs +\nMax-CSI FHM",             C["purp"],  0.12),
]
bw, bh = 0.82, 0.13
for lbl, col, yc in steps:
    ax_b.add_patch(FancyBboxPatch((0.09, yc - bh/2), bw, bh,
        boxstyle="round,pad=0.03", linewidth=0.8,
        edgecolor=col, facecolor=col, zorder=3))
    ax_b.text(0.5, yc, lbl, ha="center", va="center",
              fontsize=6.5, color="white", fontweight="bold", zorder=4)

for i in range(len(steps)-1):
    y_top = steps[i][2] - bh/2
    y_bot = steps[i+1][2] + bh/2
    ax_b.annotate("", xy=(0.5, y_top - 0.005), xytext=(0.5, y_bot + 0.005),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.7, mutation_scale=6))

# ══════════════════════════════════════════
#  C — Bar chart
# ══════════════════════════════════════════
models = ["XGB","CB","RF","DNN","ANN"]
aucs   = [0.998, 0.994, 0.991, 0.962, 0.952]
csis   = [0.937, 0.871, 0.844, 0.705, 0.668]
cols_m = [C["blue"],C["green"],C["mgnt"],C["salm"],C["red"]]

x = np.arange(len(models))
w = 0.35
ax_c.bar(x-w/2, aucs, w, color=cols_m, alpha=0.90)
ax_c.bar(x+w/2, csis, w, color=cols_m, alpha=0.42,
         edgecolor=cols_m, linewidth=0.6)
ax_c.set_ylim(0.60, 1.04)
ax_c.set_xticks(x); ax_c.set_xticklabels(models, fontsize=7)
ax_c.set_yticks([0.6,0.7,0.8,0.9,1.0])
ax_c.set_yticklabels(["0.6","0.7","0.8","0.9","1.0"], fontsize=6)
ax_c.set_ylabel("Score", fontsize=6.5)
ax_c.set_title("C  Model Performance", fontsize=7, fontweight="bold",
               color=C["red"], pad=4)
ax_c.spines[["top","right"]].set_visible(False)
ax_c.spines[["left","bottom"]].set_linewidth(0.5)
ax_c.tick_params(length=2, width=0.5)
# annotate top bars
for i,(a,c) in enumerate(zip(aucs,csis)):
    ax_c.text(i-w/2, a+0.004, f"{a:.3f}", ha="center", fontsize=4.8,
              color=cols_m[i], fontweight="bold")
    ax_c.text(i+w/2, c+0.004, f"{c:.3f}", ha="center", fontsize=4.8,
              color=cols_m[i])
# legend
h1 = mpatches.Patch(color="grey", alpha=0.9, label="AUC")
h2 = mpatches.Patch(color="grey", alpha=0.4, label="CSI")
ax_c.legend(handles=[h1,h2], fontsize=6, frameon=False,
            loc="lower left", handlelength=1)

# ══════════════════════════════════════════
#  D — CSI curve + Key findings
# ══════════════════════════════════════════
ax_d.text(0.5, 0.97, "D  Max-CSI Thresholding", ha="center", va="top",
          fontsize=7, fontweight="bold", color=C["purp"])

# CSI inset
ax_csi = fig.add_axes([0.75, 0.56, 0.23, 0.30])
tau = np.linspace(0.05, 0.95, 300)
c_xgb = 0.94*np.exp(-((tau-0.491)/0.17)**2)
c_dnn = 0.71*np.exp(-((tau-0.284)/0.20)**2)
ax_csi.plot(tau, c_xgb, color=C["blue"], lw=1.2, label="XGBoost")
ax_csi.plot(tau, c_dnn, color=C["salm"], lw=1.0, ls="--", label="DNN")
ax_csi.axvline(0.491, color=C["blue"], lw=0.7, ls=":", alpha=0.7)
ax_csi.axvline(0.284, color=C["salm"], lw=0.7, ls=":", alpha=0.7)
ax_csi.axvline(0.500, color="#999",    lw=0.5, ls="--", alpha=0.5)
ax_csi.set_xlabel("τ", fontsize=6)
ax_csi.set_ylabel("CSI", fontsize=6)
ax_csi.set_ylim(0, 1.05)
ax_csi.tick_params(labelsize=5, length=2, width=0.4)
ax_csi.spines[["top","right"]].set_visible(False)
ax_csi.spines[["left","bottom"]].set_linewidth(0.4)
ax_csi.legend(fontsize=5.5, frameon=False, loc="upper right",
              handlelength=1.2, borderpad=0.2)
ax_csi.set_title("CSI vs Threshold", fontsize=5.8, pad=2)

# Key findings
findings = [
    "✔  XGBoost: AUC = 0.998, CSI = 0.937",
    "✔  Tree > DL:  ΔCSI = +0.27",
    "✔  Max-CSI > Youden by 0.008–0.017",
    "✔  Optimal τ* = 0.491  (XGBoost)",
    "✔  Pixel-level FHM → DDMA Khagaria",
]
ax_d.text(0.04, 0.51, "Key Findings", fontsize=6.5, fontweight="bold",
          color=C["purp"], va="top")
for i, line in enumerate(findings):
    ax_d.text(0.04, 0.43 - i*0.087, line, fontsize=6, color=C["dg"],
              va="top", fontfamily="monospace")
# box around findings
ax_d.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.45,
    boxstyle="round,pad=0.02", linewidth=0.7,
    edgecolor=C["purp"], facecolor="#F5EEF8", zorder=0))

# vertical dividers
for xv in [0.22, 0.435, 0.73]:
    fig.add_artist(plt.Line2D([xv, xv], [0.07, 0.93],
                              color="#CCCCCC", lw=0.5, transform=fig.transFigure))

plt.savefig("graphical_abstract.pdf",
            bbox_inches="tight", dpi=300)
plt.savefig("graphical_abstract.png",
            bbox_inches="tight", dpi=300)
plt.close()
print("✓ graphical_abstract fixed")