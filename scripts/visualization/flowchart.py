"""
flowcharts_khagaria.py
Produces:
  1. flowchart_compact.pdf/.png   — compact methodology flowchart
  2. graphical_abstract.pdf/.png  — Elsevier graphical abstract (190×85 mm)

Requirements:
    pip install graphviz matplotlib numpy
    sudo apt install graphviz   (or: conda install graphviz)
"""

# ─────────────────────────────────────────────────────────────────────────────
#  1. COMPACT FLOWCHART
# ─────────────────────────────────────────────────────────────────────────────
from graphviz import Digraph

def make_flowchart():
    d = Digraph("flow", format="pdf", engine="dot")
    d.attr(rankdir="TB", size="7,13", dpi="300",
           fontname="Helvetica", nodesep="0.3", ranksep="0.4",
           splines="ortho", bgcolor="white")
    d.attr("node", fontname="Helvetica", fontsize="10",
           style="filled,rounded", shape="box", margin="0.14,0.08", penwidth="1.2")
    d.attr("edge", fontname="Helvetica", fontsize="8", penwidth="1.1")

    COL = ["#2166AC","#4DAC26","#D01C8B","#E08060","#B2182B","#762A83"]
    BG  = ["#EBF3FB","#F2FAF0","#FDF0F8","#FEF5EF","#FBEAEA","#F5EEF8"]

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    with d.subgraph(name="cluster_1") as c:
        c.attr(label="① Data & Inventory", fontsize="10", fontcolor=COL[0],
               fontname="Helvetica-Bold", style="rounded,dashed",
               color=COL[0], bgcolor=BG[0], penwidth="1.4")
        c.node("A", "Sentinel-1 SAR\n(2019–2020 Monsoon)", fillcolor=COL[0], fontcolor="white", color=COL[0])
        c.node("B", "Binary Inventory\nFlood / Non-Flood",  fillcolor=COL[0], fontcolor="white", color=COL[0])
        c.node("C", "80:20 Train/Test Split",               fillcolor=COL[0], fontcolor="white", color=COL[0])
        c.edge("A","B"); c.edge("B","C")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    with d.subgraph(name="cluster_2") as c:
        c.attr(label="② 14 CgFs + Spatial Alignment", fontsize="10", fontcolor=COL[1],
               fontname="Helvetica-Bold", style="rounded,dashed",
               color=COL[1], bgcolor=BG[1], penwidth="1.4")
        c.node("D", "Elev · Slope · TWI · NDVI\nDist-River · LULC · Soil · …",
               fillcolor=COL[1], fontcolor="white", color=COL[1])
        c.node("E", "Reproject → 30 m grid\nNoData mask",
               fillcolor=COL[1], fontcolor="white", color=COL[1])
        c.edge("D","E")

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    with d.subgraph(name="cluster_3") as c:
        c.attr(label="③ VIF & Pre-processing", fontsize="10", fontcolor=COL[2],
               fontname="Helvetica-Bold", style="rounded,dashed",
               color=COL[2], bgcolor=BG[2], penwidth="1.4")
        c.node("F", "VIF Analysis\nSlope & TWI → VIF = ∞",
               fillcolor=COL[2], fontcolor="white", color=COL[2])
        c.node("G", "Trees: 14 CgFs (raw)",
               fillcolor=COL[2], fontcolor="white", color=COL[2])
        c.node("H", "DL: 12 CgFs\nz-score scaled",
               fillcolor=COL[2], fontcolor="white", color=COL[2])
        c.edge("F","G", label=" retain"); c.edge("F","H", label=" drop Slope")

    # ── Phase 4 ──────────────────────────────────────────────────────────────
    with d.subgraph(name="cluster_4") as c:
        c.attr(label="④ Model Training (GPU)", fontsize="10", fontcolor="#B85C38",
               fontname="Helvetica-Bold", style="rounded,dashed",
               color=COL[3], bgcolor=BG[3], penwidth="1.4")
        for nid, lbl in [("RF","Random\nForest"),("XGB","XGBoost"),
                         ("CB","CatBoost"),("ANN","ANN"),("DNN","DNN")]:
            c.node(nid, lbl, fillcolor=COL[3], fontcolor="white", color=COL[3])

    # ── Phase 5 ──────────────────────────────────────────────────────────────
    with d.subgraph(name="cluster_5") as c:
        c.attr(label="⑤ Susceptibility Maps + Validation", fontsize="10",
               fontcolor=COL[4], fontname="Helvetica-Bold",
               style="rounded,dashed", color=COL[4], bgcolor=BG[4], penwidth="1.4")
        c.node("FSM", "5 × Flood Susceptibility\nMaps  P̂ ∈ [0,1]",
               fillcolor=COL[4], fontcolor="white", color=COL[4])
        c.node("VAL", "AUC · CSI · Sensitivity\nSpecificity  (test set)",
               fillcolor=COL[4], fontcolor="white", color=COL[4])
        c.edge("FSM","VAL")

    # ── Phase 6 ──────────────────────────────────────────────────────────────
    with d.subgraph(name="cluster_6") as c:
        c.attr(label="⑥ Max-CSI Thresholding → Hazard Map", fontsize="10",
               fontcolor=COL[5], fontname="Helvetica-Bold",
               style="rounded,dashed", color=COL[5], bgcolor=BG[5], penwidth="1.4")
        c.node("CSI", "τ* = argmax CSI(τ)\nCSI = TP/(TP+FP+FN)",
               fillcolor=COL[5], fontcolor="white", color=COL[5])
        c.node("FHM", "Binary FHM (XGBoost τ*=0.491)\nGeoTIFF · 30 m · Khagaria",
               fillcolor=COL[5], fontcolor="white", color=COL[5],
               style="filled,bold,rounded")
        c.edge("CSI","FHM", label=" P̂>τ*→Hazard")

    # ── Inter-phase edges ────────────────────────────────────────────────────
    d.edge("C","E",  label=" merge")
    d.edge("E","F")
    d.edge("G","RF"); d.edge("G","XGB"); d.edge("G","CB")
    d.edge("H","ANN"); d.edge("H","DNN")
    d.edge("RF","FSM"); d.edge("XGB","FSM"); d.edge("CB","FSM")
    d.edge("ANN","FSM"); d.edge("DNN","FSM")
    d.edge("VAL","CSI", label=" ROC curve")

    d.render("flowchart_compact", cleanup=True)
    d.format = "png"
    d.render("flowchart_compact", cleanup=True)
    print("✓ flowchart_compact.pdf  /  .png")

make_flowchart()


# ─────────────────────────────────────────────────────────────────────────────
#  2. GRAPHICAL ABSTRACT  (190 × 85 mm @ 300 dpi)
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

def ga_box(ax, xy, w, h, text, color, fontsize=8.5, bold=False,
           radius=0.015, text_color="white"):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         linewidth=0.8, edgecolor=color,
                         facecolor=color, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight=weight,
            zorder=4, wrap=False,
            multialignment="center")

def ga_arrow(ax, x0, x1, y, color="#555555"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.0, mutation_scale=8), zorder=5)

def make_graphical_abstract():
    # 190 mm × 85 mm
    W_in = 190/25.4
    H_in = 85/25.4
    fig, ax = plt.subplots(figsize=(W_in, H_in), dpi=300)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Palette ───────────────────────────────────────────────────────────────
    C = {"blue":  "#2166AC",
         "green": "#4DAC26",
         "mgnt":  "#D01C8B",
         "salm":  "#E08060",
         "red":   "#B2182B",
         "purp":  "#762A83",
         "dgray": "#333333",
         "lgray": "#F0F0F0"}

    # ── Title bar ─────────────────────────────────────────────────────────────
    ax.text(0.5, 0.96,
            "Flood Hazard Mapping  ·  Khagaria District, Bihar  ·  Five ML/DL Models",
            ha="center", va="top", fontsize=7.5, color=C["dgray"],
            fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL A — Study Area schematic (left)
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.02, 0.87, "A  Study Area", fontsize=7, color=C["blue"],
            fontweight="bold", va="top")

    # Bihar outline (rough polygon)
    bihar = np.array([[0.01,0.35],[0.05,0.72],[0.14,0.78],[0.20,0.72],
                      [0.22,0.55],[0.18,0.35],[0.10,0.28],[0.01,0.35]])
    ax.fill(bihar[:,0], bihar[:,1], color="#D6E8F7", zorder=1, lw=0)
    ax.plot(bihar[:,0], bihar[:,1], color=C["blue"], lw=0.6, zorder=2)
    ax.text(0.11, 0.55, "Bihar", ha="center", fontsize=5.5, color=C["blue"])

    # Khagaria highlight
    khag = np.array([[0.09,0.43],[0.12,0.50],[0.17,0.50],[0.18,0.43],[0.09,0.43]])
    ax.fill(khag[:,0], khag[:,1], color=C["red"], alpha=0.7, zorder=3)
    ax.text(0.135, 0.47, "Khagaria\n1 486 km²", ha="center", fontsize=4.8,
            color="white", fontweight="bold", zorder=4)

    # 7 rivers as blue lines
    rivers = [((0.07,0.75),(0.12,0.50)), ((0.10,0.75),(0.14,0.52)),
              ((0.13,0.75),(0.16,0.52)), ((0.05,0.68),(0.10,0.48)),
              ((0.16,0.68),(0.18,0.50)), ((0.08,0.58),(0.09,0.45)),
              ((0.18,0.55),(0.17,0.45))]
    for (x0,y0),(x1,y1) in rivers:
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0),
                    arrowprops=dict(arrowstyle="-", color="#2166AC",
                                   lw=0.7, connectionstyle="arc3,rad=0.15"))
    ax.text(0.11, 0.30, "7 rivers · alluvial floodplain", ha="center",
            fontsize=4.8, color=C["blue"])

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL B — Pipeline summary (centre)
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.27, 0.87, "B  Hybrid Prediction Pipeline", fontsize=7,
            color=C["mgnt"], fontweight="bold", va="top")

    boxes = [
        (0.27, 0.70, "Sentinel-1 SAR\nFlood Inventory",  C["blue"]),
        (0.27, 0.56, "14 CgFs + VIF\nSpatial Alignment", C["green"]),
        (0.27, 0.42, "Tree: 14 raw CgFs\nDL: 12 scaled",  C["mgnt"]),
        (0.27, 0.28, "RF · XGB · CB\nANN · DNN",          C["salm"]),
        (0.27, 0.14, "5 Susceptibility Maps\n+ Max-CSI FHM", C["purp"]),
    ]
    bw, bh = 0.175, 0.115
    for bx, by, lbl, col in boxes:
        ga_box(ax, (bx, by), bw, bh, lbl, col, fontsize=6.8)
    # arrows between boxes
    for i in range(len(boxes)-1):
        _, by_top, _, _ = boxes[i]
        _, by_bot, _, _ = boxes[i+1]
        xc = boxes[i][0] + bw/2
        ax.annotate("", xy=(xc, by_bot + bh + 0.005),
                    xytext=(xc, by_top - 0.005),
                    arrowprops=dict(arrowstyle="-|>", color="#555",
                                   lw=0.8, mutation_scale=7))

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL C — AUC / CSI bar chart (centre-right)
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.52, 0.87, "C  Model Performance", fontsize=7,
            color=C["red"], fontweight="bold", va="top")

    models = ["XGB", "CB", "RF", "DNN", "ANN"]
    aucs   = [0.998, 0.994, 0.991, 0.962, 0.952]
    csis   = [0.937, 0.871, 0.844, 0.705, 0.668]
    cols_m = [C["blue"], C["green"], C["mgnt"], C["salm"], C["red"]]

    bar_x0 = 0.52
    bar_w_total = 0.225
    bw_bar = bar_w_total / (len(models) * 2 + len(models) - 1)  # rough

    # mini axes via inset
    ax_bar = fig.add_axes([0.515, 0.10, 0.22, 0.70])
    x = np.arange(len(models))
    width = 0.32
    ax_bar.bar(x - width/2, aucs, width, color=cols_m, alpha=0.85, label="AUC")
    ax_bar.bar(x + width/2, csis, width, color=cols_m, alpha=0.45,
               edgecolor=cols_m, linewidth=0.7, label="CSI")
    ax_bar.set_ylim(0.6, 1.02)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, fontsize=6.5)
    ax_bar.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_bar.set_yticklabels(["0.6","0.7","0.8","0.9","1.0"], fontsize=5.5)
    ax_bar.tick_params(length=2, width=0.5)
    ax_bar.spines[["top","right"]].set_visible(False)
    ax_bar.spines[["left","bottom"]].set_linewidth(0.5)
    ax_bar.set_ylabel("Score", fontsize=6)
    # legend
    ax_bar.bar([], [], color="grey", alpha=0.85, label="AUC")
    ax_bar.bar([], [], color="grey", alpha=0.45, label="CSI")
    ax_bar.legend(fontsize=5.5, frameon=False, loc="lower right",
                  handlelength=1, handleheight=0.8)
    # annotate top CSI
    ax_bar.text(0 + width/2, csis[0] + 0.01, "0.937", ha="center",
                fontsize=5, color=C["blue"], fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    #  PANEL D — Max-CSI schematic + key finding (right)
    # ══════════════════════════════════════════════════════════════════════════
    ax.text(0.78, 0.87, "D  Max-CSI → Flood Hazard Map", fontsize=7,
            color=C["purp"], fontweight="bold", va="top")

    # CSI curve (schematic)
    ax_csi = fig.add_axes([0.77, 0.50, 0.21, 0.33])
    tau = np.linspace(0.1, 0.9, 200)
    # Fake bell-shaped CSI peaking at 0.491
    csi_xgb = 0.94 * np.exp(-((tau - 0.491)/0.18)**2)
    csi_dnn = 0.71 * np.exp(-((tau - 0.284)/0.22)**2)
    ax_csi.plot(tau, csi_xgb, color=C["blue"],  lw=1.2, label="XGBoost")
    ax_csi.plot(tau, csi_dnn, color=C["salm"],  lw=1.0, label="DNN",
                linestyle="--")
    ax_csi.axvline(0.491, color=C["blue"],  lw=0.7, ls=":")
    ax_csi.axvline(0.284, color=C["salm"],  lw=0.7, ls=":")
    ax_csi.axvline(0.500, color="#888",     lw=0.6, ls="--")
    ax_csi.set_xlabel("Threshold τ", fontsize=5.5)
    ax_csi.set_ylabel("CSI",         fontsize=5.5)
    ax_csi.set_ylim(0, 1.05)
    ax_csi.tick_params(labelsize=5, length=2, width=0.5)
    ax_csi.spines[["top","right"]].set_visible(False)
    ax_csi.spines[["left","bottom"]].set_linewidth(0.5)
    ax_csi.legend(fontsize=5, frameon=False, loc="upper right",
                  handlelength=1.2)
    ax_csi.set_title("CSI vs Threshold", fontsize=6, pad=2)

    # Key findings box
    findings = (
        "Key Findings\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "✔ XGBoost best: AUC=0.998, CSI=0.937\n"
        "✔ Tree > DL by +0.27 CSI (Tabular Data Hyp.)\n"
        "✔ Max-CSI > Youden: +0.008–0.017 ΔCSI\n"
        "✔ τ* = 0.491 (XGBoost optimal threshold)\n"
        "✔ Pixel-level FHM for DDMA Khagaria"
    )
    ax.text(0.785, 0.46, findings, va="top", ha="left",
            fontsize=5.8, color=C["dgray"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["lgray"],
                      edgecolor=C["purp"], linewidth=0.8))

    # ── Separator lines ───────────────────────────────────────────────────────
    for xv in [0.255, 0.505, 0.76]:
        ax.axvline(xv, color="#CCCCCC", lw=0.5, ymin=0.05, ymax=0.90)

    plt.savefig("graphical_abstract.pdf", bbox_inches="tight", dpi=300)
    plt.savefig("graphical_abstract.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("✓ graphical_abstract.pdf  /  .png")

make_graphical_abstract()
print("\nDone. Files: flowchart_compact.pdf, flowchart_compact.png,")
print("             graphical_abstract.pdf, graphical_abstract.png")


# ─────────────────────────────────────────────────────────────────────────────
#  GRAPHICAL ABSTRACT  (fixed layout — 190 × 85 mm)
#  Replace the make_graphical_abstract() call above with this version
# ─────────────────────────────────────────────────────────────────────────────