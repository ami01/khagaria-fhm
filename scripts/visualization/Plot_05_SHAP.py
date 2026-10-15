"""
Plot_05_SHAP.py — SHAP Feature Importance for ALL 5 Models
===========================================================
Models covered: XGBoost, RandomForest, CatBoost, ANN, DNN
Output:  Fig_SHAP_All_Models.png        — 5-column bar chart (mean |SHAP|)
         Fig_SHAP_Beeswarm_<Model>.png  — beeswarm per model
         Fig_SHAP_Rank_Heatmap.png      — cross-model rank heatmap
         SHAP_Feature_Importance_All.csv

Fixes (v4)
----------
1. ANN/DNN  — DROP_FEATURE auto-detected by substring match ("slope" case-insensitive)
              so it works whether the raster is named "Slope", "Slope_30m", etc.
2. RF OOM   — cuML RF: convert_to_sklearn() tried first (zero VRAM); if unavailable,
              run chunked CPU inference with CUDA cache cleared between chunks.
3. All      — _extract_shap() handles ndarray / list-of-2 / list-of-1 uniformly.
"""

import os, gc, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import rasterio
from rasterio.warp import reproject, Resampling
import joblib

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
BASE        = "Khagaria_FSM"
RASTER_DIR  = os.path.join(BASE, "Raster/Aligned")
INVENTORY   = os.path.join(BASE, "Inventory/F_nF_Khagaria_binary.tif")
MODELS_DIR  = os.path.join(BASE, "models")
OUT_DIR     = os.path.join(BASE, "Manuscript_Figures")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLE_SHAP = 3000
N_BG_KERNEL   = 100
RANDOM_SEED   = 42

MODEL_COLORS = {
    "XGBoost":      "#E63946",
    "RandomForest": "#2A9D8F",
    "CatBoost":     "#E9C46A",
    "ANN":          "#457B9D",
    "DNN":          "#8338EC",
}

# ─── UTILITY: normalise shap_values() output ─────────────────────────────────
def _extract_shap(raw):
    """
    KernelExplainer.shap_values() returns:
      - ndarray (n, f)          ← regression / single output
      - list [(n,f), (n,f)]     ← binary classifier → take [1] (positive class)
      - list [(n,f)]            ← rare single-element list
    Always returns float32 ndarray (n, f).
    """
    if isinstance(raw, np.ndarray):
        return raw.astype(np.float32)
    if isinstance(raw, list):
        arr = raw[1] if len(raw) == 2 else raw[0]
        return np.array(arr, dtype=np.float32)
    return np.array(raw, dtype=np.float32)


# ─── UTILITY: find the column to drop by substring match ─────────────────────
def _find_drop_col(feature_names_list, substring):
    """
    Returns (column_mask, kept_names, matched_name) where column_mask is a
    boolean array with False at the matched position.
    Matching is case-insensitive substring search.
    Raises ValueError if no match found.
    """
    matches = [f for f in feature_names_list
               if substring.lower() in f.lower()]
    if not matches:
        raise ValueError(
            f"No feature matching '{substring}' found in {feature_names_list}")
    drop_name  = matches[0]   # take first match
    mask       = np.array([f != drop_name for f in feature_names_list])
    kept       = [f for f in feature_names_list if f != drop_name]
    return mask, kept, drop_name


# ─── UTILITY: infer nn.Linear layers from state_dict ────────────────────────
def _infer_linear_layers(state_dict):
    """Returns sorted list of (prefix, in_features, out_features) for each Linear."""
    linear_keys = sorted(
        [k for k in state_dict
         if k.endswith(".weight") and state_dict[k].ndim == 2],
        key=lambda k: [int(x) if x.isdigit() else x
                       for x in k.replace(".", " ").split()]
    )
    return [(k[:-len(".weight")], state_dict[k].shape[1], state_dict[k].shape[0])
            for k in linear_keys]


# ─── STEP 1: LOAD FEATURE RASTERS ────────────────────────────────────────────
print("=" * 60)
print("Loading feature rasters...")
print("=" * 60)

with rasterio.open(INVENTORY) as src:
    y_raw    = src.read(1).astype(np.float32)
    ref_meta = src.meta.copy()
    ref_h, ref_w = src.shape

y_flat     = y_raw.ravel()
valid_mask = (y_flat != -9999) & (~np.isnan(y_flat)) & (y_flat >= 0)
y_true     = (y_flat[valid_mask] > 0).astype(np.int8)
print(f"  Valid pixels: {valid_mask.sum():,}  |  Flood: {y_true.sum():,}  "
      f"Non-flood: {(y_true==0).sum():,}")

feature_files = sorted(glob.glob(os.path.join(RASTER_DIR, "*.tif")))
feature_files = [f for f in feature_files
                 if os.path.abspath(f) != os.path.abspath(INVENTORY)]
feature_names = [os.path.basename(f).replace(".tif", "") for f in feature_files]
print(f"  Features ({len(feature_names)}): {feature_names}")


def read_feature(fpath, ref_meta, ref_h, ref_w):
    with rasterio.open(fpath) as src:
        if src.height == ref_h and src.width == ref_w:
            band = src.read(1).astype(np.float32).ravel()
        else:
            dest = np.zeros((ref_h, ref_w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=dest,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_meta["transform"], dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata, dst_nodata=np.nan,
            )
            band = dest.ravel()
    return np.where(np.isfinite(band), band, 0.0).astype(np.float32)


print("  Reading all feature rasters...")
X_valid = np.empty((int(valid_mask.sum()), len(feature_files)), dtype=np.float32)
for i, fpath in enumerate(feature_files):
    band = read_feature(fpath, ref_meta, ref_h, ref_w)
    X_valid[:, i] = band[valid_mask]
    print(f"    [{i+1:02d}/{len(feature_files)}] {feature_names[i]:30s}"
          f"  range [{X_valid[:,i].min():.4f}, {X_valid[:,i].max():.4f}]")

X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

np.random.seed(RANDOM_SEED)
n_valid  = X_valid.shape[0]
idx_shap = np.random.choice(n_valid, min(N_SAMPLE_SHAP, n_valid), replace=False)
X_sample = X_valid[idx_shap].copy()
y_sample = y_true[idx_shap]

idx_bg = np.random.choice(n_valid, min(N_BG_KERNEL, n_valid), replace=False)
X_bg   = X_valid[idx_bg].copy()

n_feats = X_sample.shape[1]
print(f"\n  X_sample: {X_sample.shape}  dtype={X_sample.dtype}"
      f"  range=[{X_sample.min():.4f}, {X_sample.max():.4f}]")


# ─── STEP 2: LOAD MODELS & COMPUTE SHAP ──────────────────────────────────────
print("\n" + "=" * 60)
print("Loading models and computing SHAP values...")
print("=" * 60)

import shap
shap.initjs()

models_shap          = {}
models_feature_names = {m: feature_names for m in
                        ["XGBoost", "RandomForest", "CatBoost", "ANN", "DNN"]}

# ── 2a. XGBoost ───────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    model_xgb   = xgb.Booster()
    model_xgb.load_model(os.path.join(MODELS_DIR, "model_xgb.json"))
    safe_fnames = [f.replace(" ", "_") for f in feature_names]

    class XGBWrapper:
        def __init__(self, booster, fnames):
            self.booster = booster; self.fnames = fnames
        def predict(self, x):
            dm = xgb.DMatrix(np.array(x, dtype=np.float64), feature_names=self.fnames)
            return self.booster.predict(dm).astype(np.float64)

    wrap_xgb = XGBWrapper(model_xgb, safe_fnames)
    exp_xgb  = shap.Explainer(wrap_xgb.predict, X_bg.astype(np.float64),
                               feature_names=feature_names)
    sv_xgb   = _extract_shap(exp_xgb(X_sample.astype(np.float64)).values)
    models_shap["XGBoost"] = sv_xgb
    print(f"  XGBoost  SHAP ✓  shape: {sv_xgb.shape}")
except Exception as e:
    print(f"  XGBoost  SHAP FAILED: {e}")


# ── 2b. RandomForest ──────────────────────────────────────────────────────────
# Strategy:
#   1. TreeExplainer (sklearn RF) — fastest, exact
#   2. cuML → convert_to_sklearn() then TreeExplainer — zero VRAM
#   3. Chunked CPU KernelExplainer — clears CUDA cache between chunks
try:
    rf_path  = os.path.join(MODELS_DIR, "model_rf.pkl")
    model_rf = joblib.load(rf_path)

    # ── attempt 1: sklearn TreeExplainer ──────────────────────────────────────
    try:
        exp_rf  = shap.TreeExplainer(model_rf, feature_perturbation="interventional",
                                     data=X_bg.astype(np.float32))
        sv_rf   = _extract_shap(exp_rf.shap_values(X_sample.astype(np.float32)))
        models_shap["RandomForest"] = sv_rf
        print(f"  RandomForest SHAP (TreeExplainer/sklearn) ✓  shape: {sv_rf.shape}")

    except Exception as e1:
        print(f"  RF TreeExplainer failed ({e1})")

        # ── attempt 2: cuML → sklearn conversion ──────────────────────────────
        try:
            sk_rf   = model_rf.convert_to_sklearn()
            exp_rf2 = shap.TreeExplainer(sk_rf, feature_perturbation="interventional",
                                         data=X_bg.astype(np.float32))
            sv_rf   = _extract_shap(exp_rf2.shap_values(X_sample.astype(np.float32)))
            models_shap["RandomForest"] = sv_rf
            print(f"  RandomForest SHAP (cuML→sklearn TreeExplainer) ✓  shape: {sv_rf.shape}")

        except Exception as e2:
            print(f"  RF cuML conversion failed ({e2}); falling back to chunked CPU KernelExplainer")

            # ── attempt 3: chunked CPU KernelExplainer ─────────────────────────
            # Force CPU prediction; clear CUDA cache between each chunk.
            try:
                import torch

                def rf_predict_cpu(x):
                    """Predict on CPU; handles cuML + sklearn RF."""
                    x_np = np.array(x, dtype=np.float32)
                    for kwargs in [{"predict_model": "CPU"}, {}]:
                        try:
                            p = model_rf.predict_proba(x_np, **kwargs)
                            p = p.to_numpy() if hasattr(p, "to_numpy") else np.array(p)
                            return p[:, 1].astype(np.float64)
                        except Exception:
                            continue
                    out = model_rf.predict(x_np)
                    out = out.to_numpy() if hasattr(out, "to_numpy") else np.array(out)
                    return out.astype(np.float64)

                # Use only 30 background points and predict 50 rows per chunk
                CHUNK     = 50
                X_rf_bg   = X_bg[:30].astype(np.float32)
                X_rf_samp = X_sample[:300].astype(np.float32)
                exp_rf_k  = shap.KernelExplainer(rf_predict_cpu, X_rf_bg)

                sv_chunks = []
                for start in range(0, len(X_rf_samp), CHUNK):
                    chunk = X_rf_samp[start:start + CHUNK]
                    sv_c  = _extract_shap(
                        exp_rf_k.shap_values(chunk, silent=True))
                    sv_chunks.append(sv_c)
                    # Free CUDA memory between chunks
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"    RF chunk {start}–{start+len(chunk)} done")

                sv_rf = np.vstack(sv_chunks)
                models_shap["RandomForest"] = sv_rf
                print(f"  RandomForest SHAP (chunked KernelExplainer) ✓  shape: {sv_rf.shape}")

            except Exception as e3:
                print(f"  RandomForest SHAP FAILED (all strategies): {e3}")

except Exception as e:
    print(f"  RandomForest SHAP FAILED (load): {e}")


# ── 2c. CatBoost ──────────────────────────────────────────────────────────────
try:
    from catboost import CatBoostClassifier, Pool
    model_cat = CatBoostClassifier()
    model_cat.load_model(os.path.join(MODELS_DIR, "model_cat.cbm"))
    pool_cat  = Pool(X_sample, feature_names=feature_names)
    exp_cat   = shap.TreeExplainer(model_cat)
    sv_cat    = _extract_shap(exp_cat.shap_values(pool_cat))
    if sv_cat.ndim == 3:
        sv_cat = sv_cat[:, :, 1]
    models_shap["CatBoost"] = sv_cat
    print(f"  CatBoost SHAP ✓  shape: {sv_cat.shape}")
except Exception as e:
    print(f"  CatBoost SHAP FAILED: {e}")


# ── helper: build Sequential matching the "model.*" key namespace ─────────────
import torch
import torch.nn as nn

def _load_net(sd, n_in, dims):
    """
    Rebuild a net as  nn.Module { self.model = Sequential(...) }
    matching the "model.0.weight", "model.4.weight", ... key scheme.
    Layers: Linear→ReLU→BN→ReLU repeating, final Linear(→1).
    Tries strict=True first, then strict=False.
    """
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, in_d = [], n_in
            for out_d in dims[:-1]:
                layers += [nn.Linear(in_d, out_d), nn.ReLU(),
                           nn.BatchNorm1d(out_d), nn.ReLU()]
                in_d = out_d
            layers.append(nn.Linear(in_d, dims[-1]))
            self.model = nn.Sequential(*layers)
        def forward(self, x):
            return torch.sigmoid(self.model(x))

    net = _Net()
    try:
        net.load_state_dict(sd, strict=True)
    except Exception as err:
        print(f"    strict load failed ({err}); retrying with strict=False")
        net.load_state_dict(sd, strict=False)
    net.eval()
    return net


# ── 2d. ANN (PyTorch) ─────────────────────────────────────────────────────────
# FIX: auto-detect slope column by substring match instead of hardcoded name.
try:
    sd_ann     = torch.load(os.path.join(MODELS_DIR, "model_ann.pth"),
                            map_location="cpu")
    lin_layers = _infer_linear_layers(sd_ann)
    ann_n_in   = lin_layers[0][1]
    ann_dims   = [l[2] for l in lin_layers]
    print(f"  ANN  Linear layers (in→out): {[(l[1], l[2]) for l in lin_layers]}")

    if ann_n_in == n_feats:
        # Model was trained on all current features — no column to drop
        ann_col_mask  = np.ones(n_feats, dtype=bool)
        ann_fnames    = feature_names
        print(f"  ANN: uses all {ann_n_in} features (no column dropped)")
    else:
        # Need to drop exactly (n_feats - ann_n_in) columns.
        # Try dropping "slope" first; if that still leaves the wrong count,
        # raise a clear error listing what was found.
        n_drop = n_feats - ann_n_in
        if n_drop == 1:
            ann_col_mask, ann_fnames, dropped = _find_drop_col(feature_names, "slope")
            assert ann_col_mask.sum() == ann_n_in, (
                f"ANN: after dropping '{dropped}' got {ann_col_mask.sum()} cols, "
                f"but model expects {ann_n_in}.\n"
                f"Available features: {feature_names}")
            print(f"  ANN: dropped '{dropped}'  →  {ann_n_in} features")
        else:
            raise ValueError(
                f"ANN expects {ann_n_in} features but rasters provide {n_feats}. "
                f"Need to drop {n_drop} columns — edit DROP logic manually.")

    model_ann = _load_net(sd_ann, ann_n_in, ann_dims)

    def ann_pred(x):
        with torch.no_grad():
            return model_ann(
                torch.tensor(np.array(x, dtype=np.float32))).numpy().ravel()

    X_bg_ann   = X_bg[:50,  ann_col_mask].astype(np.float32)
    X_samp_ann = X_sample[:300, ann_col_mask].astype(np.float32)

    exp_ann = shap.KernelExplainer(ann_pred, X_bg_ann)
    sv_ann  = _extract_shap(exp_ann.shap_values(X_samp_ann, silent=False))

    models_shap["ANN"]          = sv_ann
    models_feature_names["ANN"] = ann_fnames
    print(f"  ANN  SHAP (KernelExplainer, {X_samp_ann.shape[0]} pts) ✓  shape: {sv_ann.shape}")

except Exception as e:
    import traceback
    print(f"  ANN  SHAP FAILED: {e}")
    traceback.print_exc()


# ── 2e. DNN (PyTorch) ─────────────────────────────────────────────────────────
try:
    sd_dnn     = torch.load(os.path.join(MODELS_DIR, "model_dnn.pth"),
                            map_location="cpu")
    lin_layers = _infer_linear_layers(sd_dnn)
    dnn_n_in   = lin_layers[0][1]
    dnn_dims   = [l[2] for l in lin_layers]
    print(f"  DNN  Linear layers (in→out): {[(l[1], l[2]) for l in lin_layers]}")

    if dnn_n_in == n_feats:
        dnn_col_mask = np.ones(n_feats, dtype=bool)
        dnn_fnames   = feature_names
        print(f"  DNN: uses all {dnn_n_in} features (no column dropped)")
    else:
        n_drop = n_feats - dnn_n_in
        if n_drop == 1:
            dnn_col_mask, dnn_fnames, dropped = _find_drop_col(feature_names, "slope")
            assert dnn_col_mask.sum() == dnn_n_in, (
                f"DNN: after dropping '{dropped}' got {dnn_col_mask.sum()} cols, "
                f"but model expects {dnn_n_in}.\n"
                f"Available features: {feature_names}")
            print(f"  DNN: dropped '{dropped}'  →  {dnn_n_in} features")
        else:
            raise ValueError(
                f"DNN expects {dnn_n_in} features but rasters provide {n_feats}. "
                f"Need to drop {n_drop} columns — edit DROP logic manually.")

    model_dnn = _load_net(sd_dnn, dnn_n_in, dnn_dims)

    def dnn_pred(x):
        with torch.no_grad():
            return model_dnn(
                torch.tensor(np.array(x, dtype=np.float32))).numpy().ravel()

    X_bg_dnn   = X_bg[:50,  dnn_col_mask].astype(np.float32)
    X_samp_dnn = X_sample[:300, dnn_col_mask].astype(np.float32)

    exp_dnn = shap.KernelExplainer(dnn_pred, X_bg_dnn)
    sv_dnn  = _extract_shap(exp_dnn.shap_values(X_samp_dnn, silent=False))

    models_shap["DNN"]          = sv_dnn
    models_feature_names["DNN"] = dnn_fnames
    print(f"  DNN  SHAP (KernelExplainer, {X_samp_dnn.shape[0]} pts) ✓  shape: {sv_dnn.shape}")

except Exception as e:
    import traceback
    print(f"  DNN  SHAP FAILED: {e}")
    traceback.print_exc()


if not models_shap:
    raise RuntimeError("No SHAP values computed for any model. Check model files.")
print(f"\n  Models with SHAP: {list(models_shap.keys())}")


# ─── STEP 3: MEAN |SHAP| DATAFRAMES ──────────────────────────────────────────
shap_dfs = {}
for mname, sv in models_shap.items():
    fnames   = models_feature_names.get(mname, feature_names)
    mean_abs = np.abs(sv).mean(axis=0)
    if len(mean_abs) != len(fnames):
        print(f"  WARNING: {mname} SHAP cols={len(mean_abs)} vs fnames={len(fnames)}")
        mean_abs = np.pad(mean_abs,
                          (0, max(0, len(fnames) - len(mean_abs))))[:len(fnames)]
    df = (pd.DataFrame({"Feature": fnames, "SHAP": mean_abs})
          .sort_values("SHAP", ascending=False)
          .reset_index(drop=True))
    shap_dfs[mname] = df

pd.concat([df.assign(Model=m) for m, df in shap_dfs.items()]).to_csv(
    os.path.join(OUT_DIR, "SHAP_Feature_Importance_All.csv"), index=False)
print("\n  Saved SHAP_Feature_Importance_All.csv")


# ─── STEP 4: COMBINED BAR CHART ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Plotting figures...")
print("=" * 60)

model_order = [m for m in ["XGBoost", "RandomForest", "CatBoost", "ANN", "DNN"]
               if m in shap_dfs]
n_models = len(model_order)

fig, axes = plt.subplots(
    1, n_models,
    figsize=(5.2 * n_models, max(7, len(feature_names) * 0.55)),
    facecolor="white",
)
if n_models == 1:
    axes = [axes]

for ax, mname in zip(axes, model_order):
    df    = shap_dfs[mname].sort_values("SHAP", ascending=True)
    color = MODEL_COLORS.get(mname, "#888888")
    n_f   = len(df)
    alphas = np.linspace(0.35, 1.0, n_f)
    rgba   = [(*mc.to_rgb(color), a) for a in alphas]

    bars = ax.barh(df["Feature"], df["SHAP"],
                   color=rgba, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, df["SHAP"]):
        if val > 0.001:
            ax.text(bar.get_width() + df["SHAP"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", ha="left", fontsize=7.5)

    ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
    ax.set_title(mname, fontsize=12, fontweight="bold", color=color, pad=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.25, linestyle="--", lw=0.6)
    ax.set_xlim(0, df["SHAP"].max() * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    rank_labels = [f"#{n_f-i}  {f}" for i, f in
                   enumerate(df.sort_values("SHAP", ascending=True)["Feature"])]
    ax.set_yticklabels(rank_labels)

fig.suptitle(
    "SHAP Feature Importance — All Models\n"
    "Khagaria Flood Susceptibility (Mean |SHAP Value| over sampled pixels)",
    fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout(w_pad=2.5)
out_bar = os.path.join(OUT_DIR, "Fig_SHAP_All_Models.png")
plt.savefig(out_bar, dpi=300, bbox_inches="tight", facecolor="white")
print(f"✅ Saved: {out_bar}")
plt.close()


# ─── STEP 5: BEESWARM PLOTS ──────────────────────────────────────────────────
for mname, sv in models_shap.items():
    try:
        fnames    = models_feature_names.get(mname, feature_names)
        df_sorted = shap_dfs[mname]
        top_n     = min(12, len(fnames))
        top_feats = df_sorted["Feature"].tolist()[:top_n]
        top_idx   = [fnames.index(f) for f in top_feats if f in fnames][:top_n]
        if not top_idx:
            continue

        n_pts = sv.shape[0]
        # Match X columns to the model's feature space
        if len(fnames) < n_feats:
            # same drop logic used above
            _, _, drop_name = _find_drop_col(feature_names, "slope")
            col_mask = np.array([f != drop_name for f in feature_names])
            X_plot   = X_sample[:n_pts, col_mask]
        else:
            X_plot = X_sample[:n_pts, :]

        fig_b, ax_b = plt.subplots(figsize=(11, 7), facecolor="white")
        for fi, (feat_name, col_idx) in enumerate(zip(top_feats, top_idx)):
            if col_idx >= sv.shape[1]:
                continue
            si  = sv[:, col_idx]
            xi  = X_plot[:, col_idx]
            jit = np.random.uniform(-0.35, 0.35, len(si))
            sc  = ax_b.scatter(
                si, fi + jit, c=xi, cmap="coolwarm_r", s=9, alpha=0.55,
                vmin=np.nanpercentile(xi, 5), vmax=np.nanpercentile(xi, 95),
                linewidths=0, zorder=3,
            )

        ax_b.set_yticks(range(top_n))
        ax_b.set_yticklabels(top_feats, fontsize=10)
        ax_b.axvline(0, color="black", lw=1.0, linestyle="--", alpha=0.6)
        ax_b.set_xlabel("SHAP Value  (positive → ↑ flood probability)", fontsize=11)
        ax_b.set_title(
            f"{mname} — SHAP Beeswarm (Top {top_n} Features)\n"
            "Point colour = Feature value  (blue=low, red=high)",
            fontsize=11, fontweight="bold")
        ax_b.grid(axis="x", alpha=0.2, linestyle="--", lw=0.6)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

        cbar = plt.colorbar(sc, ax=ax_b, shrink=0.6, pad=0.02, aspect=25)
        cbar.set_label("Feature value (relative)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        fig_b.tight_layout()
        out_bee = os.path.join(OUT_DIR, f"Fig_SHAP_Beeswarm_{mname}.png")
        plt.savefig(out_bee, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✅ Saved: {out_bee}")
        plt.close()

    except Exception as e:
        print(f"  Beeswarm for {mname} failed: {e}")


# ─── STEP 6: CROSS-MODEL RANK HEATMAP ────────────────────────────────────────
print("\nGenerating cross-model SHAP rank comparison...")
try:
    rank_data = {}
    for mname in model_order:
        df = shap_dfs[mname].copy()
        df["Rank"] = range(1, len(df) + 1)
        rank_data[mname] = df.set_index("Feature")["Rank"]

    rank_df = pd.DataFrame(rank_data)
    rank_df = rank_df.loc[shap_dfs[model_order[0]]["Feature"]]

    n_fp = rank_df.shape[0]
    fig_h, ax_h = plt.subplots(
        figsize=(2.5 + 1.8 * n_models, max(6, n_fp * 0.52)),
        facecolor="white")

    im = ax_h.imshow(rank_df.values, cmap="RdYlGn_r", aspect="auto",
                     vmin=1, vmax=n_fp)
    ax_h.set_xticks(range(n_models))
    ax_h.set_xticklabels(model_order, fontsize=11, fontweight="bold", rotation=15)
    ax_h.set_yticks(range(n_fp))
    ax_h.set_yticklabels(rank_df.index, fontsize=9)

    for i in range(n_fp):
        for j in range(rank_df.shape[1]):
            val = rank_df.iloc[i, j]
            if not np.isnan(val):
                ax_h.text(j, i, f"#{int(val)}", ha="center", va="center",
                          fontsize=8, fontweight="bold",
                          color="white" if val <= n_fp // 3 else "black")

    cbar_h = plt.colorbar(im, ax=ax_h, shrink=0.5, pad=0.02)
    cbar_h.set_label("Importance Rank (1 = most important)", fontsize=9)
    cbar_h.ax.invert_yaxis()
    ax_h.set_title(
        "Cross-Model Feature Importance Rank Comparison\n"
        "(SHAP-based ranking — green = most important)",
        fontsize=12, fontweight="bold")
    ax_h.grid(False)
    fig_h.tight_layout()

    out_hm = os.path.join(OUT_DIR, "Fig_SHAP_Rank_Heatmap.png")
    plt.savefig(out_hm, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {out_hm}")
    plt.close()

except Exception as e:
    print(f"  Rank heatmap skipped: {e}")

print("\n✅ All SHAP figures complete.")
print(f"   Outputs in: {OUT_DIR}")