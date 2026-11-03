# Khagaria flood susceptibility and hazard mapping

Code, data and results for **"Flood Susceptibility and Hazard Zonation of Khagaria District, North Bihar, India: A Machine Learning Approach Using Multi-Mission SAR Data"**, under review at the *Journal of Earth System Science*.

## What the study does
A SAR-derived flood inventory (union of the 2008, 2020 and 2021 events; ENVISAT ASAR and Sentinel-1) labels every 30 m pixel of the district as flood or non-flood. Five models (RandomForest, XGBoost, CatBoost, ANN, DNN) are trained on **14 conditioning factors** to give flood-susceptibility probability maps. Each map is turned into a binary hazard map with the threshold that maximises the Critical Success Index (**Max-CSI**), which is compared with the Youden-J threshold. SHAP explains the models.

Pixels are split 70 / 10 / 20 (train / validation / test, seed 42). Early stopping and thresholds use the validation split only; every reported metric is on the independent test split. Only the 1,632,234 pixels inside the district (1,469 km²) are modelled.

## Headline results (test set, Max-CSI threshold from the validation set): `results_v2/metrics/test_metrics_maxcsi.csv`
| Model | τ* | AUC | CSI | Sensitivity | Specificity | Hazard area (km²) |
|---|---|---|---|---|---|---|
| XGBoost | 0.493 | 0.992 | 0.923 | 0.959 | 0.950 | 822.0 |
| CatBoost | 0.494 | 0.979 | 0.873 | 0.932 | 0.914 | 824.8 |
| RandomForest | 0.487 | 0.971 | 0.848 | 0.921 | 0.891 | 827.6 |
| DNN | 0.446 | 0.951 | 0.804 | 0.909 | 0.834 | 856.4 |
| ANN | 0.429 | 0.945 | 0.795 | 0.913 | 0.810 | 875.6 |

- **Inventory:** 824.4 km² flooded = 56.1 % of the 1,469 km² modelled district.
- **Max-CSI vs Youden:** Max-CSI wins for all five models, by +0.002 to +0.008 CSI. Max-CSI thresholds are lower than Youden's.
- **Calibration:** the neural networks are the best calibrated (ECE 0.013–0.014, against 0.023–0.054 for the trees).
- **SHAP:** Rainfall, Geomorphology, Soil and Elevation are ranks 1–4 in all five models; TWI, Slope, Aspect and Curvature are ranks 11–14.
- **Model agreement:** all five models give the same class on 85.4 % of the district.
- **Soil artefact:** the straight east–west line visible in the maps is a unit boundary in the ~1 km soil layer.

## Scripts in this repo
Only the plotting scripts are kept here; everything they read is already in `results_v2/`, `data/` and `models_v2/`.
```bash
conda env create -f environment_v2.yml && conda activate khagaria-fhm-v2
python scripts/v2/03_figures.py      # figures → results_v2/figures/
python scripts/v2/04_schematics.py   # flowchart + graphical abstract
```
`scripts/v2/nets.py` defines the ANN/DNN architectures, needed to load `models_v2/model_ann.pth` and `model_dnn.pth`.

The data-preparation, training and SHAP scripts are available from the repository owner on request.

## Layout
| Path | What |
|---|---|
| `data/predictors_30m_v2/` | 14 aligned factor rasters, EPSG:32645, 30 m, 1769 × 1935, NoData −9999 outside the district |
| `data/inventory/label_30m_v2.tif` | label on the same grid: 1 flood, 0 non-flood, 255 outside the district |
| `data/district_mask_30m.tif` | 1 = modelled district pixel |
| `data/vectors/` | district boundary, streams, roads, lithology and HydroSHEDS basins (shapefiles) |
| `models_v2/` | `model_xgb.json`, `model_cat.cbm`, `model_rf.joblib` (scikit-learn), `model_ann.pth` / `model_dnn.pth` (architectures in `scripts/v2/nets.py`), `scaler.pkl`. Feature order: `results_v2/run_config.json` → `features` |
| `results_v2/metrics/` | test metrics at Max-CSI / Youden / 0.5, AUC train/val/test, VIF, correlation matrix, area statistics, natural breaks, model agreement |
| `results_v2/curves/` | ROC (train/val/test), CSI/sensitivity/specificity vs threshold (test), calibration tables: per model, as CSV |
| `results_v2/shap/` | mean \|SHAP\| and ranks (`mean_abs_shap.csv`), per-pixel SHAP values (`shap_values_<model>.npy`, 5000 × 14) and their feature values (`shap_sample_features.csv`) |
| `results_v2/training/` | training histories (XGBoost/CatBoost AUC per round, ANN/DNN loss per epoch), `sample_20k_pixels.csv` (balanced pixels with features, label, split, coordinates and all five predictions), `split_indices.npz` |
| `results_v2/maps/` | `probability/Map_*.tif`, `binary/Binary_*_MaxCSI.tif`, `binary/Agreement_count_MaxCSI.tif` (0–5 models), `classified/Classified_*_NaturalBreaks.tif` |
| `results_v2/figures/` | all figures |

See `docs/VISUALIZATION_GUIDE.md` for the interactive-visualisation plan.
