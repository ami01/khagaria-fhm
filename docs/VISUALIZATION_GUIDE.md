# Guide for interactive visualisation 

Everything below uses the **v2** outputs (see the README for why v2 replaces the submitted version). Nothing heavy has to be re-run: every table is a CSV and every map a GeoTIFF.

**Conventions**
- Models: `XGBoost`, `CatBoost`, `RandomForest`, `ANN`, `DNN` (these names are used in all file names).
- Features (in model order): Aspect, Curvature, Dis2Roads, Dis2stream, Elevation, Geomorphology, Lithology, LULC, Rainfall, Slope, Soil, SPI, Stream_Density, TWI. The canonical list is `results_v2/run_config.json` → `features`.
- Categorical features (integer class codes): Geomorphology, Lithology, LULC, Soil. The code → class-name tables are not in the repo yet; ask the repository owner.
- Rasters: EPSG:32645 (UTM 45N), 30 m, 1769 × 1935 cells, all on the same grid. NoData outside the district: −9999 for float rasters, 255 for uint8. Reproject to EPSG:4326 / 3857 for web maps, and consider cloud-optimised GeoTIFFs or pre-rendered tiles.

| Visual | Data | Notes |
|---|---|---|
| **Training inputs**: distributions, flood vs non-flood, by split | `results_v2/training/sample_20k_pixels.csv` | 10,000 flood + 10,000 non-flood district pixels: 14 features, `label`, `split` (train/val/test), `x_utm`/`y_utm`, and `p_<model>` predicted probabilities |
| **Correlation / VIF** | `results_v2/metrics/correlation_matrix.csv`, `VIF.csv` | all VIF < 2 |
| **Input layer explorer**: toggle the 14 factors | `data/predictors_30m_v2/*_30m.tif`, boundary `data/vectors/Khagaria_Boundary_UTM.shp`, roads, streams, lithology in `data/vectors/` | `data/predictors_30m/Slope_30m.tif` (the old file) is actually TWI. Use the v2 one |
| **Flood inventory vs prediction** | `data/inventory/label_30m_v2.tif` vs `results_v2/maps/binary/Binary_<model>_MaxCSI.tif` | a confusion map (TP/FP/FN/TN) per model is a nice view |
| **Model parameters** | `results_v2/run_config.json` | hyperparameters, layer sizes, split sizes, thresholds, natural breaks, VIF rule |
| **Fine tuning / training history** | `results_v2/training/history_XGBoost.csv` (train & val AUC per round), `history_CatBoost.csv` (val AUC, logloss), `history_ANN.csv` / `history_DNN.csv` (train & val loss, val AUC per epoch) | XGBoost hit the 1,000-round cap; early stopping did not trigger |
| **Model comparison** | `results_v2/metrics/test_metrics_all_thresholds.csv` (Max-CSI, Youden, 0.5; with TP/FP/FN/TN), `auc_train_val_test.csv` | test set = 326,447 pixels |
| **ROC curves** | `results_v2/curves/roc_<model>_{train,val,test}.csv` (fpr, tpr, threshold) | ≤ 2,000 points per curve |
| **Threshold explorer**: slider of CSI / sensitivity / specificity / confusion counts vs threshold | `results_v2/curves/threshold_curve_<model>_test.csv` (step 0.005) | mark the Max-CSI and Youden thresholds from `run_config.json` → `thresholds`. This is the paper's methodological point |
| **Calibration / reliability diagram** | `results_v2/curves/calibration_<model>_test.csv` | 10 bins: mean predicted vs observed flood frequency |
| **Feature importance (global)** | `results_v2/shap/mean_abs_shap.csv` (Model, Feature, mean_abs_SHAP, rank) | normalise per model before comparing (RF is in probability units, the others in log-odds) |
| **SHAP beeswarm / dependence plots** | `results_v2/shap/shap_values_<model>.npy` (5000 × 14, same feature order) + `shap_sample_features.csv` (feature values, label, pixel_index) | rows align one to one |
| **Susceptibility and hazard maps**: swipe between models | `results_v2/maps/probability/Map_<model>.tif` (0–1), `classified/Classified_<model>_NaturalBreaks.tif` (0–4 = Very Low … Very High), `binary/Binary_<model>_MaxCSI.tif` | class breaks: `results_v2/metrics/natural_breaks_xgboost.csv` |
| **Model agreement** | `results_v2/maps/binary/Agreement_count_MaxCSI.tif` (0–5 models flagging flood), `metrics/agreement_areas.csv` | 85 % of the district is unanimous |
| **Area statistics** | `results_v2/metrics/area_statistics.csv` | km² and % per class and model; district = 1,469 km² |

Suggested stack: Python (Plotly / Dash or Streamlit, `rasterio`, `geopandas`) or a static site with Leaflet / MapLibre and pre-tiled rasters. Please work on a branch (e.g. `viz`) and open a pull request, so that `main` stays in sync with the manuscript.
