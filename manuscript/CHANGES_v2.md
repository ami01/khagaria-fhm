# manuscript_v2.tex: changes from the submitted manuscript.tex (JESS-D-26-00905)

All numbers come from `results_v2/` and were produced by the v2 scripts (not in this repository for now). `make_v2.py` builds `manuscript_v2.tex` from `manuscript.tex`, so the edits are reviewable in one place.

## Changed findings
| Claim in the submitted text | v2 |
|---|---|
| Slope and TWI have infinite VIF in flat terrain | The Slope layer was a copy of TWI. With real slope, the maximum VIF is 1.82 and Slope–TWI r = −0.28 |
| ANN/DNN use 12 features (Slope, TWI dropped) | All five models use all 14 |
| XGBoost AUC 0.998, CSI 0.937, τ* 0.491 | AUC 0.992, CSI 0.923, τ* 0.493 |
| DNN / ANN CSI 0.705 / 0.668; thresholds 0.284 / 0.372; "probability miscalibration" | CSI 0.804 / 0.795; thresholds 0.446 / 0.429; the neural networks are the **best**-calibrated models (ECE 0.013–0.014) |
| Tree–DL CSI gap 0.232 | 0.119 |
| Max-CSI beats Youden by +0.008 to +0.017 (mean +0.013); Max-CSI thresholds higher than Youden's | +0.002 to +0.008 (mean +0.0048); Max-CSI thresholds **lower** than Youden's; fixed 0.5 is within 0.0035 CSI |
| DL hazard maps 20 % larger than inventory (≈1,010 km²) | 4–6 % larger (856 / 876 km²) |
| Inventory 839.40 km²; district 1,486 km² | 824.4 km² inside the modelled district of 1,469.0 km² (the old figure included pixels outside the district) |
| XGBoost Very High = 22.9 % of the district | 47.0 % (the old percentages were of the whole raster rectangle) |
| SHAP top four: Geomorphology, Soil, Rainfall, Elevation | Same four in every model; order differs (XGBoost: Rainfall first) |
| Non-flood pixels sampled ~1:1 from areas unaffected | Every district pixel is labelled; 56.1 % flood, no resampling |
| 80:20 split; "independent validation" | 70/10/20; early stopping and thresholds on validation, metrics on test only (v1 stopped early on the test set and picked thresholds on the full raster) |
| GPU / cuML / proxy-RF SHAP | CPU; scikit-learn RF; SHAP on the trained forest itself |

## New in v2
- An input-audit paragraph (Section 3.3).
- Calibration (ECE) in Table 3 and in the discussion.
- Model agreement (85.4 % of the district unanimous).
- The soil-map artefact in the maps, and a caveat that rainfall partly acts as a location proxy.
- A training-curves figure in the supplement.
- Limitations: the union inventory labels other events' floods as non-flood; categorical codes vs one-hot for the neural networks.

## Removed
- GAIN-importance comparisons (no GAIN analysis in this repo).
- Claims about specific geomorphology/soil classes being "red/blue" in the beeswarms: the class-code order is arbitrary; the code → name tables are needed to restore them.

## Still for the authors (red in the PDF)
- **Use of AI Tools:** the earlier draft said AI was not used for data analysis. That is no longer true for v2.
- **Data Availability:** the repo is private. Make it public with a Zenodo DOI, or keep "on request".
- **Author contributions:** check whether the re-analysis changes them.
- **Resubmission:** decide how to handle this with JESS (e.g. contact the editor before review, or a corrected version with the revision).
