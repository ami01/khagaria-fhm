import os
import glob
import numpy as np
import rasterio
import pandas as pd

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE           = "Khagaria_FSM"
PREDICTION_DIR = os.path.join(BASE, "Final_Binary_Maps_MaxCSI/")
OUTPUT_CSV     = os.path.join(BASE, "Final_Binary_Maps_MaxCSI/raster_areas_sqkm.csv")

# Get list of all .tif files
raster_files = glob.glob(os.path.join(PREDICTION_DIR, "*.tif"))

results = []

for r_file in raster_files:
    with rasterio.open(r_file) as src:
        # Read the first band
        data = src.read(1)
        
        # Get pixel resolution (usually in meters for projected CRS)
        res = src.res
        pixel_area_m2 = res[0] * res[1]
        
        # Calculate counts for each class
        count_0 = np.sum(data == 0)
        count_1 = np.sum(data == 1)
        
        # Calculate area in sq. km (1 km^2 = 1,000,000 m^2)
        area_0_km2 = (count_0 * pixel_area_m2) / 1_000_000
        area_1_km2 = (count_1 * pixel_area_m2) / 1_000_000
        
        results.append({
            "filename": os.path.basename(r_file),
            "pixel_res_m": res[0],
            "count_0": count_0,
            "count_1": count_1,
            "area_0_sqkm": round(area_0_km2, 2),
            "area_1_sqkm": round(area_1_km2, 2)
        })

# Create DataFrame and save
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(df[['filename', 'area_0_sqkm', 'area_1_sqkm']])
print(f"\nResults saved to: {OUTPUT_CSV}")