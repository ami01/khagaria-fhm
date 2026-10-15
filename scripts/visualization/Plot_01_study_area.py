"""
FIGURE 1: Study Area Map — 4-panel layout  [NO pygadm]
=======================================================
pygadm is completely abandoned — it crashes on every call due to a
pandas version mismatch in the installed version.

All boundary data downloaded directly:
  - Natural Earth 110m countries  → India outline + Asia context
  - Natural Earth 10m states      → Indian state boundaries (incl. full J&K claim)
  - GADM v4.1 India Level-2 JSON  → Bihar districts + Khagaria
    URL: https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip

All files cached in Boundary_Cache/ — only downloaded once.
First run downloads ~80 MB total.
"""

import os, warnings, urllib.request, zipfile
import numpy as np
import geopandas as gpd
import rasterio, rasterio.plot as rplot
import matplotlib, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import contextily as ctx
from pyproj import Transformer

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.8})

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
BASE      = "Khagaria_FSM"
DEM       = os.path.join(BASE, "Raster/Aligned/Elevation_30m.tif")
FLOOD_INV = os.path.join(BASE, "Inventory/F_nF_Khagaria_binary.tif")
RIVERS    = os.path.join(BASE, "Shapefiles/Rivers_Khagaria.shp")
OUT_DIR   = os.path.join(BASE, "Manuscript_Figures")
CACHE     = os.path.join(BASE, "Boundary_Cache")
for d in (OUT_DIR, CACHE): os.makedirs(d, exist_ok=True)

TILE_PROVIDER = ctx.providers.OpenStreetMap.Mapnik
DEM_CMAP      = "plasma"
C = dict(india_f="#D6E8F2", india_e="#555555", bihar_f="#F5C97A", bihar_e="#444444",
         khag_f="#E05C3A",  khag_e="#111111",  neigh="#EFEFEF",   river="#2979FF",
         flood="#1a6aff",   nflood="#FDD835")


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD / CACHE
# ═══════════════════════════════════════════════════════════════════════════════
def fetch(url, dest, label):
    if os.path.exists(dest):
        return True
    print(f"   Downloading {label} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest,"wb") as f:
            f.write(r.read())
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        if os.path.exists(dest): os.remove(dest)
        return False

def unzip(zpath, dest_dir):
    with zipfile.ZipFile(zpath) as z: z.extractall(dest_dir)
    os.remove(zpath)

def get_ne_countries():
    shp = os.path.join(CACHE, "ne_110m_admin_0_countries.shp")
    if not os.path.exists(shp):
        z = os.path.join(CACHE, "ne110.zip")
        if fetch("https://naturalearth.s3.amazonaws.com/110m_cultural/"
                 "ne_110m_admin_0_countries.zip", z, "NE 110m countries"):
            unzip(z, CACHE)
    return shp

def get_ne_states():
    shp = os.path.join(CACHE, "ne_10m_admin_1_states_provinces.shp")
    if not os.path.exists(shp):
        z = os.path.join(CACHE, "ne10states.zip")
        if fetch("https://naturalearth.s3.amazonaws.com/10m_cultural/"
                 "ne_10m_admin_1_states_provinces.zip", z, "NE 10m states (~25 MB)"):
            unzip(z, CACHE)
    return shp

def get_gadm_india_l2():
    """GADM v4.1 India Level-2 (districts) as GeoJSON."""
    fname = os.path.join(CACHE, "gadm41_IND_2.json")
    if not os.path.exists(fname):
        z = fname + ".zip"
        if fetch("https://geodata.ucdavis.edu/gadm/gadm4.1/json/"
                 "gadm41_IND_2.json.zip", z, "GADM India districts L2 (~80 MB)"):
            unzip(z, CACHE)
    return fname


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def wm(gdf): return gdf.to_crs(epsg=3857)

def gridlines(ax, xlim, ylim, nl=4, nla=4, fs=7.5):
    ti = Transformer.from_crs("EPSG:3857","EPSG:4326",always_xy=True)
    tf = Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
    lo0,la0 = ti.transform(xlim[0],ylim[0]); lo1,la1 = ti.transform(xlim[1],ylim[1])
    for lon in np.linspace(np.ceil(lo0), np.floor(lo1), nl):
        xm,_ = tf.transform(lon,(la0+la1)/2)
        ax.axvline(xm,color="white",lw=0.5,ls="--",alpha=0.6,zorder=3)
        ax.text(xm,ylim[0],f"{lon:.0f}°E",ha="center",va="bottom",fontsize=fs,color="#222",zorder=6)
    for lat in np.linspace(np.ceil(la0), np.floor(la1), nla):
        _,ym = tf.transform((lo0+lo1)/2,lat)
        ax.axhline(ym,color="white",lw=0.5,ls="--",alpha=0.6,zorder=3)
        lbl = f"{lat:.0f}°N" if lat>=0 else f"{abs(lat):.0f}°S"
        ax.text(xlim[0],ym,lbl,ha="left",va="center",fontsize=fs,color="#222",zorder=6)

def narrow(ax, x=0.93, y=0.10, sz=0.07, fs=12):
    ax.annotate("",xy=(x,y+sz),xytext=(x,y),xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>",color="black",lw=2.2,mutation_scale=18))
    ax.text(x,y-0.025,"N",transform=ax.transAxes,ha="center",va="top",
            fontsize=fs,fontweight="bold",bbox=dict(fc="white",ec="none",pad=1,alpha=0.7))

def sbar(ax):
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(1,units="m",dimension="si-length",
                               length_fraction=0.22,height_fraction=0.014,
                               location="lower left",pad=0.6,color="black",
                               box_color="white",box_alpha=0.75,
                               font_properties={"size":9}))
    except:
        ax.annotate("├──────┤\n 10 km",xy=(0.03,0.05),xycoords="axes fraction",
                    fontsize=8,fontfamily="monospace",
                    bbox=dict(fc="white",ec="grey",alpha=0.8,pad=2))

def plbl(ax, t):
    ax.text(0.015,0.975,t,transform=ax.transAxes,fontsize=13,fontweight="bold",
            va="top",ha="left",zorder=10,bbox=dict(fc="white",ec="none",alpha=0.65,pad=2))

def blbl(ax, t, x, y, fs=19):
    ax.text(x,y,t,fontsize=fs,fontweight="bold",color="white",
            ha="center",va="center",zorder=8,
            path_effects=[pe.withStroke(linewidth=2.5,foreground="black")])

def inv_pts(path, n=4000):
    from rasterio.transform import xy as rxy
    with rasterio.open(path) as s: d=s.read(1); crs=s.crs; t=s.transform
    def sp(r,c):
        np.random.seed(42); idx=np.random.choice(len(r),min(n,len(r)),replace=False)
        xs,ys=rxy(t,r[idx],c[idx])
        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs,ys),crs=crs)
    return sp(*np.where(d==1)), sp(*np.where(d==0))


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("📥 Loading boundaries (Boundary_Cache/ used for all downloads)...")

# Natural Earth
countries  = gpd.read_file(get_ne_countries()).to_crs(4326)
asia       = countries[countries["CONTINENT"]=="Asia"].copy()
india_poly = countries[countries["SOVEREIGNT"]=="India"].copy()
states_all = gpd.read_file(get_ne_states()).to_crs(4326)
india_st   = states_all[states_all["admin"]=="India"].copy()
print(f"   India states (NE 10m): {len(india_st)} ✓")

# GADM Level-2 for India
gadm_l2 = gpd.read_file(get_gadm_india_l2()).to_crs(4326)
print(f"   GADM L2 loaded: {len(gadm_l2)} districts ✓")
print(f"   Columns: {[c for c in gadm_l2.columns if 'NAME' in c.upper()]}")

# Identify correct columns (GADM 4.1 uses NAME_1 for state, NAME_2 for district)
s_col = next((c for c in gadm_l2.columns if c=="NAME_1"), "NAME_1")
d_col = next((c for c in gadm_l2.columns if c=="NAME_2"), "NAME_2")

bihar_dists = gadm_l2[gadm_l2[s_col]=="Bihar"].copy()
khagaria    = gadm_l2[(gadm_l2[s_col]=="Bihar") &
                       (gadm_l2[d_col].str.contains("Khagar",case=False,na=False))].copy()
print(f"   Bihar districts: {len(bihar_dists)} ✓   Khagaria: {len(khagaria)} rows: {khagaria[d_col].tolist()}")

bihar_outline = bihar_dists.dissolve()
other_dists   = bihar_dists[~bihar_dists.geometry.intersects(khagaria.union_all())]

# DEM + inventory
flood_pts, nflood_pts = inv_pts(FLOOD_INV)

dem_data=None; dem_ext=None; dem_crs="EPSG:4326"
try:
    from rasterio.mask import mask as riomask
    from shapely.geometry import mapping

    with rasterio.open(DEM) as s:
        dem_crs = s.crs
        # Clip DEM to Khagaria boundary
        khag_clip = khagaria.to_crs(dem_crs)
        shapes   = [mapping(geom) for geom in khag_clip.geometry]
        clipped, clipped_transform = riomask(s, shapes, crop=True, nodata=np.nan)
        dem_data = clipped[0].astype(float)
        dem_data[dem_data < -9000] = np.nan
        # Rebuild extent: matplotlib imshow needs (left, right, bottom, top)
        from rasterio.transform import array_bounds
        h, w   = dem_data.shape
        bounds = array_bounds(h, w, clipped_transform)
        # array_bounds returns (bottom, left, top, right)
        dem_ext = (bounds[1], bounds[3], bounds[0], bounds[2])
    print(f"   DEM clipped to Khagaria: {dem_data.shape} ✓")
except Exception as e:
    print(f"   DEM skipped: {e}")

# Colormap range: use valid pixels only (within boundary)
vmin = np.nanpercentile(dem_data, 2)  if dem_data is not None else 0
vmax = np.nanpercentile(dem_data, 98) if dem_data is not None else 1
print(f"   DEM range: {vmin:.1f} – {vmax:.1f} m")

# Clip inventory points to Khagaria boundary
try:
    khag_4326 = khagaria.to_crs(4326)
    flood_pts   = gpd.clip(flood_pts.to_crs(4326),   khag_4326).to_crs(flood_pts.crs)
    nflood_pts  = gpd.clip(nflood_pts.to_crs(4326),  khag_4326).to_crs(nflood_pts.crs)
    print(f"   Inventory clipped → flood: {len(flood_pts)}, non-flood: {len(nflood_pts)}")
except Exception as e:
    print(f"   Inventory clip skipped: {e}")

rivers = gpd.read_file(RIVERS).to_crs(dem_crs) if os.path.exists(RIVERS) else None


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL BORDER HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def add_border(ax, lw=1.5, color="black"):
    """Draw a visible box border around an axes using its spine."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)

def gridlines_geo(ax, xlim_geo, ylim_geo, nl=4, nla=4, fs=8, color="#333333"):
    """
    Draw lat/lon gridlines directly on a geographic (EPSG:4326) axes.
    xlim_geo, ylim_geo are in degrees.
    """
    lons = np.linspace(np.ceil(xlim_geo[0]*4)/4,  np.floor(ylim_geo[1]*4)/4,  nl)
    # compute from actual bounds
    lo0,lo1 = xlim_geo; la0,la1 = ylim_geo
    lons = np.linspace(np.ceil(lo0*2)/2,  np.floor(lo1*2)/2,  nl)
    lats = np.linspace(np.ceil(la0*2)/2,  np.floor(la1*2)/2,  nla)
    for lon in lons:
        ax.axvline(lon, color="white", lw=0.6, ls="--", alpha=0.7, zorder=3)
        ax.text(lon, la0, f"{lon:.1f}°E", ha="center", va="bottom",
                fontsize=fs, color=color, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))
    for lat in lats:
        ax.axhline(lat, color="white", lw=0.6, ls="--", alpha=0.7, zorder=3)
        lbl = f"{lat:.1f}°N" if lat >= 0 else f"{abs(lat):.1f}°S"
        ax.text(lo0, lat, lbl, ha="left", va="center",
                fontsize=fs, color=color, zorder=6,
                bbox=dict(fc="white", ec="none", alpha=0.6, pad=1))

def narrow_geo(ax, x=0.95, y=0.06, sz=0.06, fs=11):
    ax.annotate("", xy=(x, y+sz), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0, mutation_scale=16))
    ax.text(x, y-0.02, "N", transform=ax.transAxes, ha="center", va="top",
            fontsize=fs, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1, alpha=0.8))

def sbar_geo(ax, xlim, ylim, loc=(0.03, 0.04), fs=8):
    import math
    lat_c = (ylim[0]+ylim[1])/2
    deg_per_10km = 10 / (111.32 * math.cos(math.radians(lat_c)))
    x0 = xlim[0] + (xlim[1]-xlim[0])*loc[0]
    y0 = ylim[0] + (ylim[1]-ylim[0])*loc[1]
    x1 = x0 + deg_per_10km
    bh = (ylim[1]-ylim[0])*0.007
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0-deg_per_10km*0.1, y0-bh*2.5), deg_per_10km*1.5, bh*6,
        boxstyle="square,pad=0", fc="white", ec="none", alpha=0.8, zorder=9))
    ax.plot([x0,x1],[y0,y0], color="black", lw=3, solid_capstyle="butt", zorder=10)
    ax.plot([x0,x0],[y0-bh,y0+bh], color="black", lw=1.5, zorder=10)
    ax.plot([x1,x1],[y0-bh,y0+bh], color="black", lw=1.5, zorder=10)
    ax.text((x0+x1)/2, y0+bh*2, "10 km", ha="center", va="bottom",
            fontsize=fs, fontweight="bold", zorder=10)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE — 3-panel layout: A (top-left), B (bottom-left), C (right, full height)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n🖼️  Composing 3-panel figure...")

fig = plt.figure(figsize=(20, 12), facecolor="white")

# Outer GridSpec: left column (A+B) | right column (C)
gs_outer = gridspec.GridSpec(
    1, 2, figure=fig,
    width_ratios=[1, 1.6],
    left=0.03, right=0.97, top=0.96, bottom=0.03,
    wspace=0.06
)
# Left column split into A (top) and B (bottom)
gs_left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs_outer[0, 0],
    hspace=0.06
)
axA = fig.add_subplot(gs_left[0])
axB = fig.add_subplot(gs_left[1])
axC = fig.add_subplot(gs_outer[0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# PANEL A — India location map (Web Mercator)
# ─────────────────────────────────────────────────────────────────────────────
print("   Panel A: India...")
wm(asia).plot(ax=axA, color="#DDDDDD", edgecolor="#AAAAAA", lw=0.3, zorder=1)
wm(india_st).plot(ax=axA, color=C["india_f"], edgecolor=C["india_e"], lw=0.35, zorder=2)
wm(india_poly).boundary.plot(ax=axA, color="#111111", lw=1.3, zorder=3)
wm(bihar_outline).plot(ax=axA, color=C["bihar_f"], edgecolor=C["bihar_e"], lw=1.0, zorder=4)
wm(khagaria).plot(ax=axA, color=C["khag_f"], edgecolor=C["khag_e"], lw=1.2, zorder=5)

bA = wm(india_poly).total_bounds
mA = (bA[2]-bA[0])*.06; nA = (bA[3]-bA[1])*.06
axA.set_xlim(bA[0]-mA, bA[2]+mA); axA.set_ylim(bA[1]-nA, bA[3]+nA)

try:
    ctx.add_basemap(axA, source=TILE_PROVIDER, zoom=5, alpha=0.35, zorder=0, attribution_size=5)
except Exception as e: print(f"   Tile A: {e}")

gridlines(axA, (bA[0]-mA, bA[2]+mA), (bA[1]-nA, bA[3]+nA), nl=4, nla=4)
add_border(axA); axA.set_xticks([]); axA.set_yticks([])
plbl(axA, "A.")

iB = wm(india_poly).total_bounds
blbl(axA, "India", (iB[0]+iB[2])/2, (iB[1]+iB[3])/2 - (iB[3]-iB[1])*.05)

axA.legend(handles=[
    mpatches.Patch(facecolor=C["bihar_f"], edgecolor=C["bihar_e"], label="Bihar State"),
    mpatches.Patch(facecolor=C["khag_f"],  edgecolor=C["khag_e"],  label="Khagaria District"),
], loc="lower left", fontsize=8, framealpha=0.9, edgecolor="grey")

# Red locator box
bk  = wm(khagaria).total_bounds
pk  = max(bk[2]-bk[0], bk[3]-bk[1]) * 1.8
axA.add_patch(mpatches.Rectangle(
    (bk[0]-pk/2, bk[1]-pk/2), (bk[2]-bk[0])+pk, (bk[3]-bk[1])+pk,
    lw=1.8, ec="red", fc="none", ls="--", zorder=6))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL B — Bihar districts (Web Mercator)
# ─────────────────────────────────────────────────────────────────────────────
print("   Panel B: Bihar...")
wm(other_dists).plot(ax=axB, color=C["neigh"], edgecolor=C["india_e"], lw=0.4, zorder=2)
wm(bihar_outline).boundary.plot(ax=axB, color=C["bihar_e"], lw=1.6, zorder=4)
wm(khagaria).plot(ax=axB, color=C["khag_f"], edgecolor=C["khag_e"], lw=1.4, zorder=5)

bB = wm(bihar_outline).total_bounds
mB = (bB[2]-bB[0])*.04; nB = (bB[3]-bB[1])*.04
axB.set_xlim(bB[0]-mB, bB[2]+mB); axB.set_ylim(bB[1]-nB, bB[3]+nB)

try:
    ctx.add_basemap(axB, source=TILE_PROVIDER, zoom=8, alpha=0.5, zorder=1, attribution_size=5)
except Exception as e: print(f"   Tile B: {e}")

gridlines(axB, (bB[0]-mB, bB[2]+mB), (bB[1]-nB, bB[3]+nB), nl=4, nla=4)
add_border(axB); axB.set_xticks([]); axB.set_yticks([])
plbl(axB, "B.")

tf2 = Transformer.from_crs("EPSG:4326","EPSG:3857",always_xy=True)
for _, row in bihar_dists.iterrows():
    if row[d_col] in ["Patna","Muzaffarpur","Bhagalpur","Darbhanga","Gaya","Purnia","Saharsa"]:
        xm, ym = tf2.transform(row.geometry.centroid.x, row.geometry.centroid.y)
        axB.text(xm, ym, row[d_col], fontsize=6.5, ha="center", fontstyle="italic",
                 color="#333", zorder=6, bbox=dict(fc="white",ec="none",alpha=0.55,pad=1))

kxm,kym = tf2.transform(float(khagaria.geometry.centroid.x.iloc[0]),
                         float(khagaria.geometry.centroid.y.iloc[0]))
axB.text(kxm, kym, "Khagaria", fontsize=9.5, ha="center", fontweight="bold",
         color="white", zorder=7,
         bbox=dict(boxstyle="round,pad=0.25", fc=C["khag_f"], alpha=0.9, ec="none"))

bk2 = wm(khagaria).total_bounds
pk2 = max(bk2[2]-bk2[0], bk2[3]-bk2[1]) * 0.35
axB.add_patch(mpatches.Rectangle(
    (bk2[0]-pk2, bk2[1]-pk2), (bk2[2]-bk2[0])+2*pk2, (bk2[3]-bk2[1])+2*pk2,
    lw=1.8, ec="red", fc="none", ls="--", zorder=6))


# ─────────────────────────────────────────────────────────────────────────────
# PANEL C — DEM + Flood Inventory (combined, large, geographic CRS)
# ─────────────────────────────────────────────────────────────────────────────
print("   Panel C: Khagaria DEM + Inventory...")

# Work entirely in EPSG:4326 for clean lat/lon gridlines
khag_4326  = khagaria.to_crs(4326)
bC_geo     = khag_4326.total_bounds          # (minlon, minlat, maxlon, maxlat)
pC_lon     = (bC_geo[2]-bC_geo[0]) * 0.05
pC_lat     = (bC_geo[3]-bC_geo[1]) * 0.05
xlim_C     = (bC_geo[0]-pC_lon, bC_geo[2]+pC_lon)
ylim_C     = (bC_geo[1]-pC_lat, bC_geo[3]+pC_lat)

axC.set_xlim(xlim_C); axC.set_ylim(ylim_C)
axC.set_aspect("equal")

# ── DEM via rasterio.plot.show (handles extent automatically) ────────────────
if dem_data is not None:
    # Reproject dem to 4326 for display
    import rasterio.warp as rwarp
    from rasterio.crs import CRS as RioCRS

    try:
        with rasterio.open(DEM) as src:
            dst_crs = RioCRS.from_epsg(4326)
            transform_4326, w4326, h4326 = rwarp.calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            dem_4326 = np.empty((h4326, w4326), dtype=np.float32)
            rwarp.reproject(
                source      = rasterio.band(src, 1),
                destination = dem_4326,
                src_transform  = src.transform,
                src_crs        = src.crs,
                dst_transform  = transform_4326,
                dst_crs        = dst_crs,
                resampling     = rwarp.Resampling.bilinear,
                src_nodata     = src.nodata,
                dst_nodata     = np.nan
            )
        dem_4326 = dem_4326.astype(float)
        dem_4326[dem_4326 < -9000] = np.nan

        # Clip to Khagaria in 4326
        from rasterio.mask import mask as riomask2
        from rasterio.transform import from_bounds
        from shapely.geometry import mapping as shp_mapping
        import tempfile, uuid

        # Write reprojected DEM to a temp file so we can mask it
        tmp_path = os.path.join(CACHE, "dem_4326_tmp.tif")
        with rasterio.open(
            tmp_path, "w", driver="GTiff",
            height=h4326, width=w4326, count=1,
            dtype=dem_4326.dtype, crs=dst_crs,
            transform=transform_4326, nodata=np.nan
        ) as dst:
            dst.write(dem_4326, 1)

        with rasterio.open(tmp_path) as src2:
            shapes_4326 = [shp_mapping(g) for g in khag_4326.geometry]
            dem_clip, dem_clip_tf = riomask2(src2, shapes_4326, crop=True, nodata=np.nan)
            dem_clip = dem_clip[0].astype(float)
            dem_clip[dem_clip < -9000] = np.nan
            hc, wc = dem_clip.shape
            from rasterio.transform import array_bounds as ab
            b4 = ab(hc, wc, dem_clip_tf)
            ext_4326 = (b4[1], b4[3], b4[0], b4[2])  # left, right, bottom, top

        vmin_c = np.nanpercentile(dem_clip, 2)
        vmax_c = np.nanpercentile(dem_clip, 98)
        print(f"   DEM 4326 range: {vmin_c:.1f}–{vmax_c:.1f} m  shape:{dem_clip.shape}")

        imC = axC.imshow(dem_clip, extent=ext_4326, cmap=DEM_CMAP, origin="upper",
                         interpolation="bilinear", vmin=vmin_c, vmax=vmax_c,
                         zorder=1, aspect="auto")

        # Colorbar inside right side of panel C
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axC)
        cax = divider.append_axes("right", size="3%", pad=0.08)
        cb = plt.colorbar(imC, cax=cax)
        cb.set_label("Elevation (m)", fontsize=10, labelpad=6)
        cb.ax.tick_params(labelsize=8)

    except Exception as e:
        print(f"   DEM reproject failed: {e}; using fallback")
        axC.set_facecolor("#C0D8C0")
else:
    axC.set_facecolor("#C0D8C0")

# ── Boundary ──────────────────────────────────────────────────────────────────
khag_4326.boundary.plot(ax=axC, color="black", lw=1.8, zorder=5)

# ── Rivers ────────────────────────────────────────────────────────────────────
if rivers is not None:
    rivers.to_crs(4326).plot(ax=axC, color=C["river"], lw=1.0, alpha=0.9, zorder=4)

# ── Flood inventory points ────────────────────────────────────────────────────
try:
    fp4  = flood_pts.to_crs(4326)
    nfp4 = nflood_pts.to_crs(4326)
    fp4  = gpd.clip(fp4,  khag_4326)
    nfp4 = gpd.clip(nfp4, khag_4326)
    fp4.plot( ax=axC, color=C["flood"],  markersize=7, marker="o", alpha=0.85, zorder=7)
    nfp4.plot(ax=axC, color=C["nflood"], markersize=7, marker="^", alpha=0.85, zorder=7)
    print(f"   Inventory plotted: flood={len(fp4)}, non-flood={len(nfp4)}")
except Exception as e:
    print(f"   Inventory plot error: {e}")

# ── Lat/Lon gridlines (native geo axes) ───────────────────────────────────────
gridlines_geo(axC, xlim_C, ylim_C, nl=4, nla=5, fs=8.5)

# ── Decorations ──────────────────────────────────────────────────────────────
add_border(axC); axC.set_xticks([]); axC.set_yticks([])
plbl(axC, "C.")
narrow_geo(axC, x=0.96, y=0.06, sz=0.06)

# Approximate scale bar: ~0.1° lon ≈ 10 km at 26°N
sbar_geo(axC, xlim_C, ylim_C)

# District label
blbl(axC, "Khagaria",
     (bC_geo[0]+bC_geo[2])/2,
     (bC_geo[1]+bC_geo[3])/2 + (bC_geo[3]-bC_geo[1])*0.22,
     fs=20)

# Inventory legend
axC.legend(handles=[
    mpatches.Patch(facecolor=C["flood"],  edgecolor="black", lw=0.5, label="Flood Inventory"),
    mpatches.Patch(facecolor=C["nflood"], edgecolor="black", lw=0.5, label="Non-Flood Inventory"),
] + ([mpatches.Patch(facecolor=C["river"], edgecolor="none", label="Rivers")] if rivers is not None else []),
loc="lower right", fontsize=9.5, framealpha=0.92, edgecolor="grey",
title="Inventory", title_fontsize=9)


# ─── SAVE ─────────────────────────────────────────────────────────────────────
for ext, kw in [(".png", dict(dpi=300)), (".pdf", dict(format="pdf"))]:
    p = os.path.join(OUT_DIR, f"Fig01_Study_Area_Final{ext}")
    plt.savefig(p, bbox_inches="tight", facecolor="white", **kw)
    print(f"✅ {p}")
plt.close()