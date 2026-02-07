#!/usr/bin/env python3
import os
import sys
import tempfile

import numpy as np
import pygmt
from pygmt.datasets import load_earth_relief

from obspy.clients.fdsn import Client
from obspy import UTCDateTime

# ------------------------------------------------------------
# CONFIG / CONSTANTS
# ------------------------------------------------------------

# Environment bootstrap (PyGMT/GMT pathing & suppress GDAL plugin noise)
os.environ.setdefault("GMT_LIBRARY_PATH", os.path.join(sys.prefix, "lib"))
os.environ.setdefault("GMT_SHAREDIR", os.path.join(sys.prefix, "share", "gmt"))
os.environ.pop("GDAL_DRIVER_PATH", None)  # avoid mismatched GDAL plugins from base Anaconda
os.environ.setdefault("GMT_REMOTE_CACHE", os.path.join(os.path.expanduser("~"), ".gmt", "cache"))
os.makedirs(os.environ["GMT_REMOTE_CACHE"], exist_ok=True)

# IRIS / metadata config
FDSN_SOURCE = "IRIS"
NETWORK     = "SN"
STATION_PAT = "IS*"     # all NTS infrasound array elements
CHANNEL_PAT = "*DF"

# Time window covering all SPE shots
T_START = UTCDateTime(2011, 5, 3, 0, 0, 0)
T_END   = UTCDateTime(2016,10,12,23,59,59)

# Explosion epicenter (common SPE GT location) – currently not plotted
EXP_LAT  = 37.2212
EXP_LON  = -116.0609
EXP_NAME = "SPE"

# Output
SAVE      = False
SAVE_PATH = "/Users/serinawang/Desktop/SPE1_LLNL"
FNAME     = "SPE_station_map.png"

# Region (lon_min, lon_max, lat_min, lat_max)
# Set to None to auto-fit around stations (+ epicenter, if included).
MANUAL_REGION = [-116.075, -116.033, 37.17, 37.23]  # [lon_min, lon_max, lat_min, lat_max]

# Map appearance (keep same relative size as PE1A figure)
PROJECTION = "M14c"
LABEL_FONT = "10p,Helvetica,black"
LABEL_OFFSET = "-0.25c/0.15c"
LABEL_JUSTIFY = "RB"
SCALE_BAR = "jBL+o0.5c/0.7c+w1k+u"  # bottom-left scale bar
RASTER_TRANSPARENCY = 40  # 0 = fully opaque topo

# Symbol sizes/styles
SYM_HEX  = "h0.50c"  # infrasound array
SYM_STAR = "a0.50c"  # explosion

FILL_INFRA = "black"
FILL_STAR  = "red"
PEN_SYM    = "black"

# Topography preference (tries in order; first that works is used)
TOPO_PREF = ("01s", "03s", "15s")


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def collect_stations_from_iris(
    source=FDSN_SOURCE,
    network=NETWORK,
    station=STATION_PAT,
    channel=CHANNEL_PAT,
    starttime=T_START,
    endtime=T_END,
):
    """
    Query IRIS FDSN for SN.IS* stations and return
    dict: {sta: {"lat": ..., "lon": ..., "network": ...}}
    """
    client = Client(source)

    try:
        inv = client.get_stations(
            network=network,
            station=station,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            level="station",
        )
    except Exception as e:
        raise RuntimeError(f"FDSN get_stations failed: {e}")

    stations = {}
    for net in inv:
        for sta in net.stations:
            code = sta.code     # e.g., IS31, IS41, ...
            lat = sta.latitude
            lon = sta.longitude
            stations[code] = {
                "lat": float(lat),
                "lon": float(lon),
                "network": net.code,
            }
    return stations


def group_arrays(infra_raw):
    """
    Collapse ISxx element stations into array IDs like IS1, IS2, ..., IS6.

    Example:
      IS31, IS32, IS33 -> array 'IS3'
      IS41, IS42 -> array 'IS4'

    Returns dict: {array_id: {"lat": mean_lat, "lon": mean_lon}}
    """
    groups = {}
    for sta, info in infra_raw.items():
        # Default: use full station code if not in IS## pattern
        array_id = sta
        # Simple pattern for NTS arrays: 'IS' + array_number + element_number
        # e.g., IS31 -> IS3, IS41 -> IS4
        if len(sta) >= 3 and sta.startswith("IS") and sta[2].isdigit():
            array_id = sta[:3]  # 'IS3', 'IS4', etc.

        groups.setdefault(array_id, {"lats": [], "lons": []})
        groups[array_id]["lats"].append(info["lat"])
        groups[array_id]["lons"].append(info["lon"])

    grouped = {}
    for arr, vals in groups.items():
        grouped[arr] = {
            "lat": float(np.mean(vals["lats"])),
            "lon": float(np.mean(vals["lons"])),
        }
    return grouped


def compute_region_from_points(lats, lons, pad_min=0.01, pad_frac=0.01):
    """
    Build padded region [lon_min, lon_max, lat_min, lat_max]
    around given lat/lon lists.

    pad_min and pad_frac are set smaller than before to 'zoom in' a bit.
    """
    lats = list(lats)
    lons = list(lons)
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    dlat = max(lat_max - lat_min, pad_min)
    dlon = max(lon_max - lon_min, pad_min)
    lat_pad = max(dlat * pad_frac, 0.02)
    lon_pad = max(dlon * pad_frac, 0.02)

    return [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad]


def load_topography(region):
    """
    Try multiple earth_relief resolutions in TOPO_PREF order.
    Return the first that works.
    """
    last_err = None
    for res in TOPO_PREF:
        try:
            return load_earth_relief(resolution=res, region=region)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError("Unable to load earth_relief for region. Last error: %s" % last_err)


def write_legend_and_get_path():
    """
    Create a temporary legend spec file and return its path.
    Caller should remove the file after fig.legend().
    """
    legend_entries = f"""\
S 0.3c a 0.35c {FILL_STAR} 0.5p,{PEN_SYM} 0.9c Explosion ground truth
S 0.3c h 0.32c {FILL_INFRA} 0.5p,{PEN_SYM} 0.9c Infrasound array
"""
    tf = tempfile.NamedTemporaryFile("w", suffix=".leg", delete=False)
    with tf:
        tf.write(legend_entries)
    return tf.name


def add_label(fig, lon, lat, text):
    """Place a station/array label next to its symbol."""
    fig.text(
        x=lon,
        y=lat,
        text=text,
        font=LABEL_FONT,
        offset=LABEL_OFFSET,
        justify=LABEL_JUSTIFY,
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    # GMT defaults (match PE1A style)
    pygmt.config(
        MAP_FRAME_TYPE="plain",
        FORMAT_GEO_MAP="ddd.xx",
        MAP_ANNOT_OBLIQUE="0",
        FONT_LABEL="8p,Helvetica,black",
        MAP_SCALE_HEIGHT="7p",
    )

    # Make scale bar line thicker
    pygmt.config(MAP_TICK_PEN_PRIMARY="1.5p")  # try 2p, 3p, etc.

    # Collect element station coordinates from IRIS
    infra_raw = collect_stations_from_iris()
    if not infra_raw:
        raise RuntimeError("No SN.IS* stations found from IRIS query.")

    # Group to array centroids and simplified labels (IS1, IS2, ...)
    infra = group_arrays(infra_raw)

    # Region: MANUAL_REGION > auto-fit around arrays (+ epicenter if desired)
    if MANUAL_REGION is not None:
        region = MANUAL_REGION
    else:
        all_lats = [v["lat"] for v in infra.values()]
        all_lons = [v["lon"] for v in infra.values()]
        # If you want epicenter to influence window, uncomment below
        # if EXP_LAT is not None and EXP_LON is not None:
        #     all_lats.append(EXP_LAT)
        #     all_lons.append(EXP_LON)
        region = compute_region_from_points(all_lats, all_lons)

    fig = pygmt.Figure()

    # Background topo
    try:
        grid = load_topography(region)
        fig.grdimage(
            grid,
            region=region,
            projection=PROJECTION,
            shading=True,
            transparency=RASTER_TRANSPARENCY,
            frame=["xa0.02.02+lLongitude", "ya0.02.02+lLatitude"],
        )
    except Exception as e:
        print(f"[WARN] Topo load failed: {e}\nFalling back to coast basemap.")
        fig.basemap(region=region, projection=PROJECTION,
                    frame=["xaf+lLongitude", "yaf+lLatitude"])
        fig.coast(land="lightgray", water="lightblue", shorelines=True)

    # Plot arrays as single hexagons at centroids with numeric labels only
    for arr, info in sorted(infra.items()):
        fig.plot(
            x=info["lon"], y=info["lat"],
            style=SYM_HEX,
            fill=FILL_INFRA,
            pen=PEN_SYM,
        )

        # Convert 'IS3' -> '3', 'IS4' -> '4', else keep original
        label_text = arr
        if arr.startswith("IS") and len(arr) >= 3 and arr[2].isdigit():
            label_text = arr[2]

        # add_label(fig, info["lon"], info["lat"], label_text)

    # Explosion epicenter (currently off)
    if EXP_LAT is not None and EXP_LON is not None:
        fig.plot(x=EXP_LON, y=EXP_LAT, style=SYM_STAR, fill=FILL_STAR, pen=PEN_SYM)
        # if EXP_NAME:
        #     fig.text(
        #         x=region[0],
        #         y=region[3],
        #         text=EXP_NAME,
        #         font="10p,Helvetica-Bold",
        #         justify="LT",
        #         offset="0.15c/0.1c",
        #     )

    # Scale bar & legend
    fig.basemap(map_scale=SCALE_BAR)

    legend_file = write_legend_and_get_path()
    fig.legend(spec=legend_file, position="JTR+jTR+o0.2c/0.2c", box="+gwhite+p0.5p")
    try:
        os.remove(legend_file)
    except OSError:
        pass

    # Show & save
    fig.show()
    if SAVE:
        os.makedirs(SAVE_PATH, exist_ok=True)
        out = os.path.join(SAVE_PATH, FNAME)
        fig.savefig(out, dpi=450)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()

