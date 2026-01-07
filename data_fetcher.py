"""
data_fetcher.py
----------------
Downloads satellite images from Mapbox
and fetches transport proximity features using OSMNX
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import osmnx as ox
from sklearn.neighbors import BallTree

# ======================
# CONFIG
# ======================
MAPBOX_TOKEN = "YOUR_TOKEN_HERE"   # keep private
CSV_FILE = "data/train.csv"
IMAGE_DIR = "data/mapbox_images"
OUTPUT_CSV = "data/train_with_transport.csv"

ZOOM = 18
IMG_SIZE = "512x512"
SCALE = 2
STYLE = "satellite-v9"

SEARCH_RADIUS_M = 40000
SLEEP_TIME = 0.3

# ======================
# SATELLITE IMAGE DOWNLOAD
# ======================
def download_satellite_images():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    df = pd.read_csv(CSV_FILE, dtype={"id": str})

    failed_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid, lat, lon = row["id"], row["lat"], row["long"]
        out_path = f"{IMAGE_DIR}/{pid}.png"

        if os.path.exists(out_path):
            continue

        url = (
            f"https://api.mapbox.com/styles/v1/mapbox/{STYLE}/static/"
            f"{lon},{lat},{ZOOM}/{IMG_SIZE}@{SCALE}x"
            f"?access_token={MAPBOX_TOKEN}"
        )

        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            else:
                failed_ids.append(pid)
        except:
            failed_ids.append(pid)

        time.sleep(SLEEP_TIME)

    if failed_ids:
        pd.Series(failed_ids).to_csv("failed_ids.csv", index=False)

# ======================
# TRANSPORT FEATURES
# ======================
def extract_transport_features():
    df = pd.read_csv(CSV_FILE, dtype={"id": str})
    house_coords = df[['lat', 'long']].values

    center = (df.lat.mean(), df.long.mean())

    metro = ox.features_from_point(
        center,
        tags={"railway": ["station"], "station": "subway"},
        dist=SEARCH_RADIUS_M
    )

    rail = ox.features_from_point(
        center,
        tags={"railway": ["station", "halt"]},
        dist=SEARCH_RADIUS_M
    )

    airport = ox.features_from_point(
        center,
        tags={"aeroway": "aerodrome"},
        dist=SEARCH_RADIUS_M
    )

    def extract_coords(gdf):
        gdf = gdf[gdf.geometry.notnull()]
        centroids = gdf.geometry.centroid
        return np.column_stack((centroids.y, centroids.x))

    def build_tree(coords):
        return BallTree(np.radians(coords), metric="haversine")

    def nearest_distance(tree, points):
        dist, _ = tree.query(np.radians(points), k=1)
        return dist.flatten() * 6371000

    df["dist_to_metro"] = nearest_distance(
        build_tree(extract_coords(metro)), house_coords
    )
    df["dist_to_rail"] = nearest_distance(
        build_tree(extract_coords(rail)), house_coords
    )
    df["dist_to_airport"] = nearest_distance(
        build_tree(extract_coords(airport)), house_coords
    )

    df.to_csv(OUTPUT_CSV, index=False)

# ======================
if __name__ == "__main__":
    download_satellite_images()
    extract_transport_features()
