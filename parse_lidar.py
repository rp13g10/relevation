import os
from array import array
from typing import List, Set

import numpy as np
import pandas as pd
import rasterio as rio
import shapefile as shp
from glob import glob
from tqdm import tqdm

cur_dir = os.path.abspath(os.path.dirname(__file__))


def get_available_folders() -> Set[str]:
    all_lidar_dirs = glob(os.path.join(cur_dir, "data/LIDAR-DTM-1m-*"))
    return set(all_lidar_dirs)


def load_lidar_from_folder(lidar_dir: str) -> np.ndarray:
    tif_loc = glob(os.path.join(lidar_dir, "*.tif"))[0]
    with rio.open(tif_loc) as tif:
        lidar = tif.read()

    lidar = lidar[0]
    return lidar


def load_bbox_from_folder(lidar_dir: str) -> array:
    sf_loc = glob(os.path.join(lidar_dir, "index/*.shp"))[0]

    with shp.Reader(sf_loc) as sf:
        bbox = sf.bbox

    return bbox


# TODO: Switch this over to an array operation to improve performance
def explode_lidar(lidar: np.ndarray, bbox: array) -> pd.DataFrame:
    base_easting = bbox[0]
    base_northing = bbox[1]

    size_e, size_s = lidar.shape

    records = []
    for offset_e in tqdm(range(size_e), "Processing File"):
        for offset_n in range(size_s):
            easting = base_easting + offset_e
            northing = base_northing + offset_n

            # Coords start bottom left, array indexes start top left
            offset_s = -(offset_n + 1)
            elevation = lidar[offset_s, offset_e]

            record = (int(easting), int(northing), elevation)

            records.append(record)

    lidar_df = pd.DataFrame.from_records(
        records, columns=["easting", "northing", "elevation"]
    )

    return lidar_df
