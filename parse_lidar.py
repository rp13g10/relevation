import os
from array import array
from typing import Set

import numpy as np
import pandas as pd
import rasterio as rio
import shapefile as shp
from glob import glob
from tqdm import tqdm

cur_dir = os.path.abspath(os.path.dirname(__file__))

# NOTE: This script will need to run in a Docker container with data directory
#       mounted as a volume


def get_available_folders() -> Set[str]:
    all_lidar_dirs = glob(os.path.join(cur_dir, "data/LIDAR-DTM-1m-*"))
    return set(all_lidar_dirs)


def load_lidar_from_folder(lidar_dir: str) -> np.ndarray:
    tif_loc = glob(os.path.join(lidar_dir, "*.tif"))[0]
    with rio.open(tif_loc) as tif:
        lidar = tif.read()

    lidar = lidar[0]
    return lidar


def load_bbox_from_folder(lidar_dir: str) -> np.ndarray:
    sf_loc = glob(os.path.join(lidar_dir, "index/*.shp"))[0]

    with shp.Reader(sf_loc) as sf:
        bbox = sf.bbox

    bbox = np.array(bbox, dtype=int)

    return bbox


def explode_lidar(lidar: np.ndarray, bbox: np.ndarray) -> pd.DataFrame:
    # Get array dimensions
    size_e, size_s = lidar.shape

    # Collapse elevations to 1 dimension, left to right then top to bottom
    elevations = lidar.flatten(order="C")

    # Repeat eastings by array (A, B, A, B)
    eastings = np.tile(range(bbox[0], bbox[2]), size_s).astype("int32")

    # Repeat northings by element (A, A, B, B)
    northings = np.repeat(range(bbox[3] - 1, bbox[1] - 1, -1), size_e).astype(
        "int32"
    )

    # Create dataframe from columns
    lidar_df = pd.DataFrame.from_dict(
        {"easting": eastings, "northing": northings, "elevation": elevations},
        orient="columns",
    )
    return lidar_df


def add_partition_keys(lidar_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: Decide on best way to set up these partitions, current proposal
    #       results in 1m per e/n partition pair
    lidar_df.loc[:, "easting_ptn"] = lidar_df["easting"] // 1000
    lidar_df.loc[:, "northing_ptn"] = lidar_df["northing"] // 1000
    return lidar_df


def store_lidar(Lidar_df: pd.DataFrame):
    """Will store dataframe contents to ScyllaDB"""

    # Base code: https://stackoverflow.com/questions/49108809/how-to-insert-pandas-dataframe-into-cassandra
    # Split by partition rather than random allocation
    pass
