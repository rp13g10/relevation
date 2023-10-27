import os
import re
from functools import lru_cache
from glob import glob
from typing import Set, Tuple, Iterator

import numpy as np
import pandas as pd
import rasterio as rio
import shapefile as shp

cur_dir = os.path.abspath(os.path.dirname(__file__))


def get_available_folders() -> Set[str]:
    """Get a list of all of the data folders which are available within the
    data directory of this package. Data must have been downloaded from the
    DEFRA website:
      https://environment.data.gov.uk/DefraDataDownload/?Mode=survey
    Data should be from the composite DTM layer at 1m resolution, and folder
    names should match the 'LIDAR-DTM-1m-*' pattern. Any zip archives should be
    extracted before running this script.

    Raises:
        FileNotFoundError: If no folders matching the above pattern are found,
          an error will be raised.

    Returns:
        Set[str]: A set containing the absolute path to each data folder
    """
    all_lidar_dirs = glob(os.path.join(cur_dir, "../data/LIDAR-DTM-1m-*"))
    if not all_lidar_dirs:
        raise FileNotFoundError("No files found in data directory!")
    return set(all_lidar_dirs)


def load_lidar_from_folder(lidar_dir: str) -> np.ndarray:
    """For a given data folder, read in the contents of the .tif file within as
    a numpy array.

    Args:
        lidar_dir (str): The location of the data folder to be loaded

    Returns:
        np.ndarray: The contents of the .tif file within the provided data
          folder. Each file represents an area of 5km^2, so the shape of this
          array will be 5000*5000
    """
    tif_loc = glob(os.path.join(lidar_dir, "*.tif"))[0]
    with rio.open(tif_loc) as tif:
        lidar = tif.read()

    lidar = lidar[0]
    return lidar


def load_bbox_from_folder(lidar_dir: str) -> np.ndarray:
    """For a given data folder, read in the contents of the .shp file within as
    a numpy array of length 4.

    Args:
        lidar_dir (str): The location of the data folder to be loaded

    Returns:
        np.ndarray: The contents of the .shp file within the provided data
          folder. Will have 4 elements corresponding to the physical area
          represented by the corresponding .tif file in this folder.
    """
    sf_loc = glob(os.path.join(lidar_dir, "index/*.shp"))[0]

    with shp.Reader(sf_loc) as sf:
        bbox = sf.bbox

    bbox = np.array(bbox, dtype=int)

    return bbox


@lru_cache(16)
def generate_file_id(lidar_dir: str) -> str:
    """Extract the OS grid reference for a given LIDAR file based on its full
    location on the filesystem.

    Args:
        lidar_dir (str): The full path to a LIDAR file, expected format is
          /some/path/relevation/data/LIDAR-DTM-1m-YYYY-XXDDxx

    Returns:
        str: The OS grid reference for the file, expected format is
          XXDDxx (e.g. SU20ne)
    """

    id_match = re.search(r"[A-Z][A-Z]\d\d[a-z][a-z]$", lidar_dir)

    if id_match:
        file_id = id_match.group(0)
        return file_id

    raise ValueError(
        (
            "Unable to extract grid reference from provided lidar_dir: "
            + lidar_dir
        )
    )


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
    lidar_df.loc[:, "easting_ptn"] = lidar_df["easting"] // 100
    lidar_df.loc[:, "northing_ptn"] = lidar_df["northing"] // 100
    return lidar_df


def add_file_ids(lidar_df: pd.DataFrame, lidar_dir: str) -> pd.DataFrame:
    lidar_df.loc[:, "file_id"] = generate_file_id(lidar_dir)

    return lidar_df


def iter_dfs() -> Iterator[Tuple[pd.DataFrame, str]]:
    lidar_dirs = get_available_folders()

    for lidar_dir in lidar_dirs:
        lidar = load_lidar_from_folder(lidar_dir)
        bbox = load_bbox_from_folder(lidar_dir)
        file_id = generate_file_id(lidar_dir)

        lidar_df = explode_lidar(lidar, bbox)
        lidar_df = add_partition_keys(lidar_df)
        lidar_df = add_file_ids(lidar_df, lidar_dir)

        yield lidar_df, file_id
