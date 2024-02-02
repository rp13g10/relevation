import os
import re
from functools import lru_cache
from glob import glob
from typing import Set, Tuple, Iterator

import numpy as np
import pandas as pd
import rasterio as rio
from tqdm.contrib.concurrent import process_map

cur_dir = os.path.abspath(os.path.dirname(__file__))


def get_available_folders(data_dir: str) -> Set[str]:
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
    all_lidar_dirs = glob(
        os.path.join(data_dir, "lidar/lidar_composite_dtm-*")
    )
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
    tfw_loc = glob(os.path.join(lidar_dir, "*.tfw"))[0]

    with open(tfw_loc, "r", encoding="utf8") as fobj:
        tfw = fobj.readlines()

    easting_min = int(float(tfw[4].strip()))
    easting_max = easting_min + 5000

    northing_max = int(float(tfw[5].strip())) + 1
    northing_min = northing_max - 5000

    bbox = np.array(
        [easting_min, northing_min, easting_max, northing_max], dtype=int
    )

    return bbox


@lru_cache(16)
def generate_file_id(lidar_dir: str) -> str:
    """Extract the OS grid reference for a given LIDAR file based on its full
    location on the filesystem.

    Args:
        lidar_dir (str): The full path to a LIDAR file, expected format is
          /some/path/relevation/data/lidar_composite_dtm_YYYY-1-XXDDxx

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
    """Parse a given array containing LIDAR data, and the corresponding
    bounding box. Returns a long-form dataframe containg eastings, northings
    and elevations.

    Args:
        lidar (np.ndarray): An array containing LIDAR data
        bbox (np.ndarray): The bounding box corresponding to the provided
          lidar array

    Returns:
        pd.DataFrame: A dataframe containing the 'easting', 'northing' and
          'elevation' columns
    """
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
    """Generate a new 'easting_ptn' and 'northing_ptn' column in the LIDAR
    dataframe. These will be used for the partitioning of data within the
    database, speeding up retrieval of data.
    Partition columns are derived by dividing the corresponding coordinate by
    100 and truncating the output.

    Args:
        lidar_df (pd.DataFrame): A dataframe containing LIDAR data

    Returns:
        pd.DataFrame: The input dataset, with additional 'easting_ptn' and
          'northing_ptn' columns.
    """
    # TODO: Decide on best way to set up these partitions, current proposal
    #       results in 1m per e/n partition pair
    lidar_df.loc[:, "easting_ptn"] = lidar_df["easting"] // 100
    lidar_df.loc[:, "northing_ptn"] = lidar_df["northing"] // 100
    return lidar_df


def add_file_ids(lidar_df: pd.DataFrame, lidar_dir: str) -> pd.DataFrame:
    """Generate a file ID for a given file name, and store it in the provided
    dataframe under the 'file_id' column name. The file ID will be the OS grid
    reference for the provided file name. For example, lidar_composite_dtm_2022-1-SU20ne
    would generate a file ID of SU20ne.

    Args:
        lidar_df (pd.DataFrame): A dataframe containing LIDAR data
        lidar_dir (str): The name of the file from which `lidar_df` was created

    Returns:
        pd.DataFrame: The input dataset, with an additional 'file_id' column
    """
    lidar_df.loc[:, "file_id"] = generate_file_id(lidar_dir)

    return lidar_df


def parse_lidar_folder(lidar_dir):
    lidar = load_lidar_from_folder(lidar_dir)
    bbox = load_bbox_from_folder(lidar_dir)

    lidar_df = explode_lidar(lidar, bbox)
    lidar_df = add_partition_keys(lidar_df)
    lidar_df = add_file_ids(lidar_df, lidar_dir)

    return lidar_df
