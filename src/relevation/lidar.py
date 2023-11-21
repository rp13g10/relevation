"""
Defines utility functions which facilitate the retrieval of elevation
information for a given latitude/longitude.
"""

import os
from array import array
from functools import lru_cache
from glob import glob

# from multiprocessing import Manager, Process
from typing import Dict, List, Tuple

# from io import BytesIO

import numpy as np
import redis
import rasterio as rio
import shapefile as shp

from bng_latlon import WGS84toOSGB36

# TODO: Create separate code env for this w. updated requirements file

# Determine required file
# Fetch required file
cur_dir = os.path.abspath(os.path.dirname(__file__))

# Associate each bounding box with key (probably directory)
# Dict/cache holds data
# manager = Manager()
# lidar_dict = manager.dict()
redis_cli = redis.Redis(host="localhost", port=6379)


@lru_cache(1)
def get_all_bboxes() -> Dict[str, List[int]]:
    """Return a dictionary containing the names of all folders within the data
    directory which contain LIDAR data. For each directory, the corresponding
    bounding box is stored. This allows the user to quickly determine the
    correct LIDAR file to use for a particular easting/northing.

    Returns:
        Dict[str, List[int]]: Maps each folder containing LIDAR data to the
          corresponding bounding box.
    """

    bboxes = redis_cli.hgetall("bboxes")
    if not bboxes:
        bboxes = {}
        all_lidar_dirs = glob(os.path.join(cur_dir, "data/LIDAR-DTM-1m-*"))
        for lidar_dir in all_lidar_dirs:
            # Locate shapefile
            sf_loc = glob(os.path.join(lidar_dir, "index/*.shp"))[0]

            # Fetch boundaries
            with shp.Reader(sf_loc) as sf:
                bbox = sf.bbox
            bbox = np.array(bbox, dtype="int")
            bbox = bbox.tobytes()

            # Store to dictionary
            bboxes[lidar_dir.encode("utf8")] = bbox
        redis_cli.hmset("bboxes", bboxes)
    else:
        bboxes = {
            key.decode("utf8"): np.frombuffer(value, dtype="int")
            for key, value in bboxes.items()
        }

    return bboxes


@lru_cache(1)
def fetch_lidar_from_folder(lidar_dir: str) -> array:
    """For a provided folder containing LIDAR data, read the contents of the
    contained TIF file, which is an array containing the elevations for each
    point within an area of 5000m x 5000m.

    Args:
        lidar_dir (str): The location of a folder containing LIDAR data, this
          will typically be one of the keys found within the output of
          get_all_bboxes()

    Returns:
        np.ndarray: Contains the elevations for each point within an area of
          5000m x 5000m
    """

    lidar_buffer = redis_cli.get(lidar_dir)
    if lidar_buffer is None:
        tif_loc = glob(os.path.join(lidar_dir, "*.tif"))[0]
        with rio.open(tif_loc) as tif:
            lidar = tif.read(1)

        lidar_buffer = lidar.tobytes()
        redis_cli.set(lidar_dir, lidar_buffer)
    else:
        lidar = np.frombuffer(lidar_buffer, dtype="float32")
        lidar = lidar.reshape(5000, 5000)

    return lidar


def get_lidar_for_bng_reference(
    easting: float, northing: float
) -> Tuple[array, List[int]]:
    """For a given BNG reference, retrieve the LIDAR array which contains its
    elevation data and the corresponding bounding box.

    Args:
        easting (float): BNG Easting
        northing (float): BNG Northing

    Returns:
        Tuple[np.ndarray, List[int]]: The lidar data for the provided grid
          reference, and the corresponding bounding box.
    """

    # Fetch all available folders & boundaries
    all_bboxes = get_all_bboxes()

    # Find first available folder with boundaries that cover target grid ref
    try:
        lidar_dir = next(
            lidar_dir
            for lidar_dir, bbox in all_bboxes.items()
            if bbox[0] <= easting
            and bbox[2] >= easting
            and bbox[1] <= northing
            and bbox[3] >= northing
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            (
                "Unable to retrieve LIDAR information for: "
                f"{easting:.2f}, {northing:.2f}"
            )
        ) from exc

    # Fetch boundaries for desired folder
    bbox = all_bboxes[lidar_dir]

    # Fetch LIDAR data for desired folder
    lidar = fetch_lidar_from_folder(lidar_dir)

    return lidar, bbox


def get_elevation(lat: float, lon: float) -> float:
    """For a given latitude & longitude, retrieve an estimated elevation at
    this point. This is dependent on the relevant LIDAR data being available
    in the data directory.

    Args:
        lat (float): Latitude
        lon (float): Longitude

    Returns:
        float: The elevation in metres of the provided coordinates
    """
    # Convert to grid reference
    easting, northing = WGS84toOSGB36(lat, lon)

    # Fetch LIDAR data for grid reference
    lidar, bbox = get_lidar_for_bng_reference(easting, northing)

    # Get top-left boundary
    boundary_e = bbox[0]
    boundary_n = bbox[3]

    # Calculate required offset from this point
    offset_e = easting - boundary_e
    offset_s = boundary_n - northing

    # Return elevation at this offset
    elevation = lidar[int(offset_s), int(offset_e)]  # type: ignore

    return elevation


# Ensure bbox data is pushed to redis, glob may not work as
# expected when running within subprocesses
get_all_bboxes()
