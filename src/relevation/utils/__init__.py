"""Utility functions which are used across the different modules within this
package"""

from typing import Tuple

from bng_latlon import WGS84toOSGB36


def get_coordinates(lat: float, lon: float) -> Tuple[int, int]:
    """For a given latitude and longitude, fetch the corresponding easting and
    northing. Coordinates are returned as integers, as this is how the LIDAR
    data is stored.

    Args:
        lat (float): The latitude for the target point
        lon (float): The longitude for the target point

    Returns:
        Tuple[int, int]: The easting and northing for the target point
    """

    easting, northing = WGS84toOSGB36(lat, lon)
    easting = round(easting)
    northing = round(northing)

    return easting, northing


def get_partitions(easting: int, northing: int) -> Tuple[int, int]:
    """For a given easting and northing, fetch the corresponding easting and
    northing partitions.

    Args:
        easting (int): The easting for the target point
        northing (int): The northing for the target point

    Returns:
        Tuple[int, int]: The easting partition and the northing partition
    """

    easting_ptn = round(easting / 100)
    northing_ptn = round(northing / 100)

    return easting_ptn, northing_ptn
