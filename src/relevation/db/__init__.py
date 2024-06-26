import math
import sys
from textwrap import dedent
from typing import List, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
from geopy.distance import distance
from cassandra.cluster import (
    Cluster,
)  # pylint: disable=no-name-in-module

from relevation.utils import get_coordinates, get_partitions

# TODO: Refactor this, set up as function argument. Need some thought as to
#       how this will work with pySpark
if "pytest" in sys.modules:
    sc_db, sc_sess = MagicMock(), MagicMock()
else:
    sc_db = Cluster(port=9042)
    sc_sess = sc_db.connect()
ELEVATION_CHANGE_INTERVAL = 10


def get_elevation(lat: float, lon: float) -> Union[float, None]:
    """For a given latitude & longitude, fetch the elevation from the
    active ScyllaDB cluster and return it. If the provided coordinates require
    data not in the database, None will be returned instead."""

    # Convert to grid reference
    easting, northing = get_coordinates(lat, lon)

    # Get partitions within lookup table
    easting_ptn, northing_ptn = get_partitions(easting, northing)

    # Prepare query
    # fmt: off
    query = dedent(
    f"""
    SELECT
        elevation
    FROM
        relevation.lidar
    WHERE
        easting_ptn={easting_ptn}
        AND northing_ptn={northing_ptn}
        AND easting={easting}
        AND northing={northing}
    LIMIT 1;
    """
    ).strip()
    # fmt: on

    rows = sc_sess.execute(query)

    # return rows

    for row in rows:
        return row.elevation
    return None


def _get_elevation_checkpoints(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> Tuple[List[float], List[float], float]:
    """Given a start & end point, return a list of equally spaced latitudes
    and longitudes between them. These can then be used to estimate the
    elevation change between start & end by calculating loss/gain between
    each checkpoint.

    Args:
        start_lat (float): Latitude for the start point
        start_lon (float): Longitude for the start point
        end_lat (float): Latitude for the end point
        end_lon (float): Longitude for the end point

    Returns:
        Tuple[List[float], List[float], distance]: A list of latitudes and
            a corresponding list of longitudes which represent points on an
            edge of the graph. The distance between the start & end point
            in kilometers
    """
    # Calculate distance from A to B
    dist_change = distance((start_lat, start_lon), (end_lat, end_lon))

    # Calculate number of checks required to get elevation every N metres
    dist_change_m = dist_change.meters
    no_checks = math.ceil(dist_change_m / ELEVATION_CHANGE_INTERVAL)
    no_checks = max(2, no_checks)

    # Generate latitudes & longitudes for each checkpoint
    lat_checkpoints = list(
        np.linspace(start_lat, end_lat, num=no_checks, endpoint=True)
    )
    lon_checkpoints = list(
        np.linspace(start_lon, end_lon, num=no_checks, endpoint=True)
    )

    dist_change = dist_change.kilometers

    return lat_checkpoints, lon_checkpoints, dist_change


def _calculate_elevation_change_for_checkpoints(
    lat_checkpoints: List[float],
    lon_checkpoints: List[float],
) -> Union[Tuple[float, float], Tuple[None, None]]:
    """For the provided latitude/longitude coordinates, estimate the total
    elevation gain/loss along the entire route.

    Args:
        lat_checkpoints (List[float]): A list of equally spaced latitudes
            which represent points on an edge of the graph
        lon_checkpoints (List[float]): A list of equally spaced longitudes
            which represent points on an edge of the graph

    Returns:
        Tuple[float, float]: Elevation gain in metres, elevation loss in
            metres
    """
    # Calculate elevation at each checkpoint
    elevations = []
    for lat, lon in zip(lat_checkpoints, lon_checkpoints):
        elevation = get_elevation(lat, lon)

        # If any data is missing, we cannot calculate the overall change
        if elevation is None:
            return None, None

        elevations.append(elevation)

    # Work out the sum of elevation gains/losses between checkpoints
    last_elevation = None
    elevation_gain = 0.0
    elevation_loss = 0.0
    for elevation in elevations:
        if not last_elevation:
            last_elevation = elevation
            continue
        if elevation > last_elevation:
            elevation_gain += elevation - last_elevation
        elif elevation < last_elevation:
            elevation_loss += last_elevation - elevation
        last_elevation = elevation

    return elevation_gain, elevation_loss


def get_distance_and_elevation_change(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> Union[Tuple[float, float, float], Tuple[None, None, None]]:
    """For a given start & end node, estimate the change in elevation when
    traversing the edge between them. Elevation is sampled at a resolution
    of 10 metres.

    Args:
        start_lat (int): The latitude for the start point
        start_lon (int): The longitude for the start point
        end_lat (int): The latitude for the end point
        end_lon (int): The longitude for the end point

    Returns:
        Tuple[float, float, float]: The distance change in kilometres,
          elevation gain in metres and elevation loss in metres
    """

    (
        lat_checkpoints,
        lon_checkpoints,
        dist_change,
    ) = _get_elevation_checkpoints(start_lat, start_lon, end_lat, end_lon)

    (
        elevation_gain,
        elevation_loss,
    ) = _calculate_elevation_change_for_checkpoints(
        lat_checkpoints, lon_checkpoints
    )

    if elevation_gain is None or elevation_loss is None:
        return None, None, None

    return dist_change, elevation_gain, elevation_loss
