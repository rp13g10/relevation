import os
from glob import glob
from textwrap import dedent
from typing import Set, Union

import numpy as np
import pandas as pd
import rasterio as rio
import shapefile as shp
from cassandra.cluster import Cluster, Session
from tqdm import tqdm

cur_dir = os.path.abspath(os.path.dirname(__file__))

# NOTE: Docker commands for basic ScyllaDB instance
# docker network create -d bridge rrp_net
# docker run --network rrp_net -d scylladb/scylla --smp 1
# docker run --network rrp_net -i python /bin/bash
# Within python container, can connect using IPAddress for scylladb found when
#    inspecting the scylladb container

# NOTE: This script will need to run in a Docker container with data directory
#       mounted as a volume

sc_db = Cluster(port=9042)
sc_sess = sc_db.connect()


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
    lidar_df.loc[:, "easting_ptn"] = lidar_df["easting"] // 100
    lidar_df.loc[:, "northing_ptn"] = lidar_df["northing"] // 100
    return lidar_df


def store_source_file(lidar_dir: str):
    """Will store parsed file name to ScyllaDB"""


def create_app_keyspace(session: Session):
    query = dedent(
        """
            CREATE KEYSPACE IF NOT EXISTS
                relevation
            WITH REPLICATION = {
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }
            ;
        """
    )
    session.execute(query)


def create_lidar_table(session: Session):
    query = dedent(
        """
            CREATE TABLE IF NOT EXISTS
                relevation.lidar (
                    easting_ptn int,
                    northing_ptn int,
                    easting int,
                    northing int,
                    elevation float,
                    PRIMARY KEY (
                        (easting_ptn, northing_ptn), easting, northing
                    )
                )
            WITH
                CLUSTERING ORDER BY (easting ASC)
        """
    )
    session.execute(query)


def create_dir_table(session: Session):
    query = dedent(
        """
            CREATE TABLE IF NOT EXISTS
                relevation.ingested_files (
                    file_name text PRIMARY KEY
                );
        """
    )
    session.execute(query)


# TODO: Add file ID as unique int, generate this before ingesting
#       Tag each LIDAR entry with corresponding file ID
#       Once complete, add file and ID to ingested_files table
#       Build out function to remove records from partially ingested files


def store_lidar_df(lidar_df: pd.DataFrame, session: Session):
    """Will store dataframe contents to ScyllaDB"""

    # Base code: https://stackoverflow.com/questions/49108809/how-to-insert-pandas-dataframe-into-cassandra
    # Split by partition rather than random allocation

    query = dedent(
        """
            INSERT INTO
                relevation.lidar (
                    easting_ptn,
                    northing_ptn,
                    easting,
                    northing,
                    elevation
                )
            VALUES (
                {easting_ptn:d},
                {northing_ptn:d},
                {easting:d},
                {northing:d},
                {elevation}
            );
        """
    )

    for row in tqdm(
        lidar_df.itertuples(),
        desc="Uploading records",
        total=len(lidar_df.index),
    ):
        row_query = query.format(
            easting_ptn=row.easting_ptn,
            northing_ptn=row.northing_ptn,
            easting=row.easting,
            northing=row.northing,
            elevation=row.elevation,
        )
        session.execute(row_query)


def fetch_elevation(
    easting: int, northing: int, session: Session
) -> Union[float, None]:
    easting_ptn = easting // 100
    northing_ptn = northing // 100

    query = dedent(
        f"""
            SELECT
                elevation
            FROM
                relevation.lidar
            WHERE
                easting_ptn = {easting_ptn:d}
                AND northing_ptn = {northing_ptn:d}
                AND easting = {easting:d}
                AND northing = {northing:d}
            LIMIT 1;
        """
    )

    rows = session.execute(query)
    if rows:
        elevation = list(rows)[0].elevation
        return elevation


if __name__ == "__main__":
    available_folders = get_available_folders()
    lidar_dir = list(available_folders)[0]
    lidar = load_lidar_from_folder(lidar_dir)
    bbox = load_bbox_from_folder(lidar_dir)
    lidar_df = explode_lidar(lidar, bbox)
    lidar_df = add_partition_keys(lidar_df)

    create_app_keyspace(sc_sess)
    create_lidar_table(sc_sess)
    store_lidar_df(lidar_df, sc_sess)
