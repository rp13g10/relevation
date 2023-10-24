import hashlib
import os
import random
import re
from functools import lru_cache
from glob import glob
from math import ceil
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


def add_file_ids(lidar_df: pd.DataFrame, lidar_dir: str) -> pd.DataFrame:
    lidar_df.loc[:, "file_id"] = generate_file_id(lidar_dir)

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
            };
        """
    )
    session.execute(query)


def create_lidar_table(session: Session):
    # TODO: Read into optimal partitioning strategies for single record
    #       retrieval & optimise accordingly

    query = dedent(
        """
            CREATE TABLE IF NOT EXISTS
                relevation.lidar (
                    easting_ptn int,
                    northing_ptn int,
                    easting int,
                    northing int,
                    elevation float,
                    file_id text,
                    PRIMARY KEY (
                        (easting_ptn, northing_ptn), easting, northing
                    )
                )
            WITH
                CLUSTERING ORDER BY (easting ASC);
        """
    )
    session.execute(query)


def create_dir_table(session: Session):
    query = dedent(
        """
            CREATE TABLE IF NOT EXISTS
                relevation.ingested_files (
                    file_name text PRIMARY KEY,
                    file_id text
                );
        """
    )
    session.execute(query)


def check_if_file_already_loaded(lidar_dir: str, session: Session) -> bool:
    query = dedent(
        f"""
            SELECT
                *
            FROM
                relevation.ingested_files
            WHERE
                file_name = '{lidar_dir}'
            LIMIT 1;
        """
    )

    rows = session.execute(query)
    if rows:
        return True
    return False


@lru_cache(16)
def generate_file_id(lidar_dir: str) -> str:
    file_id = hashlib.sha256(lidar_dir.encode("utf8"))
    file_id = file_id.hexdigest()
    return file_id


def mark_file_as_loaded(lidar_dir: str, session: Session):
    file_id = generate_file_id(lidar_dir)

    query = dedent(
        f"""
            INSERT INTO
                relevation.ingested_files (
                    file_name,
                    file_id
                )
            VALUES (
                {lidar_dir},
                {file_id}
            );
        """
    )

    session.execute(query)


def get_all_loaded_files(session: Session) -> Set[str]:
    query = dedent(
        """
            SELECT
                file_name
            FROM
                relevation.ingested_files;
        """
    )

    rows = session.execute(query)

    files = {row.file_name for row in rows}
    files = {generate_file_id(file) for file in files}

    return files


def delete_records_from_partially_loaded_files(session: Session):
    loaded_files = get_all_loaded_files(session)
    loaded_files = ",".join({f"'{file}'" for file in loaded_files})

    if loaded_files:
        query = dedent(
            f"""
                DELETE FROM
                    relevation.lidar
                WHERE
                    file_id NOT IN ({loaded_files});
            """
        )
    else:
        query = dedent(
            """
                TRUNCATE TABLE
                    relevation.lidar;
            """
        )

    session.execute(query)


def _generate_row_insert(row: pd.Series) -> str:
    insert_query = dedent(
        """
            INSERT INTO
                relevation.lidar (
                    easting_ptn,
                    northing_ptn,
                    easting,
                    northing,
                    elevation,
                    file_id
                )
            VALUES (
                {easting_ptn:d},
                {northing_ptn:d},
                {easting:d},
                {northing:d},
                {elevation},
                {file_id}
            );
        """
    )

    row_query = insert_query.format(
        easting_ptn=row.easting_ptn,
        northing_ptn=row.northing_ptn,
        easting=row.easting,
        northing=row.northing,
        elevation=row.elevation,
        file_id=f"'{row.file_id}'",
    )

    row_query = re.sub(r"\s+", r" ", row_query)

    return row_query


def store_lidar_df_batched(
    lidar_df: pd.DataFrame, lidar_dir: str, session: Session
):
    """Will store dataframe contents to ScyllaDB"""

    # Base code: https://stackoverflow.com/questions/49108809/how-to-insert-pandas-dataframe-into-cassandra
    # Split by partition rather than random allocation

    if check_if_file_already_loaded(lidar_dir, session):
        return None

    insert_query = dedent(
        """
        BEGIN BATCH
        {insert_statements}
        APPLY BATCH;
        """
    )

    n_chunks = ceil(len(lidar_df.index) / 1000)
    for chunk in tqdm(
        random.shuffle(np.array_split(lidar_df, n_chunks)),
        desc="Uploading Records",
        total=n_chunks,
    ):
        insert_statements = [
            _generate_row_insert(row)
            for row in chunk.itertuples()  # type: ignore
        ]
        insert_statements = "\n".join(insert_statements)

        chunk_query = insert_query.format(insert_statements=insert_statements)

        session.execute(chunk_query)

    mark_file_as_loaded(lidar_dir, session)


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
    # TODO: Give ScyllaDB slightly more resource to play with

    available_folders = get_available_folders()
    lidar_dir = list(available_folders)[0]
    lidar = load_lidar_from_folder(lidar_dir)
    bbox = load_bbox_from_folder(lidar_dir)
    lidar_df = explode_lidar(lidar, bbox)
    lidar_df = add_partition_keys(lidar_df)
    lidar_df = add_file_ids(lidar_df, lidar_dir)

    create_app_keyspace(sc_sess)
    create_lidar_table(sc_sess)
    create_dir_table(sc_sess)
    store_lidar_df_batched(lidar_df, lidar_dir, sc_sess)
