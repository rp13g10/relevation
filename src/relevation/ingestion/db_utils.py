"""These handle the configuration of a fresh ScyllaDB cluster, and the writing
of data to it."""

import os
import random
import re
from math import ceil
from textwrap import dedent
from typing import Set, Union

import numpy as np
import pandas as pd

# pylint: disable=no-name-in-module
from cassandra.cluster import (
    Session,
)
from tqdm import tqdm

from relevation.ingestion.file_utils import (
    generate_file_id,
    parse_lidar_folder,
)


def create_app_keyspace(session: Session):
    """Generate a `relevation` keyspace in the ScyllaDB instance, unless it
    already exists

    Args:
        session (Session): An active ScyllaDB session
    """

    query = dedent(
        """
            CREATE KEYSPACE IF NOT EXISTS
                relevation
            WITH REPLICATION = {
                'class': 'NetworkTopologyStrategy',
                'datacenter1': 1
            };
        """
    ).strip()
    session.execute(query)


def drop_app_keyspace(session: Session):
    """Drop the `relevation` keyspace from the ScyllaDB instance, unless it
    is already missing

    Args:
        session (Session): An active ScyllaDB session
    """
    query = "DROP KEYSPACE IF EXISTS relevation;"
    session.execute(query)


def create_lidar_table(session: Session):
    """If not already present, create a new table which will contain elevation
    data parsed from the provided LIDAR files.
    The created table will be `relevation.lidar`

    Args:
        session (Session): An active ScyllaDB session
    """

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
    ).strip()
    session.execute(query)


def create_dir_table(session: Session):
    """If not already present, create a new table which will contain a
    directory of all files which have already been loaded into
    `relevation.lidar`. The created table will be `relevation.ingested_files`

    Args:
        session (Session): An active ScyllaDB session
    """
    query = dedent(
        """
            CREATE TABLE IF NOT EXISTS
                relevation.ingested_files (
                    file_id text PRIMARY KEY,
                );
        """
    ).strip()
    session.execute(query)


def check_if_file_already_loaded(file_id: str, session: Session) -> bool:
    """For a given file ID, query the `relevation.ingested_files` table and
    deterine whether it has already been ingested or not.

    Args:
        file_id (str): The unique identifier for a LIDAR file
        session (Session): An active ScyllaDB session

    Returns:
        bool: True if the file has already been loaded, False if it hasn't
    """
    query = dedent(
        f"""
            SELECT
                *
            FROM
                relevation.ingested_files
            WHERE
                file_id = '{file_id}'
            LIMIT 1;
        """
    ).strip()

    rows = session.execute(query)
    if rows:
        return True
    return False


def mark_file_as_loaded(file_id: str, session: Session):
    """For a given file_id, create a new record in relevation.ingested_files
    to indicate that it has been loaded in its entirety.

    Args:
        file_id (str): The unique identifier for a LIDAR file
        session (Session): An active ScyllaDB session
    """
    query = dedent(
        f"""
            INSERT INTO
                relevation.ingested_files (
                    file_id
                )
            VALUES (
                '{file_id}'
            );
        """
    ).strip()

    session.execute(query)


def get_all_loaded_files(session: Session) -> Set[str]:
    """Get all of the files which have been marked as fully loaded in
    the relevation.ingested_files table

    Args:
        session (Session): An active ScyllaDB session

    Returns:
        Set[str]: All of the files which have been completely loaded into the
          database
    """
    query = dedent(
        """
            SELECT
                file_id
            FROM
                relevation.ingested_files;
        """
    ).strip()

    rows = session.execute(query)

    files = {row.file_id for row in rows}

    return files


def delete_records_from_partially_loaded_files(session: Session):
    """Delete all of the records in relevation.lidar where the file_id does
    not also appear in relevation.ingested_files. These records will be
    associated with a file where the ingestion process was interrupted before
    completion

    Args:
        session (Session): An active ScyllaDB session
    """

    # TODO: Either fix this, or remove it

    loaded_files = get_all_loaded_files(session)
    loaded_files = ",".join({f"'{file}'" for file in loaded_files})

    if loaded_files:
        query = dedent(
            f"""
                DELETE FROM 
                    elevation.lidar
                WHERE 
                    ile_id NOT IN ({loaded_files});
            """
        ).strip()
    else:
        query = dedent(
            """
                TRUNCATE TABLE
                    relevation.lidar;
            """
        ).strip()

    session.execute(query)


def _generate_row_insert(row: pd.Series) -> str:
    """Generate a CQL statement which will insert data from the provided pandas
    series into ScyllaDB when executed

    Args:
        row (pd.Series): A single row from the pandas dataframe containing data
          for the current file

    Returns:
        str: A prepared INSERT query which can be executed against the database
    """
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
                '{file_id}'
            );
        """
    ).strip()

    row_query = insert_query.format(
        easting_ptn=row.easting_ptn,
        northing_ptn=row.northing_ptn,
        easting=row.easting,
        northing=row.northing,
        elevation=row.elevation,
        file_id=row.file_id,
    )

    row_query = re.sub(r"\s+", r" ", row_query)

    return row_query


def store_lidar_df(
    lidar_df: pd.DataFrame, file_id: str, session: Session
) -> bool:
    """Will store dataframe contents to ScyllaDB

    Returns:
        bool: True if new data was written, False if the selected dataset was
          already present in the database"""

    if check_if_file_already_loaded(file_id, session):
        return False

    insert_query = dedent(
        """
        BEGIN BATCH
        {insert_statements}
        APPLY BATCH;
        """
    ).strip()

    # 500 gives chunk size just below warning limit for scylladb
    n_chunks = ceil(len(lidar_df.index) / 500)
    lidar_chunks = np.array_split(lidar_df, n_chunks)
    # Random order to prevent hammering a single partition
    random.shuffle(lidar_chunks)
    for chunk in tqdm(lidar_chunks, desc="Uploading Records", total=n_chunks):
        insert_statements = [
            _generate_row_insert(row)
            for row in chunk.itertuples()  # type: ignore
        ]
        insert_statements = "\n".join(insert_statements)

        chunk_query = insert_query.format(insert_statements=insert_statements)

        session.execute_async(chunk_query)

    mark_file_as_loaded(file_id, session)

    return True


def fetch_elevation(
    easting: int, northing: int, session: Session
) -> Union[float, None]:
    """For a given easting & northing, retrieve the elevation at that point.
    This function requires that the provided session connects to a database
    containing LIDAR data. This ingestion can be triggered by `main.py` in the
    root of the `relevation` module.

    Args:
        easting (int): The easting for the point of interest
        northing (int): The northing for the point of interest
        session (Session): An active ScyllaDB session

    Returns:
        Union[float, None]: The elevation for the specified point if it could
          be retrieved from the database. None if it could not.
    """
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
    ).strip()

    rows = session.execute(query)
    if rows:
        elevation = list(rows)[0].elevation
        return elevation


def initialize_db(session: Session):
    """Ensure that the target database is set up and ready to accept data
    from LIDAR files.

    Args:
        session (Session): An active ScyllaDB session
    """
    create_app_keyspace(session)
    create_lidar_table(session)
    create_dir_table(session)


def upload_csv(sc_container, lidar_id: str):
    """For a given lidar_id, upload the (parsed) csv file into cassandra

    Args:
        lidar_id (str): The unique identifier for a LIDAR file
    """
    copy_stmt = dedent(
        f"""
        COPY
            relevation.lidar (
                easting_ptn,
                northing_ptn,
                easting,
                northing,
                elevation,
                file_id
            )
        FROM
            'source_data/{lidar_id}.csv'
    """
    ).strip()

    copy_stmt = re.sub(r"\s+", " ", copy_stmt)

    copy_stmt = f'cqlsh --execute="{copy_stmt}"'

    sc_container.exec_run(copy_stmt)  # type: ignore


def write_df_to_csv(lidar_df: pd.DataFrame, lidar_id: str, data_dir: str):
    """Write the contents of a dataframe containing lidar data to the
    specified location, setting the file name to <lidar_id>.csv

    Args:
        lidar_df (pd.DataFrame): A dataframe containing lidar data
        lidar_id (str): The unique identifier for the lidar file which was
          used to create lidar_df
        data_dir (str): The location which the parsed dataframe should be
          exported to
    """
    csv_loc = os.path.join(data_dir, f"csv/{lidar_id}.csv")

    col_list = [
        "easting_ptn",
        "northing_ptn",
        "easting",
        "northing",
        "elevation",
        "file_id",
    ]
    lidar_df[col_list].to_csv(csv_loc, index=False, header=False)


def load_single_file(lidar_dir: str, sc_sess: Session, sc_container) -> bool:
    """For a single lidar file, parse the data and export it to csv. The
    parsed csv will then be loaded into cassandra.

    Args:
        lidar_dir (str): The absolute location of a single lidar folder, as
          provided by DEFRA
        sc_sess (Session): An active scylla/cassandra session

    Returns:
        bool: True if data was loaded successfully, False if it wasn't
    """
    lidar_id = generate_file_id(lidar_dir)

    loaded = check_if_file_already_loaded(lidar_id, sc_sess)
    if not loaded:
        lidar_df = parse_lidar_folder(lidar_dir)

        data_dir = os.path.abspath(os.path.join(lidar_dir, "../.."))

        write_df_to_csv(lidar_df, lidar_id, data_dir)

        upload_csv(sc_container, lidar_id)
        mark_file_as_loaded(lidar_id, sc_sess)

    return not loaded
