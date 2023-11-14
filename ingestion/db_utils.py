"""These handle the configuration of a fresh ScyllaDB cluster, and the writing
of data to it."""
import random
import re
from math import ceil
from textwrap import dedent
from typing import Set, Union

import numpy as np
import pandas as pd
from cassandra.cluster import Session  # pylint: disable=no-name-in-module
from tqdm import tqdm


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
    )
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
    )
    session.execute(query)


def create_dir_table(session: Session):
    """If not already present, create a new table which will contain a
    directory of all files which have already been loaded into
    `relevation.lidar`. The created table will be `relevation.ingested_files`

    Args:
        session (Session): An active ScyllaDB session
    """
    # TODO: Check replication strategies available for small datasets
    query = dedent(
        """
            CREATE TABLE IF NOT EXISTS
                relevation.ingested_files (
                    file_id text PRIMARY KEY,
                );
        """
    )
    session.execute(query)


def check_if_file_already_loaded(file_id: str, session: Session) -> bool:
    """For a given file ID, query the `relevation.ingested_files` table and
    deterine whether it has already been ingested or not.

    Args:
        file_id (str): The unique identifier for a LIDAR file
        session (Session): An active ScyllaDB session

    Returns:
        bool: _description_
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
    )

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
    )

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
    )

    rows = session.execute(query)

    files = {row.file_id for row in rows}

    return files


def delete_records_from_partially_loaded_files(session: Session):
    """Delete all of the records in relevation.lidar where the file_id does
    not also appear in relevation.ingested_files. These records will be
    associated with a file where the ingestion process was interrupted before
    completion

    Args:
        session (Session): _description_
    """
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
    )

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


def store_lidar_df(lidar_df: pd.DataFrame, file_id: str, session: Session):
    """Will store dataframe contents to ScyllaDB"""

    if check_if_file_already_loaded(file_id, session):
        return None

    insert_query = dedent(
        """
        BEGIN BATCH
        {insert_statements}
        APPLY BATCH;
        """
    )

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


def initialize_db(session: Session):
    create_app_keyspace(session)
    create_lidar_table(session)
    create_dir_table(session)
