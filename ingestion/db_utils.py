import random
import re
from math import ceil
from textwrap import dedent
from typing import Set, Union

import numpy as np
import pandas as pd
from cassandra.cluster import Session
from tqdm import tqdm


def create_app_keyspace(session: Session):
    # TODO: Determine optimal replication strategy for small cluster
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


def check_if_file_already_loaded(lidar_id: str, session: Session) -> bool:
    query = dedent(
        f"""
            SELECT
                *
            FROM
                relevation.ingested_files
            WHERE
                file_id = '{lidar_id}'
            LIMIT 1;
        """
    )

    rows = session.execute(query)
    if rows:
        return True
    return False


def mark_file_as_loaded(lidar_id: str, session: Session):
    query = dedent(
        f"""
            INSERT INTO
                relevation.ingested_files (
                    file_id
                )
            VALUES (
                {lidar_id}
            );
        """
    )

    session.execute(query)


def get_all_loaded_files(session: Session) -> Set[str]:
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


def store_lidar_df(lidar_df: pd.DataFrame, lidar_id: str, session: Session):
    """Will store dataframe contents to ScyllaDB"""

    if check_if_file_already_loaded(lidar_id, session):
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
        # Random order to prevent hammering a single partition
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

    mark_file_as_loaded(lidar_id, session)


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
