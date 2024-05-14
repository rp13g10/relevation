"""Connects to the Cassandra cluster and ingests all available data"""

# pylint: disable=no-name-in-module
import os
import warnings

import docker
from cassandra.cluster import Cluster
from tqdm import tqdm

from relevation.ingestion.file_utils import get_available_folders
from relevation.ingestion.db_utils import (
    initialize_db,
    generate_file_id,
    check_if_file_already_loaded,
    parse_lidar_folder,
    write_df_to_csv,
    upload_csv,
    mark_file_as_loaded,
)

warnings.filterwarnings(action="ignore", category=FutureWarning)

# NOTE: stack defined by backend/compose.yaml must be up before running this


DATA_DIR = "/home/ross/repos/relevation/data"
CLUSTER_URL = "unix:///home/ross/.docker/desktop/docker.sock"

client = docker.DockerClient(base_url=CLUSTER_URL)
container = client.containers.get("cassandra_1")

sc_db = Cluster(port=9042)
sc_sess = sc_db.connect()

initialize_db(sc_sess)

all_lidar_dirs = get_available_folders(DATA_DIR)

for lidar_dir in tqdm(all_lidar_dirs):
    lidar_id = generate_file_id(lidar_dir)

    loaded = check_if_file_already_loaded(lidar_id, sc_sess)
    if not loaded:
        lidar_df = parse_lidar_folder(lidar_dir)

        data_dir = os.path.abspath(os.path.join(lidar_dir, "../.."))

        write_df_to_csv(lidar_df, lidar_id, data_dir)

        upload_csv(container, lidar_id)
        mark_file_as_loaded(lidar_id, sc_sess)
