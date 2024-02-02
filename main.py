"""Connects to the Cassandra cluster and ingests all available data"""

# pylint: disable=no-name-in-module
import warnings

from cassandra.cluster import Cluster
from tqdm import tqdm

from relevation.ingestion.file_utils import get_available_folders
from relevation.ingestion.db_utils import (
    initialize_db,
    load_single_file,
)

warnings.filterwarnings(action="ignore", category=FutureWarning)

# NOTE: stack defined by backend/compose.yaml must be up before running this


DATA_DIR = "/home/ross/repos/relevation/data"


sc_db = Cluster(port=9042)
sc_sess = sc_db.connect()

initialize_db(sc_sess)

all_lidar_dirs = get_available_folders(DATA_DIR)

count = 0
for lidar_dir in tqdm(all_lidar_dirs):

    loaded = load_single_file(lidar_dir, sc_sess)

    if loaded:
        # Temporarily stop every 5th file to prevent tying up the computer
        # for too long
        count += 1
    if count == 5:
        break
