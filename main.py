"""Sets up the ScyllaDB cluster and ingests all available data"""
# pylint: disable=no-name-in-module
import warnings
from cassandra.cluster import Cluster

from relevation.ingestion.file_utils import iter_dfs
from relevation.ingestion.db_utils import initialize_db, store_lidar_df

warnings.filterwarnings(action="ignore", category=FutureWarning)

# NOTE: stack defined by backend/compose.yaml must be up before running this

# This script expects a local ScyllaDB instance to be available
sc_db = Cluster(port=9042)
sc_sess = sc_db.connect()

initialize_db(sc_sess)
for lidar_df, lidar_id in iter_dfs():
    store_lidar_df(lidar_df, lidar_id, sc_sess)
