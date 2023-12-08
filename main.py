"""Sets up the ScyllaDB cluster and ingests all available data"""
# pylint: disable=no-name-in-module
import warnings
from cassandra.cluster import Cluster

from relevation.ingestion.file_utils import iter_dfs
from relevation.ingestion.db_utils import (
    initialize_db,
    load_df,
    mark_file_as_loaded
)

warnings.filterwarnings(action="ignore", category=FutureWarning)

# NOTE: stack defined by backend/compose.yaml must be up before running this

sc_db = Cluster(port=9042)
sc_sess = sc_db.connect()

data_dir = '/home/ross/repos/relevation/data'

initialize_db(sc_sess)
for lidar_df, lidar_id in iter_dfs(data_dir):
    load_df(lidar_df, lidar_id, data_dir)
    mark_file_as_loaded(lidar_id, sc_sess)