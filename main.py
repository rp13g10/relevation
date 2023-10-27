from cassandra.cluster import Cluster

from relevation.ingestion.file_utils import iter_dfs
from relevation.ingestion.db_utils import initialize_db, store_lidar_df

# NOTE: Docker commands for basic ScyllaDB instance
# docker run --name relevation_db --volume relevation_data:/var/lib/scylla -p 9042:9042 -d scylladb/scylla


# docker network create -d bridge rrp_net
# docker run --network rrp_net -d scylladb/scylla --smp 1
# docker run --network rrp_net -i python /bin/bash
# Within python container, can connect using IPAddress for scylladb found when
#    inspecting the scylladb container

# This script expects a local ScyllaDB instance to be available
sc_db = Cluster(port=9042)
sc_sess = sc_db.connect()

initialize_db(sc_sess)
for lidar_df, lidar_id in iter_dfs():
    store_lidar_df(lidar_df, lidar_id, sc_sess)
