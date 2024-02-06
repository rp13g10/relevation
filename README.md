# relevation

This package is provided to facilitate the use of the elevation data provided by DEFRA. Its primary benefits are the automatic mapping of the data from the OS grid reference system to latitude/longitude, and the scalability of the backend database.

This package is designed for applications which involve fetching the elevation for a large number of coordinates across a wide area. It will be overkill for anything which works within a few square kms, or fetches the elevation for one point at a time. Loading the data into cassandra introduces an overhead which can only be justified when the script retrieving the data is parallelised.

## Installation

In order to use the python code in this package, it must be installed to your current python environment. Once the contents of the repo have been cloned, this can be done by entering `pip install .` in a terminal in the root of the repo.

In order for the code to function, a containerised cassandra database will need to be visible on port 9042 of the local machine. The automatic setup of this cluster will be managed via a config file in a future build. A sample docker compose.yaml file is provided in the 'backend' folder for reference.

## Usage

Make sure your database is up and running before attempting to run any of the code here.

### Downloading Data

In main.py, you will need to specify a data folder. This will need to contain two subfolders, 'lidar' and 'csv'.

Elevation data can be downloaded from 'environment.data.gov.uk/survey'. Select the areas you need and select 'Composite DTM 1m' when prompted to select a format. Extract all of your files into the 'lidar' folder.

SCREENSHOT HERE

### Ingestion

### Ongoing Usage