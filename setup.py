"""Setup configuration for the relevation package"""
from setuptools import setup

setup(
    name="relevation",
    version="0.1.0",
    where="src",
    include=["relevation"],
    install_requires=[
        "scylla-driver>=3.26.0",
        "pandas>=2.1.0",
        "rasterio>=1.3.0",
        "pyshp>=2.3.0",
        "tqdm>=4.66.0",
        "bng-latlon>=1.1",
    ],
)
