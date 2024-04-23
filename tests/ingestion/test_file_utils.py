"""Unit tests for file loading utilities"""

import os
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from relevation.ingestion import file_utils as fu


class TestGetAvailableFolders:
    """Check behaviour is as-expected when searching for files"""

    @patch("relevation.ingestion.file_utils.glob")
    def test_files_found(self, mock_glob: MagicMock):
        """Check expected output format when files are found"""
        # Arrange
        test_dir = "test/dir"
        mock_glob.return_value = ["item_one"]
        target_out = {"item_one"}
        target_call = "test/dir/lidar/lidar_composite_dtm-*"

        # Act
        result = fu.get_available_folders(test_dir)

        # Assert
        assert result == target_out
        mock_glob.assert_called_once_with(target_call)

    @patch("relevation.ingestion.file_utils.glob")
    def test_files_not_found(self, mock_glob):
        """Check exception is thrown when they aren't"""
        # Arrange
        test_dir = "dummy"
        mock_glob.return_value = {}

        # Act, Assert
        with pytest.raises(FileNotFoundError):
            _ = fu.get_available_folders(test_dir)


@patch("relevation.ingestion.file_utils.rio.open")
@patch("relevation.ingestion.file_utils.glob")
def test_load_lidar_from_folder(
    mock_glob: MagicMock, mock_rio_open: MagicMock
):
    """Check that data is passed through the function call as expected"""
    # Arrange
    mock_glob.return_value = ["file_one"]

    # Sets tif while inside the 'with X as tif' block
    mock_tif_inside = MagicMock()
    mock_tif_inside.read = MagicMock(return_value=[np.zeros(1)])

    # Sets tif while the 'with X as tif' statement is evaluated
    mock_tif_outside = MagicMock()
    mock_tif_outside.__enter__ = MagicMock(return_value=mock_tif_inside)

    mock_rio_open.return_value = mock_tif_outside

    test_lidar_dir = "/some/path"

    target_glob_call = os.path.join("/some/path", "*.tif")
    target_rio_call = "file_one"
    target_output = np.zeros(1)

    # Act
    result = fu.load_lidar_from_folder(test_lidar_dir)

    # Assert
    mock_glob.assert_called_once_with(target_glob_call)
    mock_rio_open.assert_called_once_with(target_rio_call)
    assert result == target_output


@patch("relevation.ingestion.file_utils.open")
@patch("relevation.ingestion.file_utils.glob")
def test_load_bbox_from_folder(mock_glob: MagicMock, mock_open: MagicMock):
    """Check that data is passed through the function call as expected"""
    # Arrange
    mock_glob.return_value = ["file_one"]

    # Mock out behaviour for data read
    mock_lines = [
        "1.0000000000\n",
        "0.0000000000\n",
        "0.0000000000\n",
        "-1.0000000000\n",
        "445000.5000000000\n",
        "119999.5000000000\n",
    ]
    mock_readlines = MagicMock(return_value=mock_lines, id="readlines")

    mock_fobj = MagicMock(id="fobj")
    mock_fobj.readlines = mock_readlines

    mock_handle = MagicMock(id="handle")
    mock_handle.__enter__ = MagicMock(return_value=mock_fobj, id="enter")

    mock_open.return_value = mock_handle

    # Set dummy file location
    test_lidar_dir = "/some/path"

    # Set expected outputs
    target_glob_call = os.path.join("/some/path", "*.tfw")
    target_open_args = ("file_one", "r")
    target_open_kwargs = {"encoding": "utf8"}

    target_output = np.array([445000, 115000, 450000, 120000])

    # Act
    result = fu.load_bbox_from_folder(test_lidar_dir)

    # Assert
    mock_glob.assert_called_once_with(target_glob_call)
    mock_open.assert_called_once_with(*target_open_args, **target_open_kwargs)
    assert (result == target_output).all()


def test_generate_file_id():
    """Check that file IDs are being generated properly"""
    # Arrange
    test_lidar_dir = "/some/folder/LIDAR-DTM-1m-2022-SU20ne"
    target = "SU20ne"

    # Act
    result = fu.generate_file_id(test_lidar_dir)

    # Assert
    assert result == target


def test_explode_lidar():
    """Check that LIDAR data is correctly being transformed into a pandas
    DataFrame"""

    # Arrange
    test_bbox = np.array([100000, 200000, 105000, 205000])

    # [0, 1, 2, ..., 24999999]
    test_lidar = np.array(range(0, 5000**2))
    # [[0, 1, ..., 4999], [5000, ..., 9999], ..., [24995000, ..., 24999999]]
    test_lidar = test_lidar.reshape(5000, 5000)

    target_first_five_northings = pd.DataFrame.from_dict(
        {
            "easting": [100000, 100000, 100000, 100000, 100000],
            "northing": [200000, 200001, 200002, 200003, 200004],
            "elevation": [24995000, 24990000, 24985000, 24980000, 24975000],
        }
    )
    target_first_five_eastings = pd.DataFrame.from_dict(
        {
            "easting": [100000, 100001, 100002, 100003, 100004],
            "northing": [200000, 200000, 200000, 200000, 200000],
            "elevation": [24995000, 24995001, 24995002, 24995003, 24995004],
        }
    )
    target_last_five_northings = pd.DataFrame.from_dict(
        {
            "easting": [104999, 104999, 104999, 104999, 104999],
            "northing": [204999, 204998, 204997, 204996, 204995],
            "elevation": [4999, 9999, 14999, 19999, 24999],
        }
    )
    target_last_five_eastings = pd.DataFrame.from_dict(
        {
            "easting": [104999, 104998, 104997, 104996, 104995],
            "northing": [204999, 204999, 204999, 204999, 204999],
            "elevation": [4999, 4998, 4997, 4996, 4995],
        }
    )
    # Act
    result = fu.explode_lidar(test_lidar, test_bbox)
    result_first_five_eastings = (
        result.sort_values(by=["northing", "easting"])
        .head()
        .reset_index(drop=True)
    )
    result_first_five_northings = (
        result.sort_values(by=["easting", "northing"])
        .head()
        .reset_index(drop=True)
    )
    result_last_five_eastings = (
        result.sort_values(by=["northing", "easting"], ascending=False)
        .head()
        .reset_index(drop=True)
    )
    result_last_five_northings = (
        result.sort_values(by=["easting", "northing"], ascending=False)
        .head()
        .reset_index(drop=True)
    )

    # Assert
    assert_frame_equal(
        result_first_five_eastings,
        target_first_five_eastings,
        check_dtype=False,
    )
    assert_frame_equal(
        result_first_five_northings,
        target_first_five_northings,
        check_dtype=False,
    )
    assert_frame_equal(
        result_last_five_eastings, target_last_five_eastings, check_dtype=False
    )
    assert_frame_equal(
        result_last_five_northings,
        target_last_five_northings,
        check_dtype=False,
    )


def test_add_partition_keys():
    """Check that partition keys are set correctly"""

    # Arrange
    test_lidar_df = pd.DataFrame.from_dict(
        {
            "easting": [100000, 100025, 100050, 100075, 100100],
            "northing": [9950, 9975, 10000, 10025, 10050],
        }
    )

    target = pd.DataFrame.from_dict(
        {
            "easting": [100000, 100025, 100050, 100075, 100100],
            "northing": [9950, 9975, 10000, 10025, 10050],
            # NOTE: Python rounds evens down from 0.5 and odds up from 0.5
            "easting_ptn": [1000, 1000, 1000, 1001, 1001],
            "northing_ptn": [100, 100, 100, 100, 100],
        }
    )

    # Act
    result = fu.add_partition_keys(test_lidar_df)

    # Assert
    assert_frame_equal(
        result,
        target,
        check_dtype=False,
    )


def test_add_file_ids():
    """Check that file IDs are added correctly"""

    # Arrange
    test_lidar_df = pd.DataFrame.from_dict(
        {
            "other": [1, 1, 1, 1, 1],
        }
    )

    test_lidar_dir = "/some/path/LIDAR-DTM-1m-2022-SU20ne"

    target = pd.DataFrame.from_dict(
        {"other": [1, 1, 1, 1, 1], "file_id": ["SU20ne"] * 5}
    )

    # Act
    result = fu.add_file_ids(test_lidar_df, test_lidar_dir)

    # Assert
    assert_frame_equal(
        result,
        target,
        check_dtype=False,
    )


@pytest.mark.skip()
def test_iter_dfs():
    """This test is not present as it would only serve to show that python
    is able to iterate through a list correctly"""
