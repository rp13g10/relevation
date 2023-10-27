import os
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from relevation.ingestion import file_utils as fu


class TestGetAvailableFolders:
    """Check behaviour is as-expected when searching for files"""

    @patch("relevation.ingestion.file_utils.glob")
    def test_files_found(self, mock_glob):
        """Check expected output format when files are found"""
        # Arrange
        mock_glob.return_value = ["item_one"]
        target = {"item_one"}

        # Act
        result = fu.get_available_folders()

        # Assert
        assert result == target

    @patch("relevation.ingestion.file_utils.glob")
    def test_files_not_found(self, mock_glob):
        """Check exception is thrown when they aren't"""
        # Arrange
        mock_glob.return_value = {}

        # Act, Assert
        with pytest.raises(FileNotFoundError):
            _ = fu.get_available_folders()


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


@patch("relevation.ingestion.file_utils.shp.Reader")
@patch("relevation.ingestion.file_utils.glob")
def test_load_bbox_from_folder(
    mock_glob: MagicMock, mock_shp_reader: MagicMock
):
    """Check that data is passed through the function call as expected"""
    # Arrange
    mock_glob.return_value = ["file_one"]

    # Sets sf while inside the 'with X as sf' block
    mock_sf_inside = MagicMock()
    mock_sf_inside.bbox = np.zeros(1)

    # Sets sf while the 'with X as sf' statement is evaluated
    mock_sf_outside = MagicMock()
    mock_sf_outside.__enter__ = MagicMock(return_value=mock_sf_inside)

    mock_shp_reader.return_value = mock_sf_outside

    test_lidar_dir = "/some/path"

    target_glob_call = os.path.join("/some/path", "index/*.shp")
    target_shp_call = "file_one"
    target_output = np.zeros(1, dtype=int)

    # Act
    result = fu.load_bbox_from_folder(test_lidar_dir)

    # Assert
    mock_glob.assert_called_once_with(target_glob_call)
    mock_shp_reader.assert_called_once_with(target_shp_call)
    assert result == target_output


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
            "elevation": [24995000, 24990000, 24985000, 24980000, 24975000],
            "easting": [100000, 100000, 100000, 100000, 100000],
            "northing": [200000, 200001, 200002, 200003, 200004],
        }
    )
    target_first_five_eastings = pd.DataFrame.from_dict(
        {
            "elevation": [24995000, 24995001, 24995002, 24995003, 24995004],
            "easting": [100000, 100001, 100002, 100003, 100004],
            "northing": [200000, 200000, 200000, 200000, 200000],
        }
    )
    target_last_five_northings = pd.DataFrame.from_dict(
        {
            "elevation": [24999, 19999, 14999, 9999, 4999],
            "easting": [104999, 104999, 104999, 104999, 104999],
            "northing": [204995, 204996, 204997, 204998, 204999],
        }
    )
    target_last_five_eastings = pd.DataFrame.from_dict(
        {
            "elevation": [4995, 4996, 4997, 4998, 4999],
            "easting": [104995, 104996, 104997, 104998, 104999],
            "northing": [204999, 204999, 204999, 204999, 204999],
        }
    )
    # Act
    result = fu.explode_lidar(test_lidar, test_bbox)
    result_first_five_eastings = result.sort_values(by=["northing", "easting"])
    result_first_five_northings = result.sort_values(
        by=["easting", "northing"]
    )

    # TODO: Finish writing this test, then have fun getting it to pass!

    # Assert
