"""Unit tests for core utility functions"""

from relevation.utils import get_coordinates, get_partitions

from unittest.mock import patch, MagicMock


@patch("relevation.utils.WGS84toOSGB36")
def test_get_coordinates(mock_wgs84toosgb36: MagicMock):
    """Check that the expected function output is returned and rounded
    correctly"""
    # Arrange
    mock_wgs84toosgb36.side_effect = lambda x, y: (x, y)

    test_lat = 50.1
    test_lon = 50.9

    target_easting = 50
    target_northing = 51

    # Act
    result_easting, result_northing = get_coordinates(test_lat, test_lon)

    # Assert
    assert result_easting == target_easting
    assert result_northing == target_northing


def test_get_partitions():
    """Check that the partitions are being generated correctly"""
    # Arrange
    test_easting = 12345
    test_northing = 56789

    target_easting_ptn = 12
    target_northing_ptn = 57

    # Act
    result_easting_ptn, result_northing_ptn = get_partitions(
        test_easting, test_northing
    )

    # Assert
    assert result_easting_ptn == target_easting_ptn
    assert result_northing_ptn == target_northing_ptn
