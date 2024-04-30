"""Unit test for the database utility functions"""

from collections import namedtuple
from textwrap import dedent
from unittest.mock import patch, MagicMock
import pytest
from geopy.distance import Distance
import relevation.db as rdb

mock_executor = MagicMock()
rdb.sc_sess.execute = mock_executor

MockRow = namedtuple("MockRow", ["elevation"])


class TestGetElevation:
    """Check that the correct query is being generated and executed"""

    @patch("relevation.db.get_coordinates")
    def test_data_returned(self, mock_get_coordinates: MagicMock):
        """Ensure the correct query is being generated"""

        # Arrange
        rdb.sc_sess.reset_mock()  # type: ignore
        mock_executor.reset_mock()

        def _side_effect(x, y):
            return round(x * 100), round(y * 100)

        mock_get_coordinates.side_effect = _side_effect

        mock_row = MockRow(10.0)
        mock_executor.return_value = [mock_row]

        test_lat = 123.456789
        test_lon = 987.654321

        # fmt: off
        target_query = dedent(
        """
        SELECT
            elevation
        FROM
            relevation.lidar
        WHERE
            easting_ptn=12
            AND northing_ptn=99
            AND easting=12346
            AND northing=98765
        LIMIT 1;
        """
        ).strip()
        # fmt: on

        target_value = 10.0

        # Act
        result = rdb.get_elevation(test_lat, test_lon)

        # Assert
        mock_executor.assert_called_once_with(target_query)
        assert result == target_value

    @patch("relevation.db.get_coordinates")
    def test_data_not_returned(self, mock_get_coordinates: MagicMock):
        """Ensure empty query results return None"""

        # Arrange
        rdb.sc_sess.reset_mock()  # type: ignore
        mock_executor.reset_mock()

        def _side_effect(x, y):
            return round(x * 100), round(y * 100)

        mock_get_coordinates.side_effect = _side_effect

        mock_executor.return_value = []

        test_lat = 123.456789
        test_lon = 987.654321

        target = None

        # Act
        result = rdb.get_elevation(test_lat, test_lon)

        # Assert
        assert result is target


@patch("relevation.db.distance")
def test_get_elevation_checkpoints(mock_distance: MagicMock):
    """Make sure the generated checkpoints are as-expected"""
    # Arrange
    mock_distance.return_value = Distance(meters=91.0)

    test_start_lat = 50.0
    test_start_lon = 0.0
    test_end_lat = 60.0
    test_end_lon = 10.0

    target_lat_checkpoints = [
        50.0,
        51.1111111,
        52.2222222,
        53.3333333,
        54.4444444,
        55.5555555,
        56.6666666,
        57.7777777,
        58.8888888,
        60.0,
    ]
    target_lon_checkpoints = [
        0.0,
        1.1111111,
        2.2222222,
        3.3333333,
        4.4444444,
        5.5555555,
        6.6666666,
        7.7777777,
        8.8888888,
        10.0,
    ]
    target_dist_change = 0.091

    # Act
    result_lat_checkpoints, result_lon_checkpoints, result_dist_change = (
        rdb._get_elevation_checkpoints(
            test_start_lat, test_start_lon, test_end_lat, test_end_lon
        )
    )

    # Assert
    assert result_lat_checkpoints == pytest.approx(target_lat_checkpoints)
    assert result_lon_checkpoints == pytest.approx(target_lon_checkpoints)
    assert result_dist_change == target_dist_change


class TestCalculateElevationChangeForCheckpoints:
    """Make sure elevation changes are calculated correctly"""

    @patch("relevation.db.get_elevation")
    def test_all_data_present(
        self,
        mock_get_elevation: MagicMock,
    ):
        """Standard case, elevation data available for all checkpoints"""
        # Arrange
        mock_get_elevation.side_effect = [
            35.0,
            40.0,
            38.0,
            43.0,
            40.0,
            35.0,
            35.0,
            25.0,
            28.0,
            30.0,
        ]

        test_lat_checkpoints = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        test_lon_checkpoints = test_lat_checkpoints[:]

        target_elevation_gain = 5.0 + 5.0 + 3.0 + 2.0
        target_elevation_loss = 2.0 + 3.0 + 10.0 + 5.0

        # Act
        result_elevation_gain, result_elevation_loss = (
            rdb._calculate_elevation_change_for_checkpoints(
                test_lat_checkpoints, test_lon_checkpoints
            )
        )

        # Assert
        assert result_elevation_gain == target_elevation_gain
        assert result_elevation_loss == target_elevation_loss

    @patch("relevation.db.get_elevation")
    def test_some_data_out_of_bounds(
        self,
        mock_get_elevation: MagicMock,
    ):
        """Error case, elevation data missing for some checkpoints"""
        # Arrange
        mock_get_elevation.side_effect = [
            35.0,
            40.0,
            38.0,
            43.0,
            None,
            35.0,
            30.0,
            25.0,
            28.0,
            30.0,
        ]

        test_lat_checkpoints = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        test_lon_checkpoints = test_lat_checkpoints[:]

        target_elevation_gain = None
        target_elevation_loss = None

        # Act
        result_elevation_gain, result_elevation_loss = (
            rdb._calculate_elevation_change_for_checkpoints(
                test_lat_checkpoints, test_lon_checkpoints
            )
        )

        # Assert
        assert result_elevation_gain is target_elevation_gain
        assert result_elevation_loss is target_elevation_loss


class TestGetDistanceAndElevationChange:
    """Ensure the correct values are being returned when calculating
    distance and elevation changes"""

    @patch("relevation.db._calculate_elevation_change_for_checkpoints")
    @patch("relevation.db._get_elevation_checkpoints")
    def test_all_data_present(
        self,
        mock_get_elevation_checkpoints: MagicMock,
        mock_calculate_elevation_change_for_checkpoints: MagicMock,
    ):
        """Standard case, data was returned for all data points"""
        # ----- Arrange -----
        mock_get_elevation_checkpoints.return_value = (
            "lat_checkpoints",
            "lon_checkpoints",
            "dist_change",
        )

        mock_calculate_elevation_change_for_checkpoints.return_value = (
            "elevation_gain",
            "elevation_loss",
        )

        # Dummy values
        test_start_lat = 0.0
        test_start_lon = 0.0
        test_end_lat = 0.0
        test_end_lon = 0.0

        target_dist_change = "dist_change"
        target_elevation_gain = "elevation_gain"
        target_elevation_loss = "elevation_loss"

        # ----- Act -----

        result_dist_change, result_elevation_gain, result_elevation_loss = (
            rdb.get_distance_and_elevation_change(
                test_start_lat, test_start_lon, test_end_lat, test_end_lon
            )
        )

        # ----- Assert -----

        assert result_dist_change == target_dist_change
        assert result_elevation_gain == target_elevation_gain
        assert result_elevation_loss == target_elevation_loss

    @patch("relevation.db._calculate_elevation_change_for_checkpoints")
    @patch("relevation.db._get_elevation_checkpoints")
    def test_some_data_out_of_bounds(
        self,
        mock_get_elevation_checkpoints: MagicMock,
        mock_calculate_elevation_change_for_checkpoints: MagicMock,
    ):
        """Standard case, data was returned for all data points"""
        # ----- Arrange -----
        mock_get_elevation_checkpoints.return_value = (
            "lat_checkpoints",
            "lon_checkpoints",
            "dist_change",
        )

        mock_calculate_elevation_change_for_checkpoints.return_value = (
            None,
            None,
        )

        # Dummy values
        test_start_lat = 0.0
        test_start_lon = 0.0
        test_end_lat = 0.0
        test_end_lon = 0.0

        target_dist_change = None
        target_elevation_gain = None
        target_elevation_loss = None

        # ----- Act -----

        result_dist_change, result_elevation_gain, result_elevation_loss = (
            rdb.get_distance_and_elevation_change(
                test_start_lat, test_start_lon, test_end_lat, test_end_lon
            )
        )

        # ----- Assert -----

        assert result_dist_change is target_dist_change
        assert result_elevation_gain is target_elevation_gain
        assert result_elevation_loss is target_elevation_loss
