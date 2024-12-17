import pytest
from unittest.mock import patch, MagicMock
import firedrake  
import numpy as np

from parcels.field import FiredrakeField

@pytest.fixture
def test_instance():
    mock_grid = MagicMock()
    mock_grid.mesh = MagicMock()
    variable = "test_variable"
    time = np.array([0.0, 1.0, 2.0, 3.0])
    file_list = ["file0.h5", "file1.h5", "file2.h5", "file3.h5"]
    return FiredrakeField(variable, file_list, time, "uv_2d", grid=mock_grid)


@patch("firedrake.CheckpointFile")
def test_computeTimeChunk_initial_load(mock_checkpoint_file, test_instance):
    mock_checkpoint = mock_checkpoint_file.return_value.__enter__.return_value
    mock_checkpoint.load_function.return_value = "test_function"

    test_instance.computeTimeChunk(1.0)

    assert len(test_instance._cached_data) == 3
    assert test_instance._cached_index == [0, 1, 2]
    mock_checkpoint_file.assert_any_call("file0.h5", "r")
    mock_checkpoint_file.assert_any_call("file1.h5", "r")
    mock_checkpoint_file.assert_any_call("file2.h5", "r")


@patch("firedrake.CheckpointFile")
def test_computeTimeChunk_cached_hit(mock_checkpoint_file, test_instance):
    test_instance._cached_index = [0, 1, 2]
    test_instance._cached_data = ["a","b","c"]

    test_instance.computeTimeChunk(1.0)
    mock_checkpoint_file.assert_not_called()


@patch("firedrake.CheckpointFile")
def test_computeTimeChunk_advance_time(mock_checkpoint_file, test_instance):
    mock_checkpoint = mock_checkpoint_file.return_value.__enter__.return_value
    mock_checkpoint.load_function.return_value = "test_function"
    test_instance._cached_index = [0, 1, 2]
    test_instance._cached_data = ["a","b","c"]

    test_instance.computeTimeChunk(2.0)

    assert len(test_instance._cached_data) == 3
    assert test_instance._cached_index == [1, 2, 3]
    mock_checkpoint_file.assert_any_call("file3.h5", "r")

@patch("firedrake.CheckpointFile")
def test_computeTimeChunk_end_of_time(mock_checkpoint_file, test_instance):
    mock_checkpoint = mock_checkpoint_file.return_value.__enter__.return_value
    mock_checkpoint.load_function.return_value = "test_function"
    test_instance._cached_index = [1, 2, 3]
    test_instance._cached_data = ["a","b","c"]
    test_instance.computeTimeChunk(3.0)

    assert len(test_instance._cached_data) == 3
    assert test_instance._cached_index == [2, 3, None]
    mock_checkpoint_file.assert_not_called()
