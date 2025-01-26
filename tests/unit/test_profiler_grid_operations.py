"""Unit tests for the TGI Memory Profiler implementation."""

from pathlib import Path

import numpy as np
import pytest

from tgi_profiler.profiler import (ProfilerConfig, ProfilingResult,
                                   TGIMemoryProfiler)


def create_mock_results(input_lengths: list, output_lengths: list,
                        success_matrix: np.ndarray) -> list:
    """Create mock profiling results based on a success matrix.
    
    Args:
        input_lengths: List of input sequence lengths
        output_lengths: List of output sequence lengths
        success_matrix: 2D boolean array where True indicates success
        
    Returns:
        List of ProfilingResult objects
    """
    results = []
    for i, input_len in enumerate(input_lengths):
        for j, output_len in enumerate(output_lengths):
            results.append(
                ProfilingResult(
                    input_length=input_len,
                    output_length=output_len,
                    success=bool(success_matrix[i, j]),
                    error_type=None if success_matrix[i, j] else "OOM"))
    return results


def test_refine_grid_basic_boundary(basic_profiler_config):
    """Test that _refine_grid correctly identifies and refines around a simple
    boundary.
    
    This test creates a simple diagonal boundary between success/failure
    regions and verifies that the refinement adds appropriate points around
    this boundary.
    """
    profiler = TGIMemoryProfiler(basic_profiler_config)

    # Create initial grid points
    initial_input_points = np.array([1000, 2000, 3000, 4000])
    initial_output_points = np.array([500, 1000, 1500, 2000])
    profiler.input_points = initial_input_points
    profiler.output_points = initial_output_points

    # Create a diagonal boundary where points above/right fail
    # [[ 1  1  0  0 ]
    #  [ 1  1  0  0 ]
    #  [ 0  0  0  0 ]
    #  [ 0  0  0  0 ]]
    success_matrix = np.zeros((4, 4))
    success_matrix[0:2, 0:2] = 1

    # Create corresponding results
    profiler.results = create_mock_results(initial_input_points,
                                           initial_output_points,
                                           success_matrix)

    # Run grid refinement
    profiler._refine_grid()

    # Verify refinement added appropriate points
    # 1. Should add points between boundary successes and failures
    # 2. Should maintain sorted order
    # 3. Should not duplicate points

    # Check new input points were added around boundary
    assert len(profiler.input_points) > len(initial_input_points)
    assert np.all(np.diff(profiler.input_points) > 0)  # Verify sorted
    assert 2500 in profiler.input_points  # Midpoint between 2000 and 3000

    # Check new output points were added around boundary
    assert len(profiler.output_points) > len(initial_output_points)
    assert np.all(np.diff(profiler.output_points) > 0)  # Verify sorted
    assert 1250 in profiler.output_points  # Midpoint between 1000 and 1500

    # Verify boundary points are included
    assert 2000 in profiler.input_points  # Boundary point
    assert 1000 in profiler.output_points  # Boundary point

    # Verify original points are preserved
    for point in initial_input_points:
        assert point in profiler.input_points
    for point in initial_output_points:
        assert point in profiler.output_points


def test_refine_grid_all_success(basic_profiler_config, create_profiler):
    """Test grid refinement behavior when all points are successful."""
    input_points = [1000, 2000, 3000]
    output_points = [500, 1000, 1500]
    profiler = create_profiler(basic_profiler_config, input_points,
                               output_points)

    # All points successful
    success_matrix = np.ones((3, 3))
    profiler.results = create_mock_results(input_points, output_points,
                                           success_matrix)

    # Store original points
    original_input_points = profiler.input_points.copy()
    original_output_points = profiler.output_points.copy()

    profiler._refine_grid()

    # Should not add new points when no boundary exists
    np.testing.assert_array_equal(profiler.input_points, original_input_points)
    np.testing.assert_array_equal(profiler.output_points,
                                  original_output_points)


def test_refine_grid_all_fail(basic_profiler_config, create_profiler):
    """Test grid refinement behavior when all points fail."""
    input_points = [1000, 2000, 3000]
    output_points = [500, 1000, 1500]
    profiler = create_profiler(basic_profiler_config, input_points,
                               output_points)

    # All points fail
    success_matrix = np.zeros((3, 3))
    profiler.results = create_mock_results(input_points, output_points,
                                           success_matrix)

    # Store original points
    original_input_points = profiler.input_points.copy()
    original_output_points = profiler.output_points.copy()

    profiler._refine_grid()

    # Should not add new points when no boundary exists
    np.testing.assert_array_equal(profiler.input_points, original_input_points)
    np.testing.assert_array_equal(profiler.output_points,
                                  original_output_points)


def test_refine_grid_checkerboard(basic_profiler_config, create_profiler):
    """Test grid refinement with alternating success/failure pattern."""
    input_points = [1000, 2000, 3000, 4000]
    output_points = [500, 1000, 1500, 2000]
    profiler = create_profiler(basic_profiler_config, input_points,
                               output_points)

    # Checkerboard pattern
    success_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0],
                               [0, 1, 0, 1]])
    profiler.results = create_mock_results(input_points, output_points,
                                           success_matrix)

    profiler._refine_grid()

    # Should add many refinement points due to many boundaries
    assert len(profiler.input_points) >= len(input_points) * 1.5
    assert len(profiler.output_points) >= len(output_points) * 1.5

    # Verify points are properly ordered
    assert np.all(np.diff(profiler.input_points) > 0)
    assert np.all(np.diff(profiler.output_points) > 0)


def test_refine_grid_single_boundary_point(basic_profiler_config,
                                           create_profiler):
    """Test refinement around a single boundary point."""
    input_points = [1000, 2000, 3000]
    output_points = [500, 1000, 1500]
    profiler = create_profiler(basic_profiler_config, input_points,
                               output_points)

    # Single boundary point at (2000, 1000)
    success_matrix = np.zeros((3, 3))
    success_matrix[1, 1] = 1
    profiler.results = create_mock_results(input_points, output_points,
                                           success_matrix)

    profiler._refine_grid()

    # Should add points around the single success point
    expected_input_midpoints = {1500, 2500}  # Midpoints around 2000
    expected_output_midpoints = {750, 1250}  # Midpoints around 1000

    actual_input_midpoints = set(profiler.input_points) - set(input_points)
    actual_output_midpoints = set(profiler.output_points) - set(output_points)

    assert expected_input_midpoints.issubset(actual_input_midpoints)
    assert expected_output_midpoints.issubset(actual_output_midpoints)


def test_refine_grid_validation(basic_profiler_config, create_profiler):
    """Test grid refinement input validation and error handling."""
    input_points = [1000, 2000, 3000]
    output_points = [500, 1000, 1500]
    profiler = create_profiler(basic_profiler_config, input_points,
                               output_points)

    # Test with missing results
    profiler.results = []
    with pytest.raises(ValueError, match="No results available"):
        profiler._refine_grid()

    # Test with mismatched results
    profiler.results = [
        ProfilingResult(
            input_length=5000,  # Point not in grid
            output_length=1000,
            success=True)
    ]
    with pytest.raises(ValueError, match="Result contains points not in grid"):
        profiler._refine_grid()


@pytest.mark.parametrize(
    "success_pattern,expected_new_points",
    [
        # Pattern: horizontal boundary
        (np.array([[1, 1, 1], [0, 0, 0]]), 1),
        # Pattern: vertical boundary
        (np.array([[1, 0], [1, 0], [1, 0]]), 1),
        # Pattern: diagonal boundary
        (np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]]), 2),
    ])
def test_refine_grid_boundary_patterns(basic_profiler_config, create_profiler,
                                       success_pattern, expected_new_points):
    """Test grid refinement with different boundary patterns."""
    rows, cols = success_pattern.shape
    input_points = np.arange(rows) * 1000 + 1000
    output_points = np.arange(cols) * 500 + 500

    profiler = create_profiler(basic_profiler_config, input_points,
                               output_points)
    profiler.results = create_mock_results(input_points, output_points,
                                           success_pattern)

    original_points = len(input_points) + len(output_points)
    profiler._refine_grid()
    new_points = len(profiler.input_points) + len(
        profiler.output_points) - original_points

    assert new_points >= expected_new_points
