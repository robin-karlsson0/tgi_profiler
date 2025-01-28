from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
from transformers import AutoTokenizer

from tgi_profiler.boundary_detection import BoundaryPair
from tgi_profiler.profiler import (InterpolationPoint, ProfilingResult,
                                   TGIMemoryProfiler)


@pytest.fixture
def profiler(basic_profiler_config):
    """Create a TGIMemoryProfiler instance for testing."""
    return TGIMemoryProfiler(basic_profiler_config)


LOREM_IPSUM = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'  # noqa


def test_count_tokens(profiler):

    # Test basic token counting
    num_tokens = profiler._count_tokens(LOREM_IPSUM)

    model_id = profiler.config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_tokens_gt = len(tokenizer.tokenize(LOREM_IPSUM))

    assert num_tokens == num_tokens_gt


def test_generate_exact_token_input(profiler):

    target_length = 666

    input_txt = profiler._generate_exact_token_input(target_length)

    model_id = profiler.config.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    num_tokens = len(tokenizer.tokenize(input_txt))

    assert num_tokens == target_length


###################################################
#  Test cases for _interpolate_boundary_points()
###################################################


def test_basic_midpoint_interpolation(profiler):
    """Test basic midpoint calculation between success/failure points."""
    pairs = [
        BoundaryPair(
            success_point=ProfilingResult(1000, 2000, True),
            failure_point=ProfilingResult(2000, 2000, False),
            confidence=0.9,  # High confidence for clean boundary
            region_coverage=0.8  # Good coverage for basic case
        )
    ]

    points = profiler._interpolate_boundary_points(pairs)

    assert len(points) == 1
    assert points[0].input_length == 1500
    assert points[0].output_length == 2000


def test_below_min_refinement_dist(profiler):
    """Test that pairs with differences below min_refinement_dist are skipped."""
    pairs = [
        BoundaryPair(
            success_point=ProfilingResult(1000, 2000, True),
            failure_point=ProfilingResult(1040, 2000, False),
            confidence=0.95,  # High confidence for close points
            region_coverage=0.7  # Lower coverage for dense region
        )
    ]

    points = profiler._interpolate_boundary_points(pairs)

    assert len(points) == 0


def test_diagonal_boundary_pair(profiler):
    """Test interpolation when both dimensions differ significantly."""
    pairs = [
        BoundaryPair(
            success_point=ProfilingResult(1000, 2000, True),
            failure_point=ProfilingResult(2000, 3000, False),
            confidence=0.7,  # Lower confidence for diagonal transition
            region_coverage=0.9  # High coverage for sparse diagonal region
        )
    ]

    points = profiler._interpolate_boundary_points(pairs)

    assert len(points) == 1
    assert points[0].input_length == 1500
    assert points[0].output_length == 2500


def test_multiple_boundary_pairs(profiler):
    """Test handling multiple boundary pairs simultaneously."""
    pairs = [
        BoundaryPair(
            success_point=ProfilingResult(1000, 2000, True),
            failure_point=ProfilingResult(2000, 2000, False),
            confidence=0.85,  # High confidence for horizontal boundary
            region_coverage=0.75  # Moderate coverage
        ),
        BoundaryPair(
            success_point=ProfilingResult(1500, 2000, True),
            failure_point=ProfilingResult(1500, 3000, False),
            confidence=0.8,  # High confidence for vertical boundary
            region_coverage=0.8  # Moderate coverage
        )
    ]

    points = profiler._interpolate_boundary_points(pairs)

    assert len(points) == 2
    assert points[0].input_length == 1500
    assert points[0].output_length == 2000
    assert points[1].input_length == 1500
    assert points[1].output_length == 2500


def test_empty_pairs_list(profiler):
    """Test handling of empty boundary pairs list."""
    points = profiler._interpolate_boundary_points([])
    assert len(points) == 0


def test_mixed_refinement_distances(profiler):
    """Test pairs where one dimension is below min_refinement_dist and other is above."""
    pairs = [
        BoundaryPair(
            success_point=ProfilingResult(1000, 2000, True),
            failure_point=ProfilingResult(1040, 2500, False),
            confidence=0.75,  # Moderate confidence for mixed distances
            region_coverage=0.85  # Good coverage for asymmetric region
        )
    ]

    points = profiler._interpolate_boundary_points(pairs)

    assert len(points) == 1
    assert points[0].input_length == 1020
    assert points[0].output_length == 2250


def test_different_min_refinement_dist(basic_profiler_config):
    """Test behavior with different min_refinement_dist configuration."""
    basic_profiler_config.min_refinement_dist = 100  # Increase minimum distance
    profiler = TGIMemoryProfiler(basic_profiler_config)

    pairs = [
        BoundaryPair(
            success_point=ProfilingResult(1000, 2000, True),
            failure_point=ProfilingResult(1080, 2000, False),
            confidence=0.9,  # High confidence for close points
            region_coverage=0.6  # Lower coverage for dense region
        )
    ]

    points = profiler._interpolate_boundary_points(pairs)

    assert len(points) == 0


def test_respect_config_bounds(profiler):
    """Test that interpolation points stay within configured bounds."""
    profiler.config.max_input_length = 1500  # Set artificial limit
    pairs = [
        BoundaryPair(success_point=ProfilingResult(1000, 2000, True),
                     failure_point=ProfilingResult(2000, 2000, False),
                     confidence=0.9,
                     region_coverage=0.8)
    ]
    points = profiler._interpolate_boundary_points(pairs)
    assert len(points) == 0  # Point would be at 1500, exceeds max


def test_duplicate_points(profiler):
    """Test handling of duplicate interpolation points."""
    pairs = [
        BoundaryPair(success_point=ProfilingResult(1000, 2000, True),
                     failure_point=ProfilingResult(2000, 2000, False),
                     confidence=0.9,
                     region_coverage=0.8),
        BoundaryPair(success_point=ProfilingResult(1000, 2000, True),
                     failure_point=ProfilingResult(2000, 2000, False),
                     confidence=0.8,
                     region_coverage=0.7)
    ]
    points = profiler._interpolate_boundary_points(pairs)
    assert len(points) == 1  # Should deduplicate identical points


#######################################
#  Test cases for _test_new_points()
#######################################


@pytest.fixture
def mock_test_point(profiler):
    """Create a mock test_point method that returns predictable results."""
    with patch.object(profiler, 'test_point') as mock:

        def side_effect(input_length, output_length):
            # Simulate success for smaller configurations, failure for larger ones
            success = input_length <= 1500 and output_length <= 2500
            return ProfilingResult(input_length=input_length,
                                   output_length=output_length,
                                   success=success)

        mock.side_effect = side_effect
        yield mock


def test_test_new_points_basic(profiler, mock_test_point):
    """Test basic functionality with new points."""
    # Setup initial results
    profiler.results = [
        ProfilingResult(1000, 2000, True),
        ProfilingResult(2000, 2000, False)
    ]

    # Create new points to test
    new_points = [
        InterpolationPoint(input_length=1500, output_length=2000),  # New point
        InterpolationPoint(input_length=1800, output_length=2000)  # New point
    ]

    profiler._test_new_points(new_points)

    # Verify both points were tested
    assert mock_test_point.call_count == 2
    assert len(profiler.results) == 4  # Original 2 + 2 new points


def test_test_new_points_empty(profiler, mock_test_point):
    """Test behavior with empty list of new points."""
    profiler.results = [ProfilingResult(1000, 2000, True)]
    profiler._test_new_points([])

    # Verify no points were tested
    mock_test_point.assert_not_called()
    assert len(profiler.results) == 1  # No change in results


def test_test_new_points_duplicates(profiler, mock_test_point):
    """Test handling of points that have already been tested."""
    # Setup initial results with one point
    existing_point = ProfilingResult(1000, 2000, True)
    profiler.results = [existing_point]

    # Try to test the same point again
    new_points = [
        InterpolationPoint(input_length=existing_point.input_length,
                           output_length=existing_point.output_length)
    ]

    profiler._test_new_points(new_points)

    # Verify no new tests were performed
    mock_test_point.assert_not_called()
    assert len(profiler.results) == 1  # No change in results


def test_test_new_points_error_handling(profiler):
    """Test error handling when testing points fails."""
    # Mock test_point to raise an exception
    with patch.object(profiler, 'test_point') as mock:
        mock.side_effect = Exception("Test failure")

        # Setup test points
        new_points = [
            InterpolationPoint(input_length=1500, output_length=2000),
            InterpolationPoint(input_length=1800, output_length=2000)
        ]

        # Should not raise exception, should continue with remaining points
        profiler._test_new_points(new_points)

        # Verify both points were attempted
        assert mock.call_count == 2


def test_test_new_points_mixed_duplicates(profiler, mock_test_point):
    """Test handling mix of new and previously tested points."""
    # Setup initial results
    profiler.results = [
        ProfilingResult(1000, 2000, True),
        ProfilingResult(2000, 2000, False)
    ]

    # Mix of new and existing points
    new_points = [
        InterpolationPoint(input_length=1000, output_length=2000),  # Existing
        InterpolationPoint(input_length=1500, output_length=2000),  # New
        InterpolationPoint(input_length=2000, output_length=2000)  # Existing
    ]

    profiler._test_new_points(new_points)

    # Verify only the new point was tested
    assert mock_test_point.call_count == 1
    assert len(profiler.results) == 3  # Original 2 + 1 new point


def test_test_new_points_result_recording(profiler, mock_test_point):
    """Test that results are correctly recorded and ordered."""
    # Setup initial results
    profiler.results = [ProfilingResult(1000, 2000, True)]

    # Create new points
    new_points = [
        InterpolationPoint(input_length=1500, output_length=2000),
        InterpolationPoint(input_length=1800, output_length=2000)
    ]

    profiler._test_new_points(new_points)

    # Verify results were added in order
    assert len(profiler.results) == 3
    assert profiler.results[0].input_length == 1000  # Original point
    assert profiler.results[1].input_length == 1500  # First new point
    assert profiler.results[2].input_length == 1800  # Second new point


def test_test_new_points_progress_bar(profiler, mock_test_point):
    """Test progress bar functionality."""
    profiler._current_phase = "Test Phase"
    new_points = [
        InterpolationPoint(input_length=1500, output_length=2000),
        InterpolationPoint(input_length=1800, output_length=2000)
    ]

    with patch('tgi_profiler.profiler.tqdm') as mock_tqdm:
        mock_progress = Mock()
        mock_tqdm.return_value.__enter__.return_value = mock_progress

        profiler._test_new_points(new_points)

        # Verify progress bar was created with correct parameters
        mock_tqdm.assert_called_once()
        assert mock_progress.update.call_count == 2  # Called for each point
