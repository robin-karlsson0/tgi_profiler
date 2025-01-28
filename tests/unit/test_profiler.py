from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest
from transformers import AutoTokenizer

from tgi_profiler.boundary_detection import BoundaryPair
from tgi_profiler.profiler import (ProfilerConfig, ProfilingResult,
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
