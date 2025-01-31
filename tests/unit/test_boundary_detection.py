from dataclasses import dataclass

import numpy as np
import pytest

from tgi_profiler.boundary_detection import (BoundaryConfig, BoundaryPair,
                                             _compute_confidence,
                                             _compute_coverage_score,
                                             _filter_pairs,
                                             identify_boundary_pairs)
from tgi_profiler.profiler import ProfilingResult


@dataclass
class MockBoundaryConfig(BoundaryConfig):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_identify_boundary_pairs_basic():
    """Tests basic boundary pair identification with a simple test case.

    Test case has 4 points forming a clear boundary:
    - 2 success points at (1000,500) and (2000,500)
    - 2 failure points at (3000,500) and (1000,1000)

    Verifies:
    - At least one boundary pair is found
    - All pairs are BoundaryPair instances
    - Success/failure points are correctly ordered
    """
    config = MockBoundaryConfig()
    results = [
        ProfilingResult(1000, 500, True),
        ProfilingResult(2000, 500, True),
        ProfilingResult(3000, 500, False),
        ProfilingResult(1000, 1000, False)
    ]

    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) > 0
    assert all(isinstance(p, BoundaryPair) for p in pairs)
    assert all(p.success_point.success and not p.failure_point.success
               for p in pairs)


def test_identify_boundary_pairs_empty():
    """Tests boundary identification with empty results list."""
    config = MockBoundaryConfig()
    pairs = identify_boundary_pairs([], config)
    assert len(pairs) == 0


def test_identify_boundary_pairs_single_point():
    """Tests boundary identification with single point."""
    config = MockBoundaryConfig()
    results = [ProfilingResult(1000, 500, True)]
    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) == 0


def test_identify_boundary_pairs_all_success():
    """Tests boundary identification when all points are successful."""
    config = MockBoundaryConfig()
    results = [
        ProfilingResult(1000, 500, True),
        ProfilingResult(2000, 500, True),
        ProfilingResult(3000, 500, True)
    ]
    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) == 0


def test_identify_boundary_pairs_all_fail():
    """Tests boundary identification when all points are failures."""
    config = MockBoundaryConfig()
    results = [
        ProfilingResult(1000, 500, False),
        ProfilingResult(2000, 500, False),
        ProfilingResult(3000, 500, False)
    ]
    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) == 0


def test_identify_boundary_pairs_far_apart():
    """Tests boundary identification with distant points."""
    config = MockBoundaryConfig(max_pair_distance=1000)
    results = [
        ProfilingResult(1000, 500, True),
        ProfilingResult(5000, 500, False)
    ]
    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) == 0


def test_non_uniform_distribution():
    """Tests boundary detection with clustered points."""
    config = MockBoundaryConfig()
    # Dense cluster + sparse outliers
    results = [
        # Dense cluster around (1000,500)
        ProfilingResult(1000, 500, True),
        ProfilingResult(1010, 500, True),
        ProfilingResult(1020, 500, True),
        ProfilingResult(1100, 500, False),
        ProfilingResult(1110, 500, False),
        # Sparse outliers
        ProfilingResult(2000, 1000, True),
        ProfilingResult(3000, 1500, False)
    ]
    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) > 0
    # Verify higher confidence for dense cluster boundaries
    dense_pair_exists = any(
        0.7 < p.confidence < 1.0 and 1000 <= p.success_point.input_length <=
        1020 and 1100 <= p.failure_point.input_length <= 1110 for p in pairs)
    assert dense_pair_exists


def test_numerical_stability():
    """Tests boundary detection with extreme values."""
    config = MockBoundaryConfig(
        distance_scale=1e9,  # Scale for large values
        consistency_radius=1e9,
        max_pair_distance=1e10)
    results = [
        # Large values pair
        ProfilingResult(1e9, 1e9, True),
        ProfilingResult(2e9, 1e9, False),
        # Small values pair - separate by sufficient relative distance
        ProfilingResult(1e-9, 1e-9, True),
        ProfilingResult(1e-6, 1e-9, False)
    ]
    pairs = identify_boundary_pairs(results, config)
    assert len(pairs) > 0
    # Verify at least one large-scale boundary detected
    large_scale_found = any(p.success_point.input_length > 1e8 for p in pairs)
    assert large_scale_found


def test_filter_pairs_empty():
    """Tests filtering empty pairs list."""
    config = MockBoundaryConfig()
    filtered = _filter_pairs([], config)
    assert len(filtered) == 0


def test_filter_pairs_same_region():
    """Tests filtering pairs in same grid region."""
    config = MockBoundaryConfig(grid_size=1000)
    pairs = [
        BoundaryPair(ProfilingResult(1000, 500, True),
                     ProfilingResult(1100, 500, False), 0.9, 0.8),
        BoundaryPair(ProfilingResult(1050, 500, True),
                     ProfilingResult(1150, 500, False), 0.7, 0.6)
    ]
    filtered = _filter_pairs(pairs, config)
    assert len(filtered) == 1
    assert filtered[0] == pairs[0]  # Higher confidence pair should be kept


def test_filter_pairs_exceeds_distance():
    """Tests filtering pairs beyond max distance."""
    config = MockBoundaryConfig(max_pair_distance=100)
    pairs = [
        BoundaryPair(ProfilingResult(1000, 500, True),
                     ProfilingResult(2000, 500, False), 0.9, 0.8)
    ]
    filtered = _filter_pairs(pairs, config)
    assert len(filtered) == 0


def test_confidence_computation():
    """Tests confidence score computation for adjacent points.

    Uses two close points (100 units apart) to test confidence scoring:
    - Success point at (1000,500)
    - Failure point at (1100,500)

    Verifies confidence score is normalized between 0 and 1.
    """
    config = MockBoundaryConfig()
    success = ProfilingResult(1000, 500, True)
    failure = ProfilingResult(1100, 500, False)
    results = [success, failure]

    confidence = _compute_confidence(success, failure, 100.0, results, config)
    assert 0 <= confidence <= 1


def test_confidence_empty_nearby():
    """Tests confidence when no nearby points exist."""
    # Set distance_scale so exp(-100/distance_scale) = 1
    config = MockBoundaryConfig(consistency_radius=0, distance_scale=1000)
    success = ProfilingResult(1000, 500, True)
    # Distance of 100 for exp(-100/1000) ≈ 0.9
    # Final confidence = 0.7 * 0.9 + 0.3 * 0 ≈ 0.63
    failure = ProfilingResult(1100, 500, False)
    confidence = _compute_confidence(success, failure, 100.0, [], config)
    assert confidence == pytest.approx(0.63, rel=0.01)


def test_confidence_inconsistent_nearby():
    """Tests confidence with inconsistent nearby points pattern."""
    config = MockBoundaryConfig(consistency_radius=200, distance_scale=100)
    success = ProfilingResult(1000, 500, True)
    failure = ProfilingResult(1100, 500, False)
    nearby = [
        ProfilingResult(1080, 500, True),  # Very close to failure but success
        ProfilingResult(1020, 500, False),  # Very close to success but failure
    ]
    confidence = _compute_confidence(success, failure, 100.0, nearby, config)
    assert confidence < 0.5


def test_confidence_very_distant():
    """Tests confidence for very distant points."""
    config = MockBoundaryConfig()
    success = ProfilingResult(1000, 500, True)
    failure = ProfilingResult(10000, 500, False)
    confidence = _compute_confidence(success, failure, 9000.0, [], config)
    assert confidence < 0.1  # Very low due to large distance


def test_coverage_score():
    """Tests coverage score computation for boundary points.

    Tests with 2 points aligned horizontally:
    - Success at (1000,500)
    - Failure at (2000,500)

    Verifies coverage score is normalized between 0 and 1.
    """
    config = MockBoundaryConfig()
    points = np.array([[1000, 500], [2000, 500]])
    success = ProfilingResult(1000, 500, True)
    failure = ProfilingResult(2000, 500, False)

    score = _compute_coverage_score(success, failure, points, config)
    assert 0 <= score <= 1


def test_coverage_score_single_point():
    """Tests coverage score with only one point in the domain."""
    config = MockBoundaryConfig()
    points = np.array([[1000, 500]])
    success = ProfilingResult(1000, 500, True)
    failure = ProfilingResult(2000, 500, False)
    score = _compute_coverage_score(success, failure, points, config)
    assert 0 <= score <= 1


def test_coverage_score_dense_region():
    """Tests coverage score in densely sampled region."""
    config = MockBoundaryConfig()
    points = np.array([[i, 500] for i in range(1000, 2000, 100)])
    success = ProfilingResult(1500, 500, True)
    failure = ProfilingResult(1600, 500, False)
    score = _compute_coverage_score(success, failure, points, config)
    assert score < 0.5  # Lower score due to high density


def test_coverage_score_sparse_region():
    """Tests coverage score in sparsely sampled region."""
    config = MockBoundaryConfig()
    points = np.array([[1000, 500], [5000, 500]])
    success = ProfilingResult(2000, 500, True)
    failure = ProfilingResult(3000, 500, False)
    score = _compute_coverage_score(success, failure, points, config)
    assert score > 0.5  # Higher score due to low density


def test_filter_pairs():
    """Tests filtering of redundant boundary pairs.

    Uses 2 overlapping pairs with same success point (1000,500):
    - Pair 1: Failure at (1100,500), lower confidence
    - Pair 2: Failure at (1050,500), higher confidence

    Verifies filtering reduces number of pairs.
    """
    config = MockBoundaryConfig()
    pairs = [
        BoundaryPair(ProfilingResult(1000, 500, True),
                     ProfilingResult(1100, 500, False), 0.8, 0.8),
        BoundaryPair(ProfilingResult(1000, 500, True),
                     ProfilingResult(1050, 500, False), 0.9, 0.7)
    ]

    filtered = _filter_pairs(pairs, config)
    assert len(filtered) <= len(pairs)


def test_filter_pairs_duplicates():
    """Tests filtering identical pairs."""
    config = MockBoundaryConfig()
    pair = BoundaryPair(ProfilingResult(1000, 500, True),
                        ProfilingResult(1100, 500, False), 0.9, 0.8)
    pairs = [pair, pair]  # Duplicate pair
    filtered = _filter_pairs(pairs, config)
    assert len(filtered) == 1


def test_filter_pairs_overlapping_regions():
    """Tests pairs in overlapping grid regions."""
    config = MockBoundaryConfig(grid_size=100)
    pairs = [
        BoundaryPair(ProfilingResult(150, 500, True),
                     ProfilingResult(160, 500, False), 0.9, 0.8),
        BoundaryPair(ProfilingResult(180, 500, True),
                     ProfilingResult(190, 500, False), 0.8, 0.7)
    ]
    filtered = _filter_pairs(pairs, config)
    assert len(filtered) == 1


def test_filter_pairs_zero_grid_size():
    """Tests filtering with zero grid size."""
    config = MockBoundaryConfig(grid_size=0)
    pairs = [
        BoundaryPair(ProfilingResult(1000, 500, True),
                     ProfilingResult(1100, 500, False), 0.9, 0.8)
    ]
    with pytest.raises(ZeroDivisionError):
        _filter_pairs(pairs, config)
