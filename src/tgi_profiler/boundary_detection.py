from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from scipy.spatial import KDTree

if TYPE_CHECKING:
    from tgi_profiler.profiler import ProfilingResult

from tgi_profiler.utils.colored_logging import ColoredLogger

logger = ColoredLogger(name=__name__)


@dataclass
class BoundaryPair:
    success_point: ProfilingResult
    failure_point: ProfilingResult
    confidence: float
    region_coverage: float


@dataclass
class BoundaryConfig:
    """Configuration parameters for boundary pair detection.

    Attributes:
        k_neighbors: Number of nearest neighbors to consider for local boundary
            detection
        m_random: Number of random points to sample for global boundary
            exploration
        distance_scale: Scale factor for distance-based scoring (higher = more
            tolerance for distant points)
        consistency_radius: Maximum distance to consider points for consistency
            scoring
        redundancy_weight: Weight factor for penalizing redundant boundary
            pairs (higher = stronger penalty)
        grid_size: Size of grid cells for spatial filtering; Each cell is
            grid_size x grid_size and only keeps highest-scoring boundary pair
            to prevent redundancy
        max_pair_distance: Maximum allowed distance between success/failure
            points in a pair
    """
    k_neighbors: int = 5
    m_random: int = 3
    distance_scale: float = 1000
    consistency_radius: float = 1000
    redundancy_weight: float = 0.5
    grid_size: int = 500
    max_pair_distance: float = 1e12

    def validate(self):
        """Validate configuration parameters."""
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be >= 1")
        if self.m_random < 0:
            raise ValueError("m_random must be >= 0")
        if self.distance_scale <= 0:
            raise ValueError("distance_scale must be positive")
        if self.consistency_radius <= 0:
            raise ValueError("consistency_radius must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.max_pair_distance <= 0:
            raise ValueError("max_pair_distance must be positive")


# Rename point_distance_to to avoid shadowing
def compute_point_distance(p1, p2) -> float:
    """Calculate distance between two ProfilingResults."""
    return float(
        np.linalg.norm([
            p1.input_length - p2.input_length,
            p1.output_length - p2.output_length
        ]))


def _compute_confidence(
    success: ProfilingResult,
    failure: ProfilingResult,
    knn_distance: Optional[float],
    results: List[ProfilingResult],
    config,
    weight_dist=0.7,
    weight_cons=0.3,
) -> float:
    """Compute confidence score for a boundary pair based on distance and local
    consistency.

    Confidence represents how reliable this pair is for describing the true
    boundary between successful and failed memory configurations. Higher
    confidence (closer to 1) means the pair better represents the boundary.

    Combines two metrics:
    - Distance score (70%): Higher for closer points, using exponential decay
    - Consistency score (30%): Higher when nearby points follow expected
      success/failure pattern

    Args:
        success: Point representing successful configuration
        failure: Point representing failed configuration
        knn_distance: Distance to point if from KNN search (optional)
        results: All profiling results for consistency check
        config: Configuration parameters

    Returns:
        float: Confidence score between 0 and 1
    """
    dist = compute_point_distance(success, failure)
    distance_score = np.exp(-dist / config.distance_scale)

    nearby_points = [
        r for r in results
        if compute_point_distance(r, success) < config.consistency_radius
    ]

    consistency_score = sum(
        1 for p in nearby_points if (p.success == success.success) ==
        (compute_point_distance(p, success) < compute_point_distance(
            p, failure))) / max(1, len(nearby_points))

    return weight_dist * distance_score + weight_cons * consistency_score


def _compute_coverage_score(success: ProfilingResult, failure: ProfilingResult,
                            points: np.ndarray, config) -> float:
    """Calculate how well a boundary pair represents an undersampled region.

    Coverage score indicates how valuable this pair is for improving boundary
    detection in sparse areas. Higher scores (closer to 1) mean the pair helps
    define the boundary in a region with few existing test points. Lower scores
    (closer to 0) indicate redundancy with existing boundary information.

    Score decreases exponentially with:
    1. Number of nearby test points (density)
    2. Distance between success and failure points
    3. Overlap with existing boundary pairs

    Args:
        success: Point representing successful configuration
        failure: Point representing failed configuration
        points: Array of all test points coordinates
        config: Parameters controlling density sensitivity

    Returns:
        float: Coverage score between 0 and 1
    """
    midpoint = np.array([(success.input_length + failure.input_length) / 2,
                         (success.output_length + failure.output_length) / 2])

    distances = np.linalg.norm(points - midpoint, axis=1)
    # More points nearby = higher density = lower score
    density = np.sum(np.exp(-distances / config.distance_scale))

    return np.exp(-density * config.redundancy_weight)


def _compute_sampling_weights(points: np.ndarray,
                              current_idx: int) -> np.ndarray:
    """Calculate weights for random neighbor sampling in boundary detection.

    Used in identify_boundary_pairs() to select M random neighbors beyond K
    nearest neighbors. Points farther from current_idx get higher weights,
    promoting broad boundary exploration vs local refinement.

    Args:
        points: (N,2) array of (input_length, output_length) coordinates
        current_idx: Index of current point being analyzed

    Returns:
        (N,) array of normalized distance-based weights summing to 1
    """
    distances = np.linalg.norm(points - points[current_idx], axis=1)
    return distances / distances.sum()


def _filter_pairs(pairs: List[BoundaryPair], config) -> List[BoundaryPair]:
    """Filter boundary pairs to keep most informative ones per grid region.

    Reduces redundancy by selecting highest-scoring pair per grid region while
    enforcing maximum distance constraint. Prioritizes pairs with high
    confidence and region coverage scores.

    Args:
        pairs: List of boundary pairs to filter
        config: Settings for grid size and max pair distance

    Returns:
        Filtered list with at most one high-quality pair per grid region
    """
    filtered = []
    used_regions = set()

    for pair in sorted(pairs,
                       key=lambda p: p.confidence * p.region_coverage,
                       reverse=True):
        region_key = tuple(
            map(lambda x: x // config.grid_size, [
                pair.success_point.input_length,
                pair.success_point.output_length
            ]))

        if (region_key not in used_regions and compute_point_distance(
                pair.success_point, pair.failure_point)
                < config.max_pair_distance):
            filtered.append(pair)
            used_regions.add(region_key)

    return filtered


def identify_boundary_pairs(results: List[ProfilingResult],
                            config: BoundaryConfig) -> List[BoundaryPair]:
    """Detect boundary pairs between successful and failed memory configs.

    Uses a hybrid approach combining local and global boundary detection:
    1. Local: K-nearest neighbors to find close success/failure transitions
    2. Global: M random samples weighted by distance for broad exploration
    3. Filtering: Keeps highest quality pair per grid region

    Boundary quality measured by:
    - Confidence score: Based on distance and local consistency
    - Coverage score: Favors pairs in undersampled regions

    Args:
        results: List of memory profiling results with success/failure status
        config: Parameters controlling boundary detection behavior

    Returns:
        List of boundary pairs sorted by quality, spatially filtered to reduce
            redundancy

    Raises:
        ValueError: If point coordinates are non-finite or config invalid

    Logging:
        Debug: Found pairs and their scores
        Warning: Numerical stability issues
        Error: Point processing failures
    """
    try:
        config.validate()

        if len(results) < 2:
            logger.debug("Insufficient points for boundary detection")
            return []

        points = np.array([[r.input_length, r.output_length] for r in results])
        if not np.all(np.isfinite(points)):
            logger.error("Non-finite values detected in point coordinates")
            raise ValueError("Point coordinates must be finite")

        tree = KDTree(points)
        k = min(config.k_neighbors, len(results) - 1)
        m = min(config.m_random, len(results) - 1)

        pairs = []
        for idx, result in enumerate(results):
            try:
                k_dist, k_idx = tree.query(points[idx], k=k)
                k_idx = np.atleast_1d(k_idx)

                if m > 0:
                    weights = _compute_sampling_weights(points, idx)
                    if not np.all(np.isfinite(weights)):
                        logger.warning(
                            f"Non-finite sampling weights at idx {idx}")
                        continue

                    random_idx = np.random.choice(len(results),
                                                  size=m,
                                                  p=weights)
                    neighbor_indices = np.concatenate([k_idx, random_idx])
                else:
                    neighbor_indices = k_idx

                for neighbor_idx in neighbor_indices:
                    neighbor = results[int(neighbor_idx)]
                    if result.success != neighbor.success:
                        success, failure = (
                            result, neighbor) if result.success else (neighbor,
                                                                      result)

                        dist = compute_point_distance(success, failure)
                        if dist > config.max_pair_distance:
                            continue

                        confidence = _compute_confidence(
                            success, failure,
                            k_dist[np.where(k_idx == neighbor_idx)[0]][0] if
                            neighbor_idx in k_idx else None, results, config)
                        coverage = _compute_coverage_score(
                            success, failure, points, config)

                        logger.debug(
                            f"Found boundary pair: conf={confidence:.3f}, cov={coverage:.3f}"  # noqa
                        )
                        pairs.append(
                            BoundaryPair(success, failure, confidence,
                                         coverage))

            except Exception as e:
                logger.error(f"Error processing point {idx}: {str(e)}")
                continue

        return _filter_pairs(pairs, config)

    except Exception as e:
        logger.error(f"Boundary detection failed: {str(e)}")
        raise
