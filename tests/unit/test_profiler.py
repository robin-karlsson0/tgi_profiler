import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from transformers import AutoTokenizer

from tgi_profiler.boundary_detection import BoundaryPair
from tgi_profiler.profiler import (InterpolationPoint, ProfilingResult,
                                   TGIMemoryProfiler, load_previous_results,
                                   validate_config_compatibility)


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
    """Test that pairs with differences below min_refinement_dist are skipped.
    """
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
    """Test pairs where one dimension is below min_refinement_dist and other is
    above.
    """
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
    basic_profiler_config.min_refinement_dist = 100  # Increase min. distance
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
    profiler.config.max_input_length = 1400  # Set artificial limit
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
            # Simulate success for smaller configs, failure for larger ones
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


###################################
#  validate_config_compatibility
###################################


def test_validate_config_identical(basic_profiler_config):
    """Test validation passes with identical configurations."""
    saved_config = {
        "model_id": basic_profiler_config.model_id,
        "gpu_ids": basic_profiler_config.gpu_ids,
        "min_input_length": basic_profiler_config.min_input_length,
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
    }

    # Should not raise any exception
    validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_model_mismatch(basic_profiler_config):
    """Test validation fails when model_id differs."""
    saved_config = {
        "model_id": "different-model",
        "gpu_ids": basic_profiler_config.gpu_ids,
        "min_input_length": basic_profiler_config.min_input_length,
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
    }

    with pytest.raises(ValueError,
                       match="Critical parameter mismatch for model_id"):
        validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_gpu_mismatch(basic_profiler_config):
    """Test validation fails when gpu_ids differ."""
    saved_config = {
        "model_id": basic_profiler_config.model_id,
        "gpu_ids": [0, 1],  # Different from basic_profiler_config
        "min_input_length": basic_profiler_config.min_input_length,
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
    }

    with pytest.raises(ValueError,
                       match="Critical parameter mismatch for gpu_ids"):
        validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_sequence_lengths(basic_profiler_config):
    """Test validation fails when sequence length bounds differ."""
    length_params = [
        "min_input_length", "max_input_length", "min_output_length",
        "max_output_length"
    ]

    for param in length_params:
        saved_config = {
            "model_id": basic_profiler_config.model_id,
            "gpu_ids": basic_profiler_config.gpu_ids,
            "min_input_length": basic_profiler_config.min_input_length,
            "max_input_length": basic_profiler_config.max_input_length,
            "min_output_length": basic_profiler_config.min_output_length,
            "max_output_length": basic_profiler_config.max_output_length,
            "grid_size": basic_profiler_config.grid_size,
        }
        saved_config[param] = getattr(basic_profiler_config, param) + 100

        with pytest.raises(ValueError,
                           match=f"Critical parameter mismatch for {param}"):
            validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_missing_param(basic_profiler_config):
    """Test validation fails when a critical parameter is missing."""
    saved_config = {
        "gpu_ids": basic_profiler_config.gpu_ids,
        "min_input_length": basic_profiler_config.min_input_length,
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
        # model_id intentionally omitted
    }

    with pytest.raises(ValueError,
                       match="Critical parameter mismatch for model_id"):
        validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_extra_params(basic_profiler_config):
    """Test validation passes with extra non-critical parameters."""
    saved_config = {
        "model_id": basic_profiler_config.model_id,
        "gpu_ids": basic_profiler_config.gpu_ids,
        "min_input_length": basic_profiler_config.min_input_length,
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
        "extra_param": "value",
        "another_extra": 123
    }

    # Should not raise any exception
    validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_none_values(basic_profiler_config):
    """Test validation handles None values correctly."""
    # Modify basic_profiler_config to have None gpu_ids
    basic_profiler_config.gpu_ids = None

    saved_config = {
        "model_id": basic_profiler_config.model_id,
        "gpu_ids": None,
        "min_input_length": basic_profiler_config.min_input_length,
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
    }

    # Should not raise any exception
    validate_config_compatibility(saved_config, basic_profiler_config)


def test_validate_config_type_mismatch(basic_profiler_config):
    """Test validation fails when parameter types don't match."""
    saved_config = {
        "model_id": basic_profiler_config.model_id,
        "gpu_ids": basic_profiler_config.gpu_ids,
        "min_input_length":
        str(basic_profiler_config.min_input_length),  # Wrong type
        "max_input_length": basic_profiler_config.max_input_length,
        "min_output_length": basic_profiler_config.min_output_length,
        "max_output_length": basic_profiler_config.max_output_length,
        "grid_size": basic_profiler_config.grid_size,
    }

    with pytest.raises(
            ValueError,
            match="Critical parameter mismatch for min_input_length"):
        validate_config_compatibility(saved_config, basic_profiler_config)


#######################################
#  Tests for load_previous_results()
#######################################


@pytest.fixture
def sample_results_data(basic_profiler_config):
    """Fixture providing sample results data for testing."""
    return {
        "config": {
            "model_id": basic_profiler_config.model_id,
            "gpu_ids": basic_profiler_config.gpu_ids,
            "min_input_length": basic_profiler_config.min_input_length,
            "max_input_length": basic_profiler_config.max_input_length,
            "min_output_length": basic_profiler_config.min_output_length,
            "max_output_length": basic_profiler_config.max_output_length,
            "grid_size": basic_profiler_config.grid_size,
        },
        "results": [{
            "input_length": 1000,
            "output_length": 500,
            "success": True,
            "error_type": None,
            "error_msg": None,
            "container_logs": "",
            "timestamp": "2025-01-28T23:12:21.247390"
        }, {
            "input_length": 2000,
            "output_length": 1000,
            "success": False,
            "error_type": "ContainerError",
            "error_msg": "Container failed health check",
            "container_logs": "mock logs",
            "timestamp": "2025-01-28T23:18:21.157456"
        }]
    }


@pytest.fixture
def mock_results_file(tmp_path, sample_results_data):
    """Create a temporary results file with sample data."""
    results_file = tmp_path / "profile_res_20250128_233107.json"
    with open(results_file, 'w') as f:
        json.dump(sample_results_data, f)
    return results_file


###################################
#  load_previous_results
###################################


def test_load_results_success(basic_profiler_config, mock_results_file):
    """Test successful loading of previous results."""
    basic_profiler_config.resume_from_file = mock_results_file

    saved_config, results = load_previous_results(basic_profiler_config)

    # Verify config was loaded correctly
    assert saved_config["model_id"] == basic_profiler_config.model_id
    assert saved_config["grid_size"] == basic_profiler_config.grid_size

    # Verify results were converted to ProfilingResult objects
    assert len(results) == 2
    assert isinstance(results[0], ProfilingResult)
    assert isinstance(results[1], ProfilingResult)

    # Verify result attributes
    assert results[0].success is True
    assert results[0].input_length == 1000
    assert results[0].output_length == 500
    assert results[0].error_type is None

    assert results[1].success is False
    assert results[1].input_length == 2000
    assert results[1].output_length == 1000
    assert results[1].error_type == "ContainerError"

    # Verify timestamps were parsed correctly
    assert isinstance(results[0].timestamp, datetime)
    assert str(results[0].timestamp.year) == "2025"


def test_load_results_file_not_found(basic_profiler_config, tmp_path):
    """Test error handling when results file doesn't exist."""
    basic_profiler_config.resume_from_file = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError, match="Results file not found"):
        load_previous_results(basic_profiler_config)


def test_load_results_invalid_json(basic_profiler_config, tmp_path):
    """Test error handling for invalid JSON file."""
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, 'w') as f:
        f.write("not valid json")

    basic_profiler_config.resume_from_file = invalid_file

    with pytest.raises(ValueError, match="Invalid results file format"):
        load_previous_results(basic_profiler_config)


def test_load_results_missing_required_fields(basic_profiler_config,
                                              sample_results_data, tmp_path):
    """Test error handling when required fields are missing."""
    # Add a result missing required fields
    sample_results_data["results"].append({
        "input_length": 1000,  # Missing output_length and success
        "error_type": None
    })

    incomplete_file = tmp_path / "incomplete.json"
    with open(incomplete_file, 'w') as f:
        json.dump(sample_results_data, f)

    basic_profiler_config.resume_from_file = incomplete_file

    with pytest.raises(ValueError, match="Missing required fields"):
        load_previous_results(basic_profiler_config)


def test_load_results_invalid_result_format(basic_profiler_config,
                                            sample_results_data, tmp_path):
    """Test error handling when result objects have invalid format."""
    # Modify a result to have invalid format but include all required fields
    sample_results_data["results"].append({
        "input_length": "not an integer",  # Invalid type
        "output_length": 500,
        "success": True,
        "error_type": None,
        "error_msg": None,
        "container_logs": ""
    })

    invalid_file = tmp_path / "invalid_results.json"
    with open(invalid_file, 'w') as f:
        json.dump(sample_results_data, f)

    basic_profiler_config.resume_from_file = invalid_file

    with pytest.raises(ValueError, match="input_length must be numeric"):
        load_previous_results(basic_profiler_config)


def test_load_results_incompatible_config(basic_profiler_config,
                                          sample_results_data, tmp_path):
    """Test error handling when saved config is incompatible."""
    # Modify config to be incompatible
    sample_results_data["config"]["model_id"] = "different-model"

    incompatible_file = tmp_path / "incompatible.json"
    with open(incompatible_file, 'w') as f:
        json.dump(sample_results_data, f)

    basic_profiler_config.resume_from_file = incompatible_file

    with pytest.raises(ValueError,
                       match="Critical parameter mismatch for model_id"):
        load_previous_results(basic_profiler_config)


def test_load_results_empty_results(basic_profiler_config, sample_results_data,
                                    tmp_path):
    """Test loading results file with no results."""
    sample_results_data["results"] = []

    empty_file = tmp_path / "empty_results.json"
    with open(empty_file, 'w') as f:
        json.dump(sample_results_data, f)

    basic_profiler_config.resume_from_file = empty_file

    saved_config, results = load_previous_results(basic_profiler_config)
    assert len(results) == 0


def test_load_results_missing_timestamps(basic_profiler_config,
                                         sample_results_data, tmp_path):
    """Test loading results without timestamp fields."""
    # Remove timestamps from results
    for result in sample_results_data["results"]:
        del result["timestamp"]

    no_timestamps_file = tmp_path / "no_timestamps.json"
    with open(no_timestamps_file, 'w') as f:
        json.dump(sample_results_data, f)

    basic_profiler_config.resume_from_file = no_timestamps_file

    saved_config, results = load_previous_results(basic_profiler_config)
    assert results[0].timestamp is not None  # Should have default timestamp
