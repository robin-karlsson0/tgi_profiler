"""Text Generation Inference (TGI) Memory Profiler.

A framework for empirically determining maximum sequence length capabilities
of LLM models deployed via TGI, avoiding out-of-memory (OOM) errors.

Key Features:
- Adaptive grid search to find viable (input_length, output_length) pairs
- Container-based testing with automated cleanup
- Tokenizer-aware sequence length validation
- Progress tracking and detailed logging
- JSON-serializable results

Example Usage:
    config = ProfilerConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=[0],
        min_input_length=128
        max_input_length=8192
        min_output_length=128
        max_output_length=4096
        grid_size=10
    )
    results = profile_model(config)
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from huggingface_hub import InferenceClient
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

from tgi_profiler.boundary_detection import (BoundaryPair,
                                             identify_boundary_pairs)
from tgi_profiler.config import ProfilerConfig
from tgi_profiler.tgi_container import TGIConfig, TGIContainer
from tgi_profiler.utils.colored_logging import ColoredLogger

logger = ColoredLogger(name=__name__)

# Ensure maximum input length is at least tested inupt length
INPUT_OVERHEAD_MARGIN = 50


class TokenGenerationError(Exception):
    """Raised when model fails to generate text with target token length."""

    def __init__(self, target_length: int, actual_length: int, attempts: int):
        self.target_length = target_length
        self.actual_length = actual_length
        self.attempts = attempts
        super().__init__(
            f"Failed to generate text with target length {target_length} "
            f"after {attempts} attempts. Last attempt length: {actual_length}")


@dataclass
class ProfilingResult:
    """Results from testing specific input/output length combinations.

    Captures success/failure status and diagnostic information from attempting
    to run inference with given sequence lengths.

    Attributes:
        input_length: Target input sequence length tested
        output_length: Target output sequence length tested
        success: Whether inference completed without OOM
        error_type: Classification of failure mode if unsuccessful
        container_logs: Docker container logs for debugging
        error_msg: Detailed error description if applicable
        timestamp: When test was performed

    Example:
        result = ProfilingResult(
            input_length=1024,
            output_length=2048,
            success=False,
            error_type="OOMError"
        )
    """
    input_length: int
    output_length: int
    success: bool
    error_type: Optional[str] = None
    container_logs: str = ""
    error_msg: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "input_length": self.input_length,
            "output_length": self.output_length,
            "success": self.success,
            "error_type": self.error_type,
            "error_msg": self.error_msg,
            "container_logs": self.container_logs,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class InterpolationPoint:
    input_length: int
    output_length: int


def validate_config_compatibility(saved_config: Dict,
                                  current_config: ProfilerConfig) -> None:
    """Validate compatibility between saved and current configurations.

    Checks if critical parameters match between configs to ensure safe
    resumption. Non-critical parameters are allowed to differ.

    Args:
        saved_config: Configuration dictionary from results file
        current_config: Current ProfilerConfig instance

    Raises:
        ValueError: If critical parameters don't match
    """
    critical_params = [
        "model_id",
        "gpu_ids",
        "min_input_length",
        "max_input_length",
        "min_output_length",
        "max_output_length",
        "grid_size",
        "multimodal",
    ]

    for param in critical_params:
        saved_value = saved_config.get(param)
        current_value = getattr(current_config, param)

        if saved_value != current_value:
            raise ValueError(f"Critical parameter mismatch for {param}. "
                             f"Saved value: {saved_value}, "
                             f"Current value: {current_value}")

        # If multimodal is True, also check dummy_image_path
        if param == "multimodal" and saved_value is True:
            saved_path = saved_config.get("dummy_image_path")
            current_path = str(current_config.dummy_image_path)
            if saved_path != current_path:
                raise ValueError(
                    f"Critical parameter mismatch for dummy_image_path. "
                    f"Saved value: {saved_path}, Current value: {current_path}"
                )


def load_previous_results(
        config: ProfilerConfig) -> Tuple[Dict, List[ProfilingResult]]:
    """Load and validate previous results from file.

    Args:
        config: Current ProfilerConfig with resume_from_file specified

    Returns:
        Tuple containing saved config dict and list of ProfilingResult objects

    Raises:
        FileNotFoundError: If results file doesn't exist
        ValueError: If file format is invalid or configs are incompatible
    """
    if not config.resume_from_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {config.resume_from_file}")

    try:
        with open(config.resume_from_file) as f:
            data = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("Invalid results file format")

    if not isinstance(data,
                      dict) or "config" not in data or "results" not in data:
        raise ValueError("Invalid results file format")

    # Validate config compatibility
    validate_config_compatibility(data["config"], config)

    # Convert results back to ProfilingResult objects
    results = []
    required_fields = ["input_length", "output_length", "success"]

    for r in data["results"]:
        # Validate required fields exist
        missing_fields = [field for field in required_fields if field not in r]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in result: {missing_fields}")

        # Validate field types
        if not isinstance(r["input_length"], (int, float)):
            raise ValueError("input_length must be numeric")
        if not isinstance(r["output_length"], (int, float)):
            raise ValueError("output_length must be numeric")
        if not isinstance(r["success"], bool):
            raise ValueError("success must be boolean")

        result = ProfilingResult(input_length=r["input_length"],
                                 output_length=r["output_length"],
                                 success=r["success"],
                                 error_type=r.get("error_type"),
                                 error_msg=r.get("error_msg"),
                                 container_logs=r.get("container_logs", ""))
        if "timestamp" in r:
            result.timestamp = datetime.fromisoformat(r["timestamp"])
        results.append(result)

    return data["config"], results


class TGIMemoryProfiler:
    """Memory profiler for Text Generation Inference deployments.

    Uses adaptive grid search to identify maximum viable sequence lengths
    for a given model and GPU configuration. Manages container lifecycle,
    tokenization, and result collection.

    Key Features:
        - Automated boundary detection between viable/OOM regions
        - Token-exact sequence length testing
        - Configurable retry logic and refinement rounds
        - Progress tracking and detailed logging
        - Result persistence and analysis

    Attributes:
        config: Controls profiling parameters and model configuration
        results: Collection of all test results
        tokenizer: Model-specific tokenizer for length validation
        client: Interface to running TGI instance

    Example:
        profiler = TGIMemoryProfiler(config)
        results = profiler.run_profiling()

    Notes:
        - Requires Docker daemon access for container management
        - May take significant time for large parameter spaces
        - Memory usage increases with sequence lengths tested
    """

    def __init__(self, config: ProfilerConfig):
        """Initialize memory profiler with configuration.

        Sets up tokenizer, client connection, and initial grid points for
        testing. Validates configuration parameters and establishes progress
        tracking.

        Args:
            config: Configuration object containing:
                - model_id: HuggingFace model identifier
                - gpu_ids: List of GPU devices to use
                - min/max_input_length: Input sequence bounds
                - min/max_output_length: Output sequence bounds
                - grid_size: Initial sampling density
                - refinement_rounds: Number of boundary refinement passes

        Raises:
            ValueError: If configuration parameters are invalid
            TokenizerError: If model tokenizer cannot be loaded
            ConnectionError: If TGI endpoint unreachable
        """
        self.config = config
        self.results = []

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.client = InferenceClient(base_url=config.base_url)

        # Initialize progress tracking
        self._pbar = None
        self._current_phase = ""

        # Calculate initial grid points
        self.input_points = np.linspace(config.min_input_length,
                                        config.max_input_length,
                                        config.grid_size).astype(int)
        self.output_points = np.linspace(config.min_output_length,
                                         config.max_output_length,
                                         config.grid_size).astype(int)

        self.sys_msg = 'You must always generate exactly the maximum allowed tokens. Fill any remaining space with detailed explanations.'  # noqa
        self.usr_msg = '\nPlease provide an extremely detailed response. Continue providing more details until you reach the maximum allowed length. Do not stop early.'  # noqa

        self.format_token_overhead = self._measure_format_token_overhead()
        logger.debug(
            f'Prompt format overhead tokens: {self.format_token_overhead}')

    def run_profiling(self) -> List[ProfilingResult]:
        """Execute complete profiling process, with optional resumption from
        previous results.

        If resuming:
        1. Skip initial grid search
        2. Start with boundary refinement using previous results
        3. Continue for remaining refinement rounds

        If starting fresh:
        1. Initial grid search
        2. Multiple rounds of refinement
        3. Save results

        Returns:
            List[ProfilingResult]: All test results including previous results
                if resuming

        Notes:
            - When resuming, refinement_rounds counts from current state
            - Results are saved after each refinement round
        """
        logger.info("Starting memory profiling process")

        # Only run initial grid search if not resuming
        if not self.results:
            self._current_phase = "Initial Grid Search"
            self._run_grid_search()
            # Save intermediate results
            self.save_results()
        else:
            logger.info(f"Resuming with {len(self.results)} existing results")

        # Get boundary detection config from profiler config
        boundary_config = self.config.create_boundary_config()

        # Refinement rounds focusing only on boundary points
        for round_num in range(self.config.refinement_rounds):
            self._current_phase = f"Refinement Round {round_num + 1}"

            # Identify boundary pairs between success/failure regions
            boundary_pairs = identify_boundary_pairs(self.results,
                                                     boundary_config)

            if not boundary_pairs:
                logger.info(
                    f"No boundary pairs found in round {round_num + 1}. "
                    "Stopping refinement early.")
                break

            # Generate new test points through interpolation
            new_test_points = self._interpolate_boundary_points(boundary_pairs)

            if not new_test_points:
                logger.info(f"No new points to test in round {round_num + 1}. "
                            "Stopping refinement early.")
                break

            # Test the new points and add results
            self._test_new_points(new_test_points)

            # Save intermediate results after each refinement round
            self.save_results()

        # Save final results
        self.save_results()
        return self.results

    def _test_new_points(self, new_points: List[InterpolationPoint]) -> None:
        """Test newly interpolated points near the success/failure boundary.

        Rather than retesting the entire grid, this method focuses only on
        testing new points generated through boundary interpolation. It:
        1. Filters out any points that have already been tested
        2. Shows progress specifically for boundary refinement testing
        3. Adds results to the overall results collection

        Args:
            new_points: List of InterpolationPoint objects containing
                input_length and output_length to test

        Updates:
            self.results: Appends ProfilingResult for each new test point

        Notes:
            - Skips points that have already been tested to avoid redundancy
            - Uses test_point() method for actual testing which handles Docker
              container lifecycle
            - Updates progress bar to show boundary refinement progress
        """
        if not new_points:
            logger.info("No new points to test in this refinement round")
            return

        # Create set of previously tested configurations
        tested_configs = {(result.input_length, result.output_length)
                          for result in self.results}

        # Filter out points that have already been tested
        points_to_test = [
            point for point in new_points
            if (point.input_length, point.output_length) not in tested_configs
        ]

        if not points_to_test:
            logger.info("All interpolated points have already been tested")
            return

        logger.info(f"Testing {len(points_to_test)} new boundary points")

        # Create progress bar for boundary refinement
        with tqdm(total=len(points_to_test),
                  desc=f"{self._current_phase} - Boundary Refinement") as pbar:

            for point in points_to_test:
                try:
                    # Test the point and record result
                    result = self.test_point(input_length=point.input_length,
                                             output_length=point.output_length)
                    self.results.append(result)

                    # Log success/failure status
                    status = "succeeded" if result.success else "failed"
                    logger.debug(
                        f"Point ({point.input_length}, {point.output_length}) {status}"  # noqa
                    )

                    pbar.update(1)

                except Exception as e:
                    logger.error(f"Error testing point ({point.input_length}, "
                                 f"{point.output_length}): {str(e)}")
                    # Continue with remaining points if one fails

    def _interpolate_boundary_points(
        self,
        boundary_pairs: List[BoundaryPair],
    ) -> List[InterpolationPoint]:
        """Generate new test points by interpolating between successful and
        failed configurations.

        For each boundary pair, computes the midpoint between the success and
        failure points in both input and output dimensions. Only generates
        points when at least one dimension has a difference larger than
        min_refinement_dist.

        Args:
            boundary_pairs: List of BoundaryPair objects, each containing
                success and failure ProfilingResult points with their
                confidence and coverage scores.

        Returns:
            List[InterpolationPoint]: Unique interpolation points for further
                testing. Each point contains input_length and output_length
                within the configured bounds. Points are deduplicated to avoid
                redundant testing.

        Notes:
            - Skips points if both input and output differences are below
              min_refinement_dist
            - Skips points outside the configured min/max bounds for
              input/output lengths
            - Deduplicates points with identical input/output lengths keeping
              only first occurrence
            - Empty input list results in empty output list

        Example:
            For a boundary pair with:
            - Success point: (input=1000, output=2000)
            - Failure point: (input=2000, output=2000)
            Will generate:
            - Interpolation point: (input=1500, output=2000)
            Assuming the difference exceeds min_refinement_dist and point is
            within bounds.
        """
        new_points = []

        for pair in boundary_pairs:
            input_diff = abs(pair.failure_point.input_length -
                             pair.success_point.input_length)
            output_diff = abs(pair.failure_point.output_length -
                              pair.success_point.output_length)

            # Add midpoint if at least one dimension is sufficiently separated
            do_input_ref = input_diff >= self.config.min_refinement_dist
            do_output_ref = output_diff >= self.config.min_refinement_dist
            if do_input_ref or do_output_ref:
                mid_input = (pair.success_point.input_length +
                             pair.failure_point.input_length) // 2
                mid_output = (pair.success_point.output_length +
                              pair.failure_point.output_length) // 2

                # Skip points outside configured bounds
                if (mid_input < self.config.min_input_length
                        or mid_input > self.config.max_input_length
                        or mid_output < self.config.min_output_length
                        or mid_output > self.config.max_output_length):
                    continue

                new_points.append(
                    InterpolationPoint(input_length=mid_input,
                                       output_length=mid_output))

        # Remove duplicates using a dictionary to maintain order
        seen = {}
        for point in new_points:
            seen[(point.input_length, point.output_length)] = point

        return list(seen.values())

    def _measure_format_token_overhead(self) -> int:
        """Measure token overhead from message format structure.

        Determines how many tokens are used by the message format itself
        (system message, role indicators, content structure) by creating
        a minimal message with empty/minimal content.

        Returns:
            int: Number of tokens used by the message format
        """
        system_message = {
            "role": "system",
            "content": self.sys_msg,
        }

        if not self.config.multimodal:
            # For text-only format
            user_message = {
                "role": "user",
                "content": self.usr_msg,
            }
        else:
            # For multimodal format
            user_message = {
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            "data:image/jpeg;base64,/9j/"  # Minimal base64
                        }
                    },
                    {
                        "type": "text",
                        "text": self.usr_msg
                    }
                ]
            }
        messages = [system_message, user_message]

        # Convert to string to count tokens
        format_str = str(messages)
        num_format_tokens = self._count_tokens(format_str)
        return num_format_tokens

    def _count_tokens(self, text: str) -> int:
        """Count exact number of tokens using model's tokenizer.

        Args:
            text: Input text to tokenize

        Returns:
            int: Token count

        Raises:
            TokenizerError: If tokenization fails

        Note:
            Uses model's own tokenizer for accuracy, not approximations
        """
        try:
            tokens = self.tokenizer.tokenize(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {str(e)}")
            raise

    def _generate_exact_token_input(self, target_length: int) -> str:
        """Generate text that tokenizes to exact target length.

        Uses iterative approach to create text matching target token count:
        1. Generate initial text longer than target
        2. Trim characters until token count matches target

        Args:
            target_length: Desired number of tokens in output text

        Returns:
            str: Text that tokenizes to exactly target_length tokens

        Raises:
            ValueError: If target length cannot be achieved
            TokenizerError: If tokenization fails

        Note:
            - Critical for accurate memory profiling
            - May take longer for very large target lengths
            - Returns minimal text achieving target length
        """
        base_text = "What is the meaning of life? "
        text = base_text * (target_length // 8)
        current_length = self._count_tokens(text)

        # Repeat string until it is longer than required token length
        iter_counter = 1
        while current_length < target_length:
            logger.debug(
                'Iter {iter_counter:02d} | Input length: {current_length}')
            text += base_text
            current_length = self._count_tokens(text)
            iter_counter += 1

        # Remove excess tokens one character at a time
        while True:
            new_length = self._count_tokens(text[:-1])
            if new_length < target_length:
                break
            else:
                text = text[:-1]
        logger.debug('Input length after trimming: {new_length}')

        return text

    def _generate_exact_token_output(self, target_length: int,
                                     input_txt: str) -> tuple[str, int]:
        """Generate model output with exact target token length.

        Uses messaging and retries to achieve output length within tolerance:
        1. Creates appropriate messages based on modality (text-only or
           multimodal)
        2. For multimodal, includes a dummy image in the prompt while only
           counting text tokens towards the target length
        3. Tracks best attempt across retries
        4. Verifies token count matches target

        Args:
            target_length: Target number of tokens for generated text
            input_txt: Prompt text to generate from

        Returns:
            tuple[str, int]: Generated text and its token count. For multimodal
                models, the token count excludes image tokens.

        Raises:
            TokenGenerationError: If output length not within OUTPUT_TOLERANCE
                after retries
            APIError: If model inference fails
            FileNotFoundError: If dummy image not found in multimodal mode
            PIL.UnidentifiedImageError: If image cannot be opened/processed

        Notes:
            - Uses system prompt to encourage verbose output
            - Tracks closest attempt if target not hit
            - In multimodal mode, includes a constant image overhead while
            focusing on text token capacity
            - Token count tolerance set by output_tolerance_pct config param
            - High temperature prevents early EOS token generation
        """
        messages = self._create_messages(input_txt)
        best_attempt = {'text': None, 'length': 0, 'diff': float('inf')}

        # Attempt inference and verify lengths
        for attempt in range(self.config.retries_per_point):
            logger.debug(f"Attempt {attempt + 1}/"
                         f"{self.config.retries_per_point}")

            try:
                # High temperature prevents EOS token from being generated
                response = self.client.chat_completion(
                    messages,
                    max_tokens=target_length,
                    temperature=self.config.temp,
                )
                output_txt = response.choices[0].message.content

                output_len = self._count_tokens(output_txt)
                length_diff = abs(output_len - target_length)

                logger.info(
                    f"\tOutput length: {output_len} | Expected length: {target_length} (Diff: {length_diff})"  # noqa
                )

                if length_diff < output_len * self.config.output_tolerance_pct:
                    return output_txt, output_len

                if length_diff < best_attempt['diff']:
                    best_attempt = {
                        'text': output_txt,
                        'length': output_len,
                        'diff': length_diff
                    }

            except Exception as e:
                logger.error(
                    f"Inference failed on attempt {attempt + 1}: {str(e)}")
                if attempt == self.config.retries_per_point - 1:
                    raise

        raise TokenGenerationError(target_length=target_length,
                                   actual_length=best_attempt['length'],
                                   attempts=self.config.retries_per_point)

    def _create_messages(self, input_txt: str) -> List[dict]:
        """Create messages for model inference based on modality.

        Constructs a list of messages suitable for chat completion API, with
        format varying based on whether multimodal mode is enabled. In
        text-only mode, creates standard chat messages. In multimodal mode,
        includes the dummy image alongside text in the user message.

        Args:
            input_txt: Base prompt text to include in user message. Will be
                augmented with instructions to generate maximum length output.

        Returns:
            List[dict]: Messages formatted for chat completion API. Always
                includes a system message and a user message. In multimodal
                mode, the user message contains both image and text content
                following TGI's multimodal message format.

        Raises:
            FileNotFoundError: If dummy image not found in multimodal mode
            PIL.UnidentifiedImageError: If image cannot be opened/processed

        Example formats:
            Text-only mode:
            [
                {'role': 'system', 'content': '...'},
                {'role': 'user', 'content': 'text...'}
            ]

            Multimodal mode:
            [
                {'role': 'system', 'content': '...'},
                {'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': img_data_url}},
                    {'type': 'text', 'text': 'text...'}
                ]}
            ]

        Notes:
            - System message encourages maximum token generation
            - Image is encoded at 90% quality to balance size and quality
            - Message format follows TGI chat completion API requirements
        """
        system_msg = {
            'role': 'system',
            'content': self.sys_msg,
        }

        if not self.config.multimodal:
            user_msg = {
                'role': 'user',
                'content': f'{input_txt}{self.usr_msg}',
            }
        else:
            # Read and encode the dummy image
            query_img = Image.open(self.config.dummy_image_path)
            image_quality = 90
            img_data_url = self.image_to_data_url(query_img, image_quality)

            user_msg = {
                'role':
                'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': img_data_url
                        },
                    },
                    {
                        'type': 'text',
                        'text': f'{input_txt}{self.usr_msg}',
                    },
                ],
            }

        return [system_msg, user_msg]

    def test_point(self, input_length: int,
                   output_length: int) -> ProfilingResult:
        """Test TGI memory capacity with specific sequence lengths.

        Validates if model can handle given input/output lengths without OOM:
        1. Configures and starts TGI container
        2. Generates input text of exact token length
        3. Attempts model inference with max token generation
        4. Monitors for OOM and token length errors

        Args:
            input_length: Target input sequence length
            output_length: Target output sequence length

        Returns:
            ProfilingResult containing:
            - Success/failure status
            - Actual token lengths achieved
            - Error details and container logs if failed

        Raises:
            None - errors captured in ProfilingResult

        Notes:
            - Accounts for message prompt tokens in input length
            - Returns early on TokenGenerationError
            - Captures container logs on failure
            - OOM errors indicate memory limit found
            - Success requires exact token match
        """
        tgi_config = TGIConfig(
            model_id=self.config.model_id,
            gpu_ids=self.config.gpu_ids,
            port=self.config.port,
            # Ensure tested input length includes message format tokens
            max_input_length=input_length + INPUT_OVERHEAD_MARGIN,
            max_output_length=output_length,
            hf_token=self.config.hf_token,
            hf_cache_dir=self.config.hf_cache_dir,
        )

        logger.info(
            f'Testing (input, output) = ({input_length}, {output_length})')

        try:
            with TGIContainer(tgi_config) as container:  # noqa
                logger.debug("Container started successfully")

                # Create input that will tokenize to exact length minus
                # approx. tokens required for message formatting
                input_txt = self._generate_exact_token_input(
                    input_length - self.format_token_overhead)
                logger.debug(f'Generated input text of length '
                             f'{self._count_tokens(input_txt)}')

                # Generate output text with exact token count
                # - Retry until successful or max attempts reached and
                #   return a failiure due to token length mismatch
                try:
                    out = self._generate_exact_token_output(
                        output_length, input_txt)
                    output_txt, output_len = out
                except TokenGenerationError as e:
                    logger.warning(f"Exact token generation failed: {str(e)}")
                    return ProfilingResult(input_length=input_length,
                                           output_length=output_length,
                                           success=False,
                                           error_type="TokenGenerationError",
                                           error_msg=str(e))

                logger.debug(f"Generated output text of length {output_len}")

                actual_input_tokens = self._count_tokens(
                    input_txt) + self.format_token_overhead
                return ProfilingResult(input_length=actual_input_tokens,
                                       output_length=output_len,
                                       success=True)

        except Exception as e:
            container_logs = ""
            if hasattr(e, "container") and e.container:
                container_logs = e.container.logs().decode('utf-8')

            return ProfilingResult(input_length=input_length,
                                   output_length=output_length,
                                   success=False,
                                   container_logs=container_logs,
                                   error_type=type(e).__name__,
                                   error_msg=str(e))

    def _run_grid_search(self) -> None:
        """Execute grid search across sequence length combinations.

        Systematically tests every input/output length pair in current grid:
        1. Calculates total test points needed
        2. Tests each length combination with test_point()
        3. Records results in self.results
        4. Displays progress with tqdm bar

        Updates:
            self.results: Appended with ProfilingResult for each test
            self._pbar: Progress bar showing completion status

        Note:
            Grid points defined by self.input_points and self.output_points
        """
        total_points = len(self.input_points) * len(self.output_points)

        with tqdm(total=total_points, desc=self._current_phase) as self._pbar:
            for i in self.input_points:
                for o in self.output_points:
                    result = self.test_point(int(i), int(o))
                    self.results.append(result)
                    self._pbar.update(1)

    def save_results(self) -> None:
        """Save profiling results to output directory.

        Writes JSON file containing:
        - Critical configuration parameters needed for resumption
        - All test results with timestamps
        - Success/failure data for each point

        Critical parameters include:
        - Model and GPU configuration
        - Sequence length bounds
        - Grid search parameters
        - Token generation parameters

        Non-critical parameters like output directory, logging settings, etc.
        are controlled by the current run configuration.

        Creates:
            profile_results_{timestamp}.json in config.output_dir
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.output_dir / f"profile_res_{timestamp}.json"

        # Extract critical parameters for resumption
        critical_config = {
            # Model & hardware config
            "model_id":
            self.config.model_id,
            "gpu_ids":
            self.config.gpu_ids,
            "base_url":
            self.config.base_url,

            # Sequence length bounds
            "min_input_length":
            self.config.min_input_length,
            "max_input_length":
            self.config.max_input_length,
            "min_output_length":
            self.config.min_output_length,
            "max_output_length":
            self.config.max_output_length,

            # Grid search parameters
            "grid_size":
            self.config.grid_size,
            "refinement_rounds":
            self.config.refinement_rounds,

            # Token generation parameters
            "output_tolerance_pct":
            self.config.output_tolerance_pct,
            "temp":
            self.config.temp,

            # Multimodal configuration
            "multimodal":
            self.config.multimodal,
            "dummy_image_path":
            str(self.config.dummy_image_path)
            if self.config.dummy_image_path else None,

            # Boundary detection parameters
            "k_neighbors":
            self.config.k_neighbors,
            "m_random":
            self.config.m_random,
            "distance_scale":
            self.config.distance_scale,
            "consistency_radius":
            self.config.consistency_radius,
            "redundancy_weight":
            self.config.redundancy_weight,
            "max_pair_distance":
            self.config.max_pair_distance,
            "min_refinement_dist":
            self.config.min_refinement_dist,
        }

        with open(results_file, 'w') as f:
            json.dump(
                {
                    "config": critical_config,
                    "results": [r.to_dict() for r in self.results]
                },
                f,
                indent=2)

        logger.info(f"Results saved to {results_file}")

    @staticmethod
    def image_to_data_url(img: Image,
                          img_quality: int = 80,
                          format: str = 'JPEG') -> str:
        '''
        Ref: https://github.com/huggingface/huggingface_hub/pull/2556
        '''
        with io.BytesIO() as buffer:
            img.save(buffer, format=format, quality=img_quality)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_data_url = f"data:image/{format.lower()};base64,{img_str}"
        return img_data_url


def profile_model(config: ProfilerConfig) -> List[ProfilingResult]:
    """Run profiling with given configuration, optionally resuming from
    previous results.

    Args:
        config: ProfilerConfig specifying profiling parameters and optional
            resume_from_file path

    Returns:
        List[ProfilingResult]: Results of profiling

    Raises:
        FileNotFoundError: If resume_from_file specified but not found
        ValueError: If resume file is invalid or incompatible
    """
    profiler = TGIMemoryProfiler(config)
    if config.resume_from_file:
        logger.info(
            f"Resuming from previous results: {config.resume_from_file}")
        try:
            saved_config, previous_results = load_previous_results(config)

            # Initialize profiler with previous results
            profiler.results = previous_results

            # Log resumption details
            logger.info(f"Loaded {len(previous_results)} previous results")
            logger.info("Skipping initial grid search")

            # Set current phase to indicate resumption
            profiler._current_phase = "Resuming Refinement"

        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Failed to resume from file: {e}")
            exit()

    return profiler.run_profiling()


if __name__ == "__main__":
    import os
    from pathlib import Path

    # Test configuration for quick validation
    config = ProfilerConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        gpu_ids=[0],
        min_input_length=2048,
        max_input_length=32768,  # Small range for testing
        min_output_length=512,
        max_output_length=32768,  # Small range for testing
        grid_size=2,  # Minimal grid for quick testing
        refinement_rounds=1,  # Single refinement for testing
        port=8080,
        output_dir=Path("./test_results2"),
        hf_token=os.getenv("HF_TOKEN"),
        retries_per_point=8,
        # resume_from_file='PATH/TO/YOUR/FILE.json',
    )

    # Create output directory if it doesn't exist
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting test profiling run")
    results = profile_model(config)

    # Print summary of results
    successes = sum(1 for r in results if r.success)
    logger.info(f"Completed {len(results)} test points")
    logger.info(f"Successful points: {successes}")
    logger.info(f"Failed points: {len(results) - successes}")

    # Print maximum successful lengths found
    successful_results = [r for r in results if r.success]
    if successful_results:
        max_input = max(r.input_length for r in successful_results)
        max_output = max(r.output_length for r in successful_results)
        logger.info(f"Maximum successful input length: {max_input}")
        logger.info(f"Maximum successful output length: {max_output}")
    else:
        logger.warning("No successful test points found")
