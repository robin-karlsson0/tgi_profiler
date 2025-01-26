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

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from huggingface_hub import InferenceClient
from tqdm import tqdm
from transformers import AutoTokenizer

from tgi_profiler.config import ProfilerConfig
from tgi_profiler.tgi_container import TGIConfig, TGIContainer
from tgi_profiler.utils.colored_logging import ColoredLogger

logging.basicConfig(level=logging.DEBUG)
logger = ColoredLogger(name=__name__)

INPUT_LEN_MSG_PROMPTING = 77  # Checked with Llama-3.1-8B-Instruct
OUTPUT_TOLERANCE = 100


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

    def run_profiling(self) -> List[ProfilingResult]:
        """Execute complete profiling process.

        This method orchestrates the entire profiling workflow:
        1. Initial grid search
        2. Multiple rounds of refinement
        3. Result saving

        Returns:
            List[ProfilingResult]: All test results
        """
        logger.info("Starting memory profiling process")

        # Initial grid search
        self._current_phase = "Initial Grid Search"
        self._run_grid_search()

        # Refinement rounds
        for round_num in range(self.config.refinement_rounds):
            self._current_phase = f"Refinement Round {round_num + 1}"
            self._refine_grid()
            self._run_grid_search()

        # Save final results
        self.save_results()
        return self.results

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
        1. Prompts model to generate maximum tokens
        2. Tracks best attempt across retries
        3. Verifies token count matches target

        Args:
            target_length: Target number of tokens for generated text
            input_txt: Prompt text to generate from

        Returns:
            tuple[str, int]: Generated text and its token count

        Raises:
            TokenGenerationError: If output length not within OUTPUT_TOLERANCE
                after retries
            APIError: If model inference fails

        Example:
            output, length = _generate_exact_token_output(1000, "Explain love")
            assert abs(length - 1000) < OUTPUT_TOLERANCE

        Notes:
            - Uses system prompt to encourage verbose output
            - Tracks closest attempt if target not hit
            - Retries configured via config.retries_per_llm_inference
            - Token count tolerance set by OUTPUT_TOLERANCE constant
        """
        # Create prompt that forces model to generate max length
        system_msg = {
            'role':
            'system',
            'content':
            "You must always generate exactly the maximum allowed tokens. Fill any remaining space with detailed explanations.",  # noqa
        }

        user_msg = {
            'role':
            'user',
            'content':
            f'{input_txt}\nPlease provide an extremely detailed response. Continue providing more details until you reach the maximum allowed length. Do not stop early.',  # noqa
        }

        messages = [system_msg, user_msg]

        best_attempt = {'text': None, 'length': 0, 'diff': float('inf')}

        # Attempt inference and verify lengths
        for attempt in range(self.config.retries_per_point):

            logger.debug(f"Attempt {attempt + 1}/"
                         f"{self.config.retries_per_point}")

            try:
                response = self.client.chat_completion(messages)
                output_txt = response.choices[0].message.content

                output_len = self._count_tokens(output_txt)
                length_diff = abs(output_len - target_length)

                output_len = self._count_tokens(output_txt)

                if length_diff < OUTPUT_TOLERANCE:
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
            max_input_length=input_length,
            max_output_length=output_length,
            hf_token=self.config.hf_token,
            hf_cache_dir=self.config.hf_cache_dir,
        )

        try:
            with TGIContainer(tgi_config) as container:  # noqa
                logger.debug("Container started successfully")

                # Create input that will tokenize to exact length minus
                # approx. tokens required for message formatting
                input_txt = self._generate_exact_token_input(
                    input_length - INPUT_LEN_MSG_PROMPTING)
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
                    input_txt) + INPUT_LEN_MSG_PROMPTING
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
                    result = self.test_point(i, o)
                    self.results.append(result)
                    self._pbar.update(1)

    def _refine_grid(self) -> None:
        """Adapt grid points to focus on memory boundary regions.

        Refines sampling around transition points between successful and failed
        tests:
        1. Creates success/failure matrix from current results
        2. Identifies boundary points (success adjacent to failure)
        3. Adds midpoints between boundary and neighbor points
        4. Updates input_points and output_points arrays

        Raises:
            ValueError: If no results available or invalid points found

        Updates:
            self.input_points: New input length test points
            self.output_points: New output length test points

        Note:
            Critical for efficiently finding memory limits
            Focuses computation on boundary regions
            Maintains sorted, unique point arrays
        """
        logger.info("Starting grid refinement")

        # Validate we have results to analyze
        if not self.results:
            raise ValueError("No results available for grid refinement")

        # Create dictionaries to validate and map input/output lengths
        input_dict = {x: i for i, x in enumerate(self.input_points)}
        output_dict = {x: i for i, x in enumerate(self.output_points)}

        # Validate all result points are in our grids
        for result in self.results:
            if (result.input_length not in input_dict
                    or result.output_length not in output_dict):
                raise ValueError(
                    "Result contains points not in grid: "
                    f"({result.input_length}, {result.output_length})")
        # Create success/failure matrix from results
        input_dict = {x: i for i, x in enumerate(self.input_points)}
        output_dict = {x: i for i, x in enumerate(self.output_points)}

        success_matrix = np.zeros(
            (len(self.input_points), len(self.output_points)))
        for result in self.results:
            if result.input_length in input_dict and result.output_length in output_dict:  # noqa
                i = input_dict[result.input_length]
                j = output_dict[result.output_length]
                success_matrix[i, j] = 1 if result.success else 0

        # Find boundary points (successful points adjacent to failures)
        boundary_inputs = set()
        boundary_outputs = set()

        for i in range(success_matrix.shape[0]):
            for j in range(success_matrix.shape[1]):
                if success_matrix[i, j] == 1:  # If point is successful
                    # Check neighbors
                    is_boundary = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < success_matrix.shape[0]
                                and 0 <= nj < success_matrix.shape[1]
                                and success_matrix[ni, nj] == 0):
                            is_boundary = True
                            break

                    if is_boundary:
                        boundary_inputs.add(self.input_points[i])
                        boundary_outputs.add(self.output_points[j])

        # Generate new points around boundary
        def generate_neighborhood(point: int,
                                  points_list: List[int]) -> List[int]:
            """Generate new test points between point and its neighbors."""
            idx = points_list.index(point)
            new_points = []

            # Add midpoints between point and its neighbors
            if idx > 0:
                new_points.append((point + points_list[idx - 1]) // 2)
            if idx < len(points_list) - 1:
                new_points.append((point + points_list[idx + 1]) // 2)

            return new_points

        # Generate new points around boundary
        new_input_points = set()
        new_output_points = set()

        for input_point in boundary_inputs:
            new_input_points.update(
                generate_neighborhood(input_point, self.input_points.tolist()))

        for output_point in boundary_outputs:
            new_output_points.update(
                generate_neighborhood(output_point,
                                      self.output_points.tolist()))

        # Update grids with new points
        self.input_points = np.sort(
            np.unique(
                np.concatenate(
                    [self.input_points,
                     np.array(list(new_input_points))])))

        self.output_points = np.sort(
            np.unique(
                np.concatenate(
                    [self.output_points,
                     np.array(list(new_output_points))])))

        logger.info(
            f"Grid refinement complete - Added {len(new_input_points)} input "
            f"points and {len(new_output_points)} output points")

    def save_results(self) -> None:
        """Save profiling results to output directory.

        Writes JSON file containing:
        - Full configuration parameters
        - All test results with timestamps
        - Success/failure data for each point

        Creates:
            profile_results_{timestamp}.json in config.output_dir

        Note:
            Results include raw data for external analysis
            Timestamp ensures unique filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.output_dir / f"profile_res_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(
                {
                    "config": {
                        "model_id": self.config.model_id,
                        "gpu_ids": self.config.gpu_ids,
                        "grid_size": self.config.grid_size,
                        "refinement_rounds": self.config.refinement_rounds
                    },
                    "results": [r.to_dict() for r in self.results]
                },
                f,
                indent=2)

        logger.info(f"Results saved to {results_file}")


def profile_model(config: ProfilerConfig) -> List[ProfilingResult]:
    """Convenience function to run profiling with given configuration.

    Args:
        config: ProfilerConfig specifying profiling parameters

    Returns:
        List[ProfilingResult]: Results of profiling
    """
    profiler = TGIMemoryProfiler(config)
    return profiler.run_profiling()
