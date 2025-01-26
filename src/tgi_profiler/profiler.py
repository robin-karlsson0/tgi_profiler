"""Memory profiler for Text Generation Inference (TGI) framework.

This module implements memory profiling functionality to determine the maximum
input and output sequence lengths that can be handled by a TGI instance on a
given GPU setup without running out of memory (OOM).

The profiler uses an adaptive search strategy to find the boundary curve of
viable (input_length, output_length) combinations, providing insights into the
memory constraints of different model configurations.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from huggingface_hub import InferenceClient
from tqdm import tqdm
from transformers import AutoTokenizer

from tgi_profiler.config import MLLMConfig, ProfilerConfig
from tgi_profiler.tgi_container import ContainerError, TGIConfig, TGIContainer
from tgi_profiler.utils.colored_logging import ColoredLogger

logger = ColoredLogger(name=__name__)

INPUT_LEN_MSG_PROMPTING = 52
OUTPUT_TOLERANCE = 100


@dataclass
class ProfilingResult:
    """Result from testing a specific input/output length combination.

    Attributes:
        input_length: int
            Input sequence length tested
        output_length: int
            Output sequence length tested
        success: bool
            Whether the test passed without OOM
        error_type: Optional[str]
            Type of error if test failed
        container_logs: str
            Relevant container logs from test
        timestamp: datetime
            When the test was performed
    """
    input_length: int
    output_length: int
    success: bool
    error_type: Optional[str] = None
    container_logs: str = ""
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
            "container_logs": self.container_logs,
            "timestamp": self.timestamp.isoformat()
        }


class TGIMemoryProfiler:
    """Profiler for finding maximum sequence lengths supported by TGI setup.

    This class implements the core profiling functionality, using an adaptive
    grid search strategy to find the boundary between successful and OOM
    configurations.

    Attributes:
        config: ProfilerConfig
            Configuration controlling the profiling process
        results: List[ProfilingResult]
            Collection of all test results
        tokenizer: AutoTokenizer

    Methods:
        run_profiling()
            Execute the complete profiling process
        test_point(input_length, output_length)
            Test a specific length combination
        save_results()
            Save results to output directory
    """

    def __init__(self, config: ProfilerConfig):
        """Initialize profiler with configuration.
        
        Args:
            config: ProfilerConfig object specifying profiling parameters
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
        """Count exact number of tokens in text using model's tokenizer.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens in text
        """
        try:
            tokens = self.tokenizer.tokenize(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Token counting failed: {str(e)}")
            raise

    def _generate_exact_token_input(self, target_length: int) -> str:
        """Generate text that tokenizes to exact target length.

        Uses binary search and the model's tokenizer to generate text
        that tokenizes to exactly the target number of tokens.
        
        Args:
            client: MLLMClient instance
            target_length: Desired number of tokens
            max_iterations: Maximum adjustment iterations
            
        Returns:
            Text that tokenizes to target length
        """
        # Start with a reasonable text that's likely too long

        base_text = "What is the meaning of life? "
        text = base_text * (target_length // 8)
        current_length = self._count_tokens(text)
        print(current_length)

        # Repeat string until it is longer than required token length
        while current_length < target_length:
            print(current_length)
            text += base_text
            current_length = self._count_tokens(text)

        print('final current_length:', current_length)

        # Remove excess tokens one character at a time
        while True:
            new_length = self._count_tokens(text[:-1])
            if new_length < target_length:
                break
            else:
                text = text[:-1]

        return text

    def _generate_exact_token_output(self, target_length: int,
                                     input_txt: str) -> str:
        '''
        '''
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

        # Attempt inference and verify lengths
        response = self.client.chat_completion(messages)
        output_txt = response.choices[0].message.content

        output_len = self._count_tokens(output_txt)

        logger.info(f"Generated output length: {output_len}")

        if abs(output_len - target_length) < 100:
            return output_txt, output_len
        else:
            raise ValueError("Output length mismatch")

    def test_point(self, input_length: int,
                   output_length: int) -> ProfilingResult:
        """Test a specific input/output length combination.
        
        This method:
        1. Starts a TGI container with specified lengths
        2. Attempts inference with maximum length input
        3. Monitors for OOM or other errors
        4. Returns result with success/failure and error details
        
        Args:
            input_length: Input sequence length to test
            output_length: Output sequence length to test
            
        Returns:
            ProfilingResult for the test point
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

        for attempt in range(self.config.retries_per_point):
            try:
                with TGIContainer(tgi_config) as container:

                    # Create input that will tokenize to exact length minus
                    # approx. tokens required for message formatting
                    input_txt = self._generate_exact_token_input(
                        input_length - INPUT_LEN_MSG_PROMPTING)

                    output_txt, output_len = self._generate_exact_token_output(
                        output_length, input_txt)

                    # Verify actual token counts
                    actual_input_tokens = self._count_tokens(
                        input_txt) + INPUT_LEN_MSG_PROMPTING
                    actual_output_tokens = self._count_tokens(output_txt)

                    # Allow small tolerance for output length
                    if (actual_input_tokens != input_length
                            or actual_output_tokens
                            < output_length - OUTPUT_TOLERANCE):
                        logger.warning(
                            f"Token length mismatch - Input: expected={input_length}, "
                            f"actual={actual_input_tokens}, Output: expected={output_length}, "
                            f"actual={actual_output_tokens}")
                        return ProfilingResult(
                            input_length=input_length,
                            output_length=output_length,
                            success=False,
                            error_type="TokenLengthMismatch")

                    return ProfilingResult(input_length=actual_input_tokens,
                                           output_length=actual_output_tokens,
                                           success=True)

            except (ContainerError) as e:
                error_type = "OOM" if "out of memory" in str(
                    e).lower() else "Other"
                container_logs = ""
                if hasattr(e, "container") and e.container:
                    container_logs = e.container.logs().decode('utf-8')

                # Only retry if not OOM error
                if error_type != "OOM" and attempt < self.config.retries_per_point - 1:
                    continue

                return ProfilingResult(input_length=input_length,
                                       output_length=output_length,
                                       success=False,
                                       error_type=error_type,
                                       container_logs=container_logs)

    def _run_grid_search(self) -> None:
        """Run grid search on current points.
        
        Tests all combinations of current input and output points,
        tracking results and updating progress bar.
        """
        total_points = len(self.input_points) * len(self.output_points)

        with tqdm(total=total_points, desc=self._current_phase) as self._pbar:
            for i in self.input_points:
                for o in self.output_points:
                    result = self.test_point(i, o)
                    self.results.append(result)
                    self._pbar.update(1)

    def _refine_grid(self) -> None:
        """Refine grid points around the boundary between success/failure regions.
        
        This method:
        1. Creates a success/failure matrix from current results
        2. Identifies boundary points (success points adjacent to failures)
        3. Adds new test points around the boundary
        4. Updates input_points and output_points arrays
        
        The refinement focuses sampling around the memory limit boundary
        to better characterize the transition between working and OOM states.
        
        Raises:
            ValueError: If no results are available or if results contain invalid points
        """
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
            if result.input_length in input_dict and result.output_length in output_dict:
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
            f"Grid refined: inputs {len(new_input_points)} new points, "
            f"outputs {len(new_output_points)} new points")

    def save_results(self) -> None:
        """Save profiling results to output directory.
        
        Saves both raw results and processed boundary information
        in JSON format.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.output_dir / f"profile_results_{timestamp}.json"

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
