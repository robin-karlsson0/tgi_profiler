from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tgi_profiler.constants import VALID_API_TYPES

HF_DIR = '/home/$USER/.cache/huggingface'


@dataclass
class MLLMConfig:
    """Configuration for MLLM models.

    This class holds all configuration parameters needed for MLLM API
    interactions. Different APIs may require different subsets of these
    parameters.

    Attributes:
        api_type (str): Type of API to use (e.g., "HUGGINGFACE", "ANTHROPIC")
        max_tokens: int, default=4000
            Maximum number of tokens in the response
        temperature: float, default=0.7
            Temperature for response generation (>= 0.0)
        img_quality: int, default=90
            JPEG quality for image compression (1-100)
        model_name: Optional[str], default=None
            Name/identifier of the model to use
        api_key: Optional[str], default=None
            API key for authentication (required for some APIs)
        base_url: Optional[str], default=None
            Base URL for API endpoint (required for some APIs)
    """
    api_type: str = 'HUGGINGFACE'
    max_tokens: int = 4000
    temperature: float = 0.7
    img_quality: int = 90
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.api_type.upper() not in VALID_API_TYPES:
            raise ValueError(f"Invalid API type: {self.api_type}. "
                             f"Must be one of {VALID_API_TYPES}")
        self.api_type = self.api_type.upper()


@dataclass
class ProfilerConfig:
    """Configuration for memory profiling parameters.

    This class defines the parameters that control the profiling process,
    including search ranges, grid resolution, and test methodology.

    Attributes:
        min_input_length: int
            Minimum input sequence length to test
        max_input_length: int
            Maximum input sequence length to test
        min_output_length: int
            Minimum output sequence length to test
        max_output_length: int
            Maximum output sequence length to test
        grid_size: int
            Initial grid resolution for search
        refinement_rounds: int
            Number of times to refine the grid around boundary
        retries_per_point: int
            Number of retry attempts per test point
        output_dir: Path
            Directory for saving results and logs
        model_id: str
            HuggingFace model ID to profile
        gpu_ids: List[int]
            List of GPU IDs to use for container
        port: int
            Port for TGI container
        hf_token: Optional[str]
            HuggingFace token for accessing gated models
    """
    min_input_length: int = 128
    max_input_length: int = 8192
    min_output_length: int = 128
    max_output_length: int = 4096
    grid_size: int = 8
    port: int = 8080
    refinement_rounds: int = 2
    retries_per_point: int = 3
    output_dir: Path = Path("profiler_results")
    model_id: str = ""
    gpu_ids: List[int] = None
    hf_token: Optional[str] = None
    hf_cache_dir: Optional[str] = HF_DIR
    # Inference client configuration
    base_url: Optional[str] = 'http://localhost:8080/v1'

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0]
        self.output_dir.mkdir(parents=True, exist_ok=True)
