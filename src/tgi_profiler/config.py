import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tgi_profiler.boundary_detection import BoundaryConfig
from tgi_profiler.constants import VALID_API_TYPES

USER = os.environ.get('USER')
HF_DIR = os.environ.get('HF_DIR')
HF_TOKEN = os.environ.get('HF_TOKEN')


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

    Controls profiling process parameters including sequence lengths, 
    search behavior, hardware config, and model settings.

    Attributes:
        min_input_length: Minimum input sequence length (default: 128)
        max_input_length: Maximum input sequence length (default: 8192)
        min_output_length: Minimum output sequence length (default: 128)
        max_output_length: Maximum output sequence length (default: 4096)
        grid_size: Initial grid resolution (default: 8)
        port: TGI container port (default: 8080)
        refinement_rounds: Grid refinement iterations (default: 2)
        retries_per_point: Test point retry attempts (default: 8)
        output_dir: Results directory (default: "profiler_results")
        model_id: HuggingFace model ID
        gpu_ids: GPU devices to use (default: [0])
        hf_token: HuggingFace access token (optional)
        hf_cache_dir: Model cache directory (optional)
        base_url: Inference API endpoint (default: "http://localhost:8080/v1")
        output_tolerance_pct: Tolerance [%] for output length variation
            (default: 5.0)
        temp: Temperature for response generation high enough to avoid
            predicting EOS token (default: 1.5
        resume_from_file: Optional path to resume from a previous run by
            specifying the path to the results JSON file
        multimodal: Enable multimodal testing with dummy image (default: False)
        dummy_image_path: Path to dummy image for multimodal testing (required
            if multimodal=True)

        # Boundary detection parameters
        k_neighbors: Number of nearest neighbors for local boundary detection
            (default: 5)
        m_random: Number of random samples for global exploration (default: 3)
        distance_scale: Scale factor for distance-based scoring
            (default: 1000)
        consistency_radius: Maximum distance to consider points for
            consistency (default: 1000)
        redundancy_weight: Weight factor for penalizing redundant pairs
            (default: 0.5)
        max_pair_distance: Maximum allowed distance between boundary points
            (default: 2000)
        min_refinement_dist: Minimum distance between points for further
            refinement (default: 50)

    Notes:
        Creates output directory if it doesn't exist
        Defaults gpu_ids to [0] if not specified
    """
    # Basic configuration
    min_input_length: int = 128
    max_input_length: int = 8192
    min_output_length: int = 128
    max_output_length: int = 4096
    port: int = 8080
    output_dir: Path = Path("profiler_results")
    model_id: str = ""
    gpu_ids: List[int] = None
    hf_token: Optional[str] = HF_TOKEN
    hf_cache_dir: Optional[str] = Path(HF_DIR)
    base_url: Optional[str] = 'http://localhost:8080/v1'
    output_tolerance_pct: float = 0.05
    temp = 1.5
    resume_from_file: Optional[str] = None

    # Multimodal configuration
    multimodal: bool = False
    dummy_image_path: Optional[Path] = None

    # Refinement parameters
    refinement_rounds: int = 2
    retries_per_point: int = 8

    # Boundary detection parameters
    k_neighbors: int = 5
    m_random: int = 3
    distance_scale: float = 1000
    consistency_radius: float = 1000
    redundancy_weight: float = 0.5
    max_pair_distance: float = 1e12
    min_refinement_dist: int = 50
    grid_size: int = 8

    def __post_init__(self):
        if self.gpu_ids is None:
            self.gpu_ids = [0]
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert resume_from_file to Path if provided
        if self.resume_from_file:
            self.resume_from_file = Path(self.resume_from_file)

        # Validate multimodal configuration
        if self.multimodal:
            if not self.dummy_image_path:
                raise ValueError(
                    "dummy_image_path must be provided when multimodal=True")
            self.dummy_image_path = Path(self.dummy_image_path)
            if not self.dummy_image_path.exists():
                raise FileNotFoundError(
                    f"Dummy image not found at: {self.dummy_image_path}")

    def create_boundary_config(self) -> BoundaryConfig:
        """Create a BoundaryConfig instance from profiler settings."""
        return BoundaryConfig(k_neighbors=self.k_neighbors,
                              m_random=self.m_random,
                              distance_scale=self.distance_scale,
                              consistency_radius=self.consistency_radius,
                              redundancy_weight=self.redundancy_weight,
                              grid_size=self.grid_size,
                              max_pair_distance=self.max_pair_distance)
