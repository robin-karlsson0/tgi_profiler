from dataclasses import dataclass
from typing import Optional

from tgi_profiler.constants import VALID_API_TYPES


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
