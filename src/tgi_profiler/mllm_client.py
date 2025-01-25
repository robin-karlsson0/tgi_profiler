"""Multimodal Language Model (MLLM) client library for various API providers.

This module provides a unified interface for interacting with different
multimodal language model APIs (HuggingFace, Anthropic, OpenAI). It handles
message formatting, image processing, and error handling consistently across
different providers.

Example usage:
    ```python
    config = MLLMConfig(
        api_type='ANTHROPIC',
        model_name="claude-3-sonnet",
        api_key="your-api-key"
    )

    client = MLLMClient(config=config)

    # Text-only query
    messages = [
        TextMessage("You are a helpful assistant.", role="system"),
        TextMessage("What is multimodal AI?")
    ]
    response = await client.generate(messages)

    # Multimodal query
    image = Image.open("example.jpg")
    messages = [
        TextMessage("Analyze this image.", role="system"),
        MultimodalMessage("Describe what you see.", images=[image])
    ]
    response = await client.generate(messages)
    ```

Typical usage involves:
1. Creating a configuration
2. Initializing the client with desired API type
3. Preparing messages (text-only or multimodal)
4. Generating responses

The module handles:
- Message formatting for different APIs
- Image processing and conversion
- Error handling and validation
- Response parsing and standardization
"""

import abc
import base64
import io
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic
import openai
from huggingface_hub import InferenceClient
from PIL import Image

from tgi_profiler.config import MLLMConfig
from tgi_profiler.utils.colored_logging import ColoredLogger

logger = ColoredLogger(name=__name__)


class MLLMError(Exception):
    """Base exception class for MLLM-related errors.

    This class serves as the base for all MLLM-specific exceptions. It adds
    support for including additional error details beyond the basic message.

    Attributes:
        message: str
            Human-readable error description
        details: dict
            Additional error details and context

    Example:
        ```python
        raise MLLMError(
            "Failed to process request",
            {"status_code": 500, "api_response": "..."}
        )
        ```"""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(MLLMError):
    """Raised when there are issues with MLLM configuration.

    This exception indicates that there is a problem with the provided
    configuration parameters, such as missing required values or
    invalid settings.

    Example:
        ```python
        raise ConfigurationError(
            "Missing required API key",
            {"config_parameter": "api_key"}
        )
        ```
    """
    pass


class APIError(MLLMError):
    """Raised when there are API-related errors.

    This exception indicates that an error occurred during API communication,
    such as network issues, authentication failures, or invalid responses.

    Attributes:
        message: str
            Error description
        status_code: Optional[int]
            HTTP status code if applicable
        details: dict
            Additional error context

    Example:
        ```python
        raise APIError(
            "API request failed",
            status_code=429,
            retry_after=60
        )
        ```
    """

    def __init__(self,
                 message: str,
                 status_code: Optional[int] = None,
                 **kwargs):
        super().__init__(message, {"status_code": status_code, **kwargs})


class ValidationError(MLLMError):
    """Raised when there are validation errors with inputs.

    This exception indicates that the provided inputs (messages, images,
    configuration values, etc.) fail validation checks.

    Example:
        ```python
        raise ValidationError(
            "Invalid image format",
            {"format": "gif", "allowed_formats": ["jpg", "png"]}
        )
        ```
    """
    pass


class ImageProcessingError(MLLMError):
    """Raised when there are errors processing images.

    This exception indicates that an error occurred during image processing,
    such as format conversion, resizing, or encoding.

    Example:
        ```python
        raise ImageProcessingError(
            "Failed to encode image",
            {"format": "JPEG", "error": "Invalid JPEG marker"}
        )
        ```
    """
    pass


class MessageError(MLLMError):
    """Raised when there are errors with message formatting or content.

    This exception indicates problems with message construction, such as
    invalid roles, empty content, or format incompatibilities.

    Example:
        ```python
        raise MessageError(
            "Invalid message role",
            {"role": "invalid", "allowed_roles": ["user", "system"]}
        )
        ```
    """
    pass


def get_api_key(env_var: str) -> str:
    """Get API key from environment variable.

    Args:
        env_var: Name of environment variable containing the API key

    Returns:
        str: API key

    Raises:
        ConfigurationError: If API key is not found in environment
    """
    api_key = os.getenv(env_var)
    if not api_key:
        raise ConfigurationError(
            f"API key not found in environment variable: {env_var}")
    return api_key


@dataclass
class MLLMResponse:
    """Response from an MLLM inference.

    This class provides a standardized format for responses across different
    MLLM APIs, including both the generated content and metadata about the
    generation process.

    Attributes:
        content: str
            The actual response content from the model
        tokens_used: int
            Number of tokens used in generating the response
        model_id: str
            Identifier of the model that generated the response

    Notes:
        Token counts may be calculated differently by different APIs.
        The model_id format will vary between providers.
    """
    content: str
    tokens_used: int
    model_id: str


class Message:
    """Base class for all message types.

    This class represents a message that can be sent to an MLLM. It provides
    the basic structure that both text-only and multimodal messages build upon.

    Attributes:
        role: str
            Role of the message sender (e.g., "user", "system", "assistant")
        content: Any
            Content of the message (type depends on message subclass)

    Notes:
        This is an abstract base class. Use TextMessage or MultimodalMessage
        for actual messages.
    """

    def __init__(self, content: Any, role: str = "user"):
        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to API-specific format."""
        return {"role": self.role, "content": self.content}


class TextMessage(Message):
    """Text-only message."""

    def __init__(self, text: str, role: str = "user"):
        super().__init__(text, role)


class MultimodalMessage(Message):
    """Message containing both text and images.

    This class represents a message that includes both textual content and
    one or more images. The images are automatically processed and formatted
    according to the requirements of the target API.

    Attributes:
        text: str
            Textual content of the message
        images: List[Image.Image]
            List of PIL Image objects to include
        role: str, default="user"
            Role of the message sender

    Example:
        ```python
        image = Image.open("example.jpg")
        msg = MultimodalMessage(
            text="What's in this image?",
            images=[image]
        )
        ```
    """

    def __init__(self,
                 text: str,
                 images: List[Image.Image],
                 role: str = "user"):
        content = {"text": text, "images": images}
        super().__init__(content, role)


class BaseMLLMAPI(abc.ABC):
    """Abstract base class for MLLM API implementations.

    This class defines the interface that all MLLM API implementations must
    follow. It provides common functionality for configuration validation,
    error handling, and response processing while allowing specific APIs to
    implement their own message handling logic.

    The class uses the Template Method pattern, where generate() defines the
    skeleton of the algorithm and _generate_impl() provides the API-specific
    implementation.

    Attributes:
        config: MLLMConfig
            Configuration for the API instance

    Notes:
        Subclasses must implement _generate_impl() to handle actual API calls.
        They should also validate their specific configuration requirements in
        their __init__() method.
    """

    def __init__(self, config: MLLMConfig):
        self._validate_config(config)
        self.config = config

    async def generate(
            self,
            messages: List[Message],
            config: Optional[Dict[str, Any]] = None) -> MLLMResponse:
        """Generate a response from the model.

        This method implements the template pattern for generating responses:
        1. Validates inputs
        2. Merges configurations
        3. Calls API-specific implementation
        4. Validates and processes response

        Args:
            messages: List of messages to send to the model
            config: Optional configuration overrides for this request

        Returns:
            MLLMResponse containing the model's response and metadata

        Raises:
            ValidationError: If inputs are invalid
            APIError: If API call fails
            MLLMError: For other types of errors
        """
        try:
            # Pre-processing
            self._validate_messages(messages)
            merged_config = self._merge_configs(self.config, config)

            # Core implementation (handled by subclasses)
            logger.debug(f"Generating response with {len(messages)} messages")
            response = await self._generate_impl(messages, merged_config)

            # Post-processing
            self._validate_response(response)
            logger.info(
                f"Generated response with {response.tokens_used} tokens")

            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            if isinstance(e, MLLMError):
                raise
            raise APIError(f"API call failed: {str(e)}")

    @abc.abstractmethod
    async def _generate_impl(self, messages: List[Message],
                             config: MLLMConfig) -> MLLMResponse:
        """Implementation of the actual API call.

        This must be implemented by each API subclass to handle the
        specific details of calling their respective APIs.

        Args:
            messages: Validated list of messages
            config: Merged configuration

        Returns:
            MLLMResponse: Model's response
        """
        pass

    def _merge_configs(
            self,
            base_config: MLLMConfig,
            override_config: Optional[Dict[str, Any]] = None) -> MLLMConfig:
        """Merge base config with request-specific overrides.

        Args:
            base_config: Base configuration
            override_config: Optional overrides

        Returns:
            MLLMConfig: Merged configuration
        """
        if not override_config:
            return base_config

        # Create new config with overrides
        config_dict = base_config.__dict__.copy()
        config_dict.update(override_config)
        return MLLMConfig(**config_dict)

    def _validate_response(self, response: MLLMResponse) -> None:
        """Validate the response from the API.

        Args:
            response: Response to validate

        Raises:
            ValidationError: If response is invalid
        """
        if not response.content:
            raise ValidationError("Response content cannot be empty")

        if response.tokens_used < 0:
            raise ValidationError("tokens_used cannot be negative",
                                  {"tokens_used": response.tokens_used})

    @staticmethod
    def _image_to_data_url(img: Image.Image, quality: int = 90) -> str:
        """Convert PIL Image to data URL.

        Args:
            img: PIL Image to convert
            quality: JPEG quality (1-100)

        Returns:
            Data URL string

        Raises:
            ImageProcessingError: If image processing fails
        """
        try:
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                # Paste the image using alpha channel as mask
                background.paste(img,
                                 mask=img.split()[3])  # 3 is the alpha channel
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            with io.BytesIO() as buffer:
                img.save(buffer, format='JPEG', quality=quality)
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            raise ImageProcessingError("Failed to convert image to data URL",
                                       {"error": str(e)})

    def _validate_config(self, config: MLLMConfig) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        if config.max_tokens < 1:
            raise ConfigurationError("max_tokens must be positive",
                                     {"max_tokens": config.max_tokens})

        if not 0 <= config.temperature:
            raise ConfigurationError("temperature must be >= 0",
                                     {"temperature": config.temperature})

        if not 1 <= config.img_quality <= 100:
            raise ConfigurationError("img_quality must be between 1 and 100",
                                     {"img_quality": config.img_quality})

    def _validate_messages(self, messages: List[Message]) -> None:
        """Validate message list.

        Args:
            messages: List of messages to validate

        Raises:
            ValidationError: If messages are invalid
        """
        if not messages:
            raise ValidationError("messages list cannot be empty")

        if not all(isinstance(msg, Message) for msg in messages):
            raise ValidationError("all items must be Message instances")

        if any(not msg.role for msg in messages):
            raise ValidationError("all messages must have a role")

        if any(not msg.content for msg in messages):
            raise ValidationError("all messages must have content")


class HuggingFaceAPI(BaseMLLMAPI):
    """
    HuggingFace Text Generation Inference (TGI) API implementation.

    This class implements the MLLM API interface for HuggingFace's TGI service.
    It handles connection to a TGI endpoint and formats messages according to
    the TGI API requirements.

    Attributes:
        client: InferenceClient
            HuggingFace inference client instance
        config: MLLMConfig
            Configuration for the API

    Notes:
        Requires a base_url pointing to a TGI endpoint.
        Image processing follows TGI's specific requirements.
    """

    def __init__(self, config: MLLMConfig):
        super().__init__(config)
        if not config.base_url:
            raise ConfigurationError(
                "base_url is required for HuggingFace API")
        self.client = InferenceClient(base_url=config.base_url)

    async def _generate_impl(self, messages: List[Message],
                             config: MLLMConfig) -> MLLMResponse:
        """Implementation specific to HuggingFace's API.

        Args:
            messages: Pre-validated list of messages
            config: Merged configuration

        Returns:
            MLLMResponse: Model's response

        Raises:
            APIError: If the API call fails
        """
        try:
            # Format messages for HuggingFace API
            formatted_messages = []
            for msg in messages:
                if isinstance(msg.content, dict) and "images" in msg.content:
                    # Handle multimodal message
                    content = [{"type": "text", "text": msg.content["text"]}]
                    for img in msg.content["images"]:
                        img_data_url = self._image_to_data_url(
                            img, config.img_quality)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": img_data_url
                            }
                        })
                    formatted_messages.append({
                        "role": msg.role,
                        "content": content
                    })
                else:
                    # Handle text-only message
                    formatted_messages.append(msg.to_dict())

            # Make API call
            output = self.client.chat_completion(
                model=config.model_name,
                messages=formatted_messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )

            # Convert response
            return MLLMResponse(content=output.choices[0].message.content,
                                tokens_used=output.usage.total_tokens,
                                model_id=config.model_name)

        except Exception as e:
            raise APIError(f"HuggingFace API call failed: {str(e)}",
                           details={"original_error": str(e)})


class AnthropicAPI(BaseMLLMAPI):
    """Anthropic Claude API implementation.

    This class implements the MLLM API interface for Anthropic's Claude models.
    It handles the specific requirements of the Anthropic API, including
    message formatting and image encoding.

    Attributes:
        client: anthropic.Client
            Anthropic API client instance
        config: MLLMConfig
            Configuration for the API

    Notes:
        Requires an Anthropic API key to be provided in the configuration.
        Supports both text-only and multimodal interactions.
    """

    def __init__(self, config: MLLMConfig):
        """Initialize Anthropic API client.

        Args:
            config: Configuration for the API

        Raises:
            ConfigurationError: If required configuration is missing
        """
        super().__init__(config)
        if not config.api_key:
            raise ConfigurationError("api_key is required for Anthropic API")
        self.client = anthropic.Client(api_key=config.api_key)

    async def _generate_impl(self, messages: List[Message],
                             config: MLLMConfig) -> MLLMResponse:
        """Implementation specific to Anthropic's API.

        Args:
            messages: Pre-validated list of messages
            config: Merged configuration

        Returns:
            MLLMResponse: Model's response

        Raises:
            APIError: If the API call fails
        """
        try:
            # Format messages for Anthropic API
            system_message = None
            formatted_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                    continue

                if isinstance(msg.content, dict) and "images" in msg.content:
                    # Handle multimodal message
                    content = [{"type": "text", "text": msg.content["text"]}]
                    for img in msg.content["images"]:
                        img_data = self._image_to_data_url(
                            img, config.img_quality).split(",")[
                                1]  # Remove data URL prefix
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_data
                            }
                        })
                    formatted_messages.append({
                        "role": msg.role,
                        "content": content
                    })
                else:
                    # Handle text-only message
                    formatted_messages.append(msg.to_dict())

            # Prepare API call parameters
            api_params = {
                "model": config.model_name,
                "max_tokens": config.max_tokens,
                "messages": formatted_messages,
                "temperature": config.temperature
            }

            # Add system message if present
            if system_message:
                api_params["system"] = system_message

            # Make API call
            response = self.client.messages.create(**api_params)

            # Convert response
            return MLLMResponse(content=response.content[0].text,
                                tokens_used=response.usage.output_tokens,
                                model_id=config.model_name)

        except Exception as e:
            raise APIError(f"Anthropic API call failed: {str(e)}",
                           details={"original_error": str(e)})


class OpenAIAPI(BaseMLLMAPI):
    """OpenAI GPT-4V API implementation.

    This class implements the MLLM API interface for OpenAI's GPT-4V model.
    It handles the specific requirements of the OpenAI API, including
    authentication and message formatting.

    Attributes:
        client: openai.Client
            OpenAI API client instance
        config: MLLMConfig
            Configuration for the API

    Notes:
        Requires an OpenAI API key.
        Currently supports the GPT-4V model for multimodal interactions.
    """

    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4-vision-preview",
                 max_tokens: int = 4000,
                 img_quality: int = 90):
        self.client = openai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.img_quality = img_quality

    async def generate(
            self,
            system_msg: str,
            user_msg: str,
            images: Optional[List[Image.Image]] = None) -> MLLMResponse:
        """Generate response using OpenAI GPT-4V."""
        content = []
        content.append({"type": "text", "text": user_msg})

        if images:
            for img in images:
                img_data = self._image_to_data_url(img, self.img_quality)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_data
                    }
                })

        messages = [{
            "role": "system",
            "content": system_msg
        }, {
            "role": "user",
            "content": content
        }]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.max_tokens)

        return MLLMResponse(content=response.choices[0].message.content,
                            tokens_used=response.usage.total_tokens,
                            model_id=self.model)


class MLLMClient:
    """Client for interacting with various multimodal language models.

    This class provides a unified interface for working with different MLLM
    APIs. It handles the details of API-specific message formatting, image
    processing, and error handling while presenting a consistent interface to
    the user.

    Attributes:
        api: BaseMLLMAPI
            The specific API implementation being used

    Example:
        ```python
        config = MLLMConfig(
            api_type='ANTHROPIC',
            model_name="claude-3-sonnet",
            api_key="your-api-key"
        )
        client = MLLMClient(config)
        response = await client.generate([
            TextMessage("Analyze this image"),
            MultimodalMessage("What do you see?", images=[image])
        ])
        print(response.content)
    """

    _API_REGISTRY = {
        'HUGGINGFACE': HuggingFaceAPI,
        'ANTHROPIC': AnthropicAPI,
        'OPENAI': OpenAIAPI
    }

    def __init__(self, config: MLLMConfig):
        """Initialize the MLLM client."""
        if config.api_type not in self._API_REGISTRY:
            raise ConfigurationError(
                f"Unsupported API type: {config.api_type}. "
                f"Valid types are: {', '.join(self._API_REGISTRY.keys())}")

        api_cls = self._API_REGISTRY[config.api_type]
        self.api = api_cls(config=config)
        self.config = config

        logger.info(f"Initialized MLLM client with {config.api_type} API")

    async def generate(
            self,
            messages: List[Message],
            config_override: Optional[Dict[str, Any]] = None) -> MLLMResponse:
        """Generate response using configured MLLM.

        Args:
            messages: List of messages to send
            config_override: Optional configuration overrides

        Returns:
            MLLMResponse containing model's response

        Raises:
            APIError: If API call fails
            ValidationError: If inputs are invalid
        """
        return await self.api.generate(messages, config_override)
