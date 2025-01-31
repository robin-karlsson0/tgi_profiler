"""Unit tests for MLLMClient functionality.

This module tests the MLLMClient class with a focus on the HuggingFace TGI
implementation. It covers both text-only and multimodal message handling.
"""

import warnings

import pytest

from tgi_profiler.mllm_client import MLLMClient, TextMessage

# Filter out deprecation warnings from Pydantic
warnings.filterwarnings("ignore",
                        category=DeprecationWarning,
                        module="pydantic")


@pytest.mark.asyncio
async def test_text_only_generation(mock_inference_client,
                                    mock_inference_response, mllm_config_hf):
    """Test text-only message generation with HuggingFace TGI API.

    This test verifies that:
    1. Text messages are formatted correctly
    2. API calls are made with correct parameters
    3. Responses are properly processed and returned
    """
    # Arrange
    client = MLLMClient(config=mllm_config_hf)
    messages = [
        TextMessage("You are a helpful assistant.", role="system"),
        TextMessage("This is a test message", role="user")
    ]
    mock_inference_client.chat_completion.return_value = mock_inference_response  # noqa

    # Act
    mllm_response = await client.generate(messages)

    # Assert
    expected_response = "This is a test response"
    assert mllm_response.content == expected_response
    assert mllm_response.tokens_used == 51
    assert mllm_response.model_id == mllm_config_hf.model_name

    # Verify API call parameters
    mock_inference_client.chat_completion.assert_called_once_with(
        model=mllm_config_hf.model_name,
        messages=[msg.to_dict() for msg in messages],
        max_tokens=mllm_config_hf.max_tokens,
        temperature=mllm_config_hf.temperature,
    )
