import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from docker.errors import DockerException
from docker.models.containers import Container

from tgi_profiler.mllm_client import MLLMConfig
from tgi_profiler.tgi_container import TGIConfig

MODEL = 'microsoft/Phi-3-mini-4k-instruct'  # "meta-llama/Meta-Llama-3-8B-Instruct"
HF_DIR = '/home/$USER/.cache/huggingface'
HF_TOKEN = os.environ.get('HF_TOKEN')


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring docker and network access"
    )


###################
#  TGI Container  #
###################


@pytest.fixture
def mock_docker_client():
    with patch('docker.from_env') as mock_client:
        container_mock = Mock(spec=Container)
        container_mock.status = "running"
        container_mock.reload = Mock()
        container_mock.stop = Mock()
        container_mock.remove = Mock()

        mock_client.return_value.containers.run.return_value = container_mock
        yield mock_client.return_value


@pytest.fixture
def mock_requests():
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        yield mock_get


@pytest.fixture
def single_gpu_config():
    return TGIConfig(model_id=MODEL,
                     gpu_ids=[0],
                     port=8080,
                     max_input_length=1000,
                     max_output_length=100,
                     hf_token=HF_TOKEN,
                     hf_cache_dir=Path(HF_DIR),
                     health_check_retries=3,
                     health_check_interval=20.0)


@pytest.fixture
def multi_gpu_config():
    return TGIConfig(model_id=MODEL,
                     gpu_ids=[0, 1, 2, 3],
                     port=8080,
                     max_input_length=2000,
                     max_output_length=200,
                     hf_token=HF_TOKEN,
                     hf_cache_dir=Path(HF_DIR))


@pytest.fixture
def failed_container(mock_docker_client):
    mock_docker_client.containers.run.side_effect = DockerException(
        "Failed to start")
    return mock_docker_client


@pytest.fixture
def unhealthy_container(mock_requests):
    mock_requests.return_value.status_code = 500
    return mock_requests


#################
#  MLLM Client  #
#################


@pytest.fixture
def mock_inference_response():
    """Fixture providing a standard mock response structure for TGI VLM
    inference.

    The mock structure matches exactly how the response is accessed in the
    code:
    - output.choices[0].message.content
    - output.usage.total_tokens
    """
    # Create Mock objects for nested structure
    message = Mock()
    message.content = "This is a test response"

    choice = Mock()
    choice.message = message

    usage = Mock()
    usage.total_tokens = 51

    # Create the main response object with properly configured choices list
    response = Mock()
    response.choices = [choice]
    response.usage = usage
    response.model = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    response.created = 1737511143
    response.id = ''
    response.system_fingerprint = '3.0.1-sha-bb9095a'

    return response


@pytest.fixture
def mock_inference_client():
    """Fixture providing a mocked version of the HuggingFace InferenceClient.

    This fixture patches the InferenceClient used within the MLLM client's
    HuggingFaceAPI implementation. It provides a mock that can be configured
    to return specific responses for testing different scenarios.

    Returns:
        Mock: A configured mock object that simulates the InferenceClient's
            behavior
    """
    with patch('tgi_profiler.mllm_client.InferenceClient') as mock_client:
        client_instance = Mock()
        client_instance.chat_completion = Mock()
        mock_client.return_value = client_instance
        yield client_instance


@pytest.fixture
def mllm_config_hf():
    """Fixture providing a basic HuggingFace API configuration."""
    return MLLMConfig(api_type='HUGGINGFACE',
                      model_name='meta-llama/Llama-3.2-90B-Vision-Instruct',
                      base_url='http://localhost:8080/v1',
                      max_tokens=400,
                      temperature=0.7,
                      img_quality=90)
