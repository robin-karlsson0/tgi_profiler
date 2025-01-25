import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from docker.errors import DockerException
from docker.models.containers import Container

from tgi_profiler.server.tgi_container import TGIConfig

MODEL = 'microsoft/Phi-3-mini-4k-instruct'  # "meta-llama/Meta-Llama-3-8B-Instruct"
HF_DIR = '/home/$USER/.cache/huggingface'
HF_TOKEN = os.environ.get('HF_TOKEN')


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring docker and network access"
    )


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
