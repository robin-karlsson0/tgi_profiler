from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tgi_profiler.server.tgi_container import TGIConfig, TGIContainer


@pytest.fixture
def mock_docker():
    """Fixture providing a mocked Docker client"""
    with patch('tgi_profiler.server.tgi_container.docker') as mock_docker:
        mock_docker.from_env.return_value = MagicMock()
        yield mock_docker


def test_container_initialization(single_gpu_config, mock_docker):
    """Test basic initialization of TGIContainer"""
    container = TGIContainer(single_gpu_config)

    # Verify container state after initialization
    assert container.config == single_gpu_config
    assert container._container is None
    assert not container._is_running

    # Verify Docker client was initialized
    mock_docker.from_env.assert_called_once()


def test_token_calculations(single_gpu_config):
    """Test token calculation methods"""
    container = TGIContainer(single_gpu_config)
    # Test total tokens calculation
    total_tokens = container._calculate_total_tokens()
    assert total_tokens == single_gpu_config.max_input_length + single_gpu_config.max_output_length
    # Test prefill tokens calculation
    prefill_tokens = container._calculate_prefill_tokens()
    assert prefill_tokens == single_gpu_config.max_input_length + 50
