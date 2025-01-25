import os
import time
from pathlib import Path

import docker
import pytest
from huggingface_hub import InferenceClient

from tgi_profiler.server.tgi_container import TGIConfig, TGIContainer


@pytest.mark.integration
def test_container_lifecycle(single_gpu_config):
    """Integration test for full container lifecycle including inference"""

    # Skip if no GPU available
    #    try:
    #        docker_client = docker.from_env()
    #        docker_client.info()
    #    except docker.errors.DockerException:
    #        pytest.skip("Docker not available")

    #    # Skip if no HF token
    #    hf_token = os.environ.get('HF_TOKEN')
    #    if not hf_token:
    #        pytest.skip("HF_TOKEN environment variable not set")

    # Test container lifecycle
    with TGIContainer(single_gpu_config) as container:
        # Verify container is running
        assert container.is_running
        assert container.check_health()

        # Test inference
        client = InferenceClient("http://localhost:8080")
        response = client.text_generation("What is the capital of France?",
                                          max_new_tokens=20)
        assert response and len(response) > 0

    # Verify container is stopped and removed
    docker_client = docker.from_env()
    all_containers = docker_client.containers.list(all=True)

    # Check no containers with our image are running
    matching_containers = [
        c for c in all_containers
        if single_gpu_config.tgi_image in c.image.tags
    ]
    assert len(matching_containers) == 0
