import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import docker
import requests
from docker.errors import DockerException


@dataclass
class TGIConfig:
    """Configuration for TGI container"""
    model_id: str
    gpu_ids: List[int]
    port: int
    max_input_length: int
    max_output_length: int
    hf_token: Optional[str] = None
    hf_cache_dir: Optional[Path] = None
    tgi_image: str = "ghcr.io/huggingface/text-generation-inference:3.0.1"
    health_check_retries: int = 5
    health_check_interval: float = 2.0


class ContainerError(Exception):
    """Base exception for container operations"""
    pass


class TGIContainer:
    """Manages TGI Docker container lifecycle and configuration"""

    def __init__(self, config: TGIConfig):
        self.config = config
        self._container = None
        self._is_running = False
        self._client = docker.from_env()

    @property
    def is_running(self) -> bool:
        if not self._container:
            return False
        try:
            self._container.reload()
            return self._container.status == "running"
        except DockerException:
            return False

    def start(self) -> None:
        """Start TGI container with configured parameters"""
        if self.is_running:
            return

        environment = {
            "HF_TOKEN": self.config.hf_token if self.config.hf_token else ""
        }

        volumes = {}
        if self.config.hf_cache_dir:
            volumes[str(self.config.hf_cache_dir)] = {
                "bind": "/data",
                "mode": "rw"
            }

        command = [
            "--model-id", self.config.model_id, "--max-input-length",
            str(self.config.max_input_length), "--max-total-tokens",
            str(self._calculate_total_tokens()), "--max-batch-prefill-tokens",
            str(self._calculate_prefill_tokens())
        ]

        try:
            self._container = self._client.containers.run(
                self.config.tgi_image,
                command=command,
                environment=environment,
                volumes=volumes,
                device_requests=[self._format_gpu_request()],
                ports={80: self.config.port},
                shm_size="1g",
                detach=True)

            if not self._wait_for_healthy():
                raise ContainerError("Container failed health check")

            self._is_running = True

        except DockerException as e:
            raise ContainerError(f"Failed to start container: {str(e)}")

    def stop(self) -> None:
        """Stop running container"""
        if self._container and self.is_running:
            try:
                self._container.stop(timeout=10)
                self._container.remove()
            except DockerException as e:
                raise ContainerError(f"Failed to stop container: {str(e)}")
            finally:
                self._container = None
                self._is_running = False

    def check_health(self) -> bool:
        """Check if container is healthy and responding"""
        if not self.is_running:
            return False

        try:
            response = requests.get(
                f"http://localhost:{self.config.port}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _format_gpu_request(self) -> dict:
        """Format GPU device request for docker"""
        return {
            "Driver": "nvidia",
            "Count": len(self.config.gpu_ids),
            "Capabilities": [["gpu"]],
            "DeviceIDs": [str(id) for id in self.config.gpu_ids]
        }

    def _calculate_total_tokens(self) -> int:
        """Calculate max total tokens based on input/output lengths"""
        return self.config.max_input_length + self.config.max_output_length

    def _calculate_prefill_tokens(self) -> int:
        """Calculate max batch prefill tokens"""
        return self.config.max_input_length + 50

    def _wait_for_healthy(self) -> bool:
        """Wait for container to become healthy"""
        for _ in range(self.config.health_check_retries):
            if self.check_health():
                return True
            time.sleep(self.config.health_check_interval)
        return False

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
