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
            "HF_TOKEN":
            self.config.hf_token if self.config.hf_token else "",
            "HUGGING_FACE_HUB_TOKEN":
            self.config.hf_token if self.config.hf_token else "",
        }

        volumes = {}
        if self.config.hf_cache_dir:
            volumes[str(self.config.hf_cache_dir)] = {
                "bind": "/data",
                "mode": "rw"
            }

        command = [
            "--model-id",
            self.config.model_id,
            "--max-input-length",
            str(self.config.max_input_length),
            "--max-total-tokens",
            str(self._calculate_total_tokens()),
            "--max-batch-prefill-tokens",
            str(self._calculate_prefill_tokens()),
            # "--json-output",
        ]

        ports = {80: self.config.port}
        device_requests = [self._format_gpu_request()]

        # Log configuration before starting container
        self._log_container_config(command=command,
                                   environment=environment,
                                   volumes=volumes,
                                   device_requests=device_requests,
                                   ports=ports)

        try:
            self._container = self._client.containers.run(
                self.config.tgi_image,
                command=command,
                environment=environment,
                volumes=volumes,
                device_requests=device_requests,
                ports=ports,
                shm_size="1g",
                detach=True)

            print(f"Started container {self._container.id}")

            if not self._wait_for_healthy():
                # Get final logs before failing
                print("Final container logs:")
                print(self._container.logs().decode('utf-8'))
                raise ContainerError("Container failed health check")

            self._is_running = True

        except Exception as e:
            if self._container:
                print("Container logs before error:")
                print(self._container.logs().decode('utf-8'))
            raise ContainerError(
                f"Unexpected error starting container: {str(e)}")

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
            print("Container not running during health check")
            return False

        try:
            url = f"http://localhost:{self.config.port}/health"
            print(f"Trying health check at: {url}")
            response = requests.get(url, timeout=5)
            success = response.status_code == 200
            if success:
                print("Health check successful")
            else:
                print(
                    f"Health check failed with status {response.status_code}: {response.text}"
                )
            return success
        except requests.RequestException as e:
            print(f"Health check failed: {str(e)}")
            return False

    def _format_gpu_request(self) -> dict:
        """Format GPU device request for docker"""
        return {
            "Driver": "nvidia",
            "DeviceIDs": [str(id) for id in self.config.gpu_ids],
            "Capabilities": [["gpu"]]
        }

    def _calculate_total_tokens(self) -> int:
        """Calculate max total tokens based on input/output lengths"""
        return self.config.max_input_length + self.config.max_output_length

    def _calculate_prefill_tokens(self) -> int:
        """Calculate max batch prefill tokens"""
        return self.config.max_input_length + 50

    def _wait_for_healthy(self) -> bool:
        """Wait for container to become healthy"""
        print(
            f"Waiting for container health... Retries: {self.config.health_check_retries}"
        )

        for attempt in range(self.config.health_check_retries):
            print(f"\nHealth check attempt {attempt + 1}")

            # Check container state
            try:
                self._container.reload()
                state = self._container.attrs['State']
                print(f"Container state: {state['Status']}")

                # Print recent logs
                print("Recent container logs:")
                print(self._container.logs(tail=50).decode('utf-8'))

                # If container exited, show exit code and error
                if state['Status'] == 'exited':
                    print(f"Container exited with code {state['ExitCode']}")
                    print(f"Error: {state.get('Error', 'No error message')}")
                    return False

            except Exception as e:
                print(f"Error checking container state: {str(e)}")

            if self.check_health():
                return True
            time.sleep(self.config.health_check_interval)

        print("Container failed all health check attempts")
        return False

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()

    def _log_container_config(self, command, environment, volumes,
                              device_requests, ports):
        """Log container configuration in docker run format"""

        # Format GPU device string
        gpu_str = f"device={','.join(str(id) for id in self.config.gpu_ids)}"

        # Format volume mounts
        volume_strs = []
        for host_path, mount_info in volumes.items():
            volume_strs.append(f"{host_path}:{mount_info['bind']}")

        # Format environment variables
        env_strs = [f"{k}={v}" for k, v in environment.items() if v]

        # Format port mappings
        port_strs = [
            f"{host}:{container}" for container, host in ports.items()
        ]

        # Build command string representation
        cmd_str = "\n".join(
            [
                "Docker container configuration:", "docker run \\",
                f"    --gpus '{gpu_str}' \\", "    --shm-size 1g \\"
            ] + [f"    -p {port_str} \\" for port_str in port_strs] +
            [f"    -v {volume_str} \\" for volume_str in volume_strs] +
            [f"    -e {env_str} \\" for env_str in env_strs] +
            [f"    {self.config.tgi_image} \\"] + [f"    {' '.join(command)}"])

        print(cmd_str)
