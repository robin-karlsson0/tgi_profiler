import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import docker
import requests
from docker.errors import DockerException

from tgi_profiler.utils.colored_logging import ColoredLogger

logger = ColoredLogger(name=__name__)


@dataclass
class TGIConfig:
    """Configuration class for managing Text Generation Inference (TGI) Docker
    containers.

    This class holds all necessary configuration parameters for starting and
    managing a TGI container instance. It handles model specification, hardware
    allocation, networking, token limitations, and authentication settings.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf" or 
        "mistralai/Mistral-7B-v0.1"). Must be a model compatible with TGI.

    gpu_ids : List[int]
        List of GPU device IDs to use for the container. For example, [0] for
        single GPU or [0,1] for multi-GPU. These IDs should match available
        NVIDIA devices on the host.

    port : int
        Host port to bind the container's HTTP interface. Must be available
        and within valid port range (1024-65535). The container's internal port
        80 will be mapped to this host port.

    max_input_length : int
        Maximum number of tokens allowed in the input prompt. Should be within
        the model's context window size (e.g., 4096 for most 7B models).

    max_output_length : int
        Maximum number of tokens allowed in the generated output. Combined with
        max_input_length, should not exceed model's context window.

    hf_token : Optional[str], default=None
        HuggingFace API token for accessing gated models. Required for
        commercial models like Llama-3. Format: "hf_..." or "Bearer hf_...".

    hf_cache_dir : Optional[Path], default=None
        Local directory path for caching model files. If provided, will be
        mounted at /data inside the container.
            Ex: /home/$USER/.cache/huggingface/

    tgi_image : str, default="ghcr.io/huggingface/text-generation-inference:3.0.1"  # noqa
        Docker image to use for the TGI container. Should be a valid TGI image
        from HuggingFace's container registry.

    health_check_retries : int, default=5
        Number of times to retry health checks when starting the container.
        Each retry is separated by health_check_interval seconds.

    health_check_interval : float, default=20.0
        Time in seconds to wait between health check attempts.

    Examples
    --------
    Basic configuration for running a model on a single GPU:
    >>> config = TGIConfig(
    ...     model_id="microsoft/Phi-3-mini-4k-instruct",
    ...     gpu_ids=[0],
    ...     port=8080,
    ...     max_input_length=4096,
    ...     max_output_length=512,
    ...     hf_cache_dir=/home/$USER/.cache/huggingface,
    ...     hf_token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    ... )

    Configuration with model caching and custom health check settings:
    >>> config = TGIConfig(
    ...     model_id="microsoft/Phi-3-mini-4k-instruct",
    ...     gpu_ids=[0,1],
    ...     port=8000,
    ...     max_input_length=8192,
    ...     max_output_length=1024,
    ...     hf_token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    ...     hf_cache_dir=/home/$USER/.cache/huggingface,
    ...     health_check_retries=10,
    ...     health_check_interval=20.0
    ... )

    Notes
    -----
    - The total context length (max_input_length + max_output_length) must not exceed
      the model's maximum context window.
    - The container requires NVIDIA Docker runtime and compatible GPU drivers.
    - Port conflicts will cause container startup to fail.
    - Invalid GPU IDs will cause container startup to fail.
    """
    model_id: str
    gpu_ids: List[int]
    port: int
    max_input_length: int
    max_output_length: int
    hf_token: Optional[str] = None
    hf_cache_dir: Optional[Path] = None
    tgi_image: str = "ghcr.io/huggingface/text-generation-inference:3.0.1"
    health_check_retries: int = 600
    health_check_interval: float = 1.0


class ContainerError(Exception):
    """Base exception for container operations"""
    pass


class TGIContainer:
    """A container manager for Text Generation Inference (TGI) service.

    This class handles the lifecycle of a Docker container running the
    Huggingface Text Generation Inference (TGI) service. It provides methods
    for starting, stopping, and monitoring the health of the container, as well
    as managing its configuration.

    The container is configured using the TGIConfig dataclass, which specifies
    all necessary parameters such as model ID, GPU allocation, and networking
    settings. The container manager supports context manager protocol for
    automated resource cleanup.

    Attributes
    ----------
    config : TGIConfig
        Configuration object containing all parameters for the TGI container.
    _container : docker.models.containers.Container | None
        Internal reference to the Docker container object, None if not running.
    _is_running : bool
        Internal flag tracking container running state.
    _client : docker.client.DockerClient
        Docker client instance for container operations.

    Methods
    -------
    start()
        Starts the TGI container with configured parameters.
    stop()
        Stops and removes the running container.
    check_health()
        Verifies container health status.
    is_running
        Property that checks if container is currently running.

    Examples
    --------
    Basic usage with context manager:
    >>> config = TGIConfig(
    ...     model_id="microsoft/Phi-3-mini-4k-instruct",
    ...     gpu_ids=[0],
    ...     port=8080,
    ...     max_input_length=4096,
    ...     max_output_length=2048
    ... )
    >>> with TGIContainer(config) as container:
    ...     # Container is automatically started
    ...     if container.check_health():
    ...         print("Container is healthy")
    ...     # Container is automatically stopped on exit

    Manual container management:
    >>> container = TGIContainer(config)
    >>> try:
    ...     container.start()
    ...     # Perform operations with container
    ... finally:
    ...     container.stop()

    Notes
    -----
    - Requires Docker daemon to be running and accessible
    - NVIDIA Container Toolkit must be installed for GPU support
    - Container logs are automatically captured and displayed during health
        checks
    - Health checks are performed against the container's HTTP endpoint
    - Container is automatically removed when stopped

    Raises
    ------
    ContainerError
        If container operations (start/stop) fail or health checks timeout
    DockerException
        If Docker daemon is not accessible or returns errors
    """

    def __init__(self, config: TGIConfig):
        """Initialize a new TGI container manager.

        Creates a new container manager instance with the specified
        configuration. The container is not started automatically - use start()
        method or context manager protocol to start the container.

        Parameters
        ----------
        config : TGIConfig
            Configuration object containing all necessary parameters for the
            TGI container. This includes model specifications, hardware
            requirements, and runtime settings.

        Raises
        ------
        DockerException
            If Docker daemon is not accessible

        Notes
        -----
        - The initialization only creates the manager instance, it does not
          start the container
        - Docker client is initialized immediately to fail fast if Docker
          daemon is not accessible
        - Container state tracking variables are initialized to None/False

        Example
        -------
        >>> config = TGIConfig(
        ...     model_id="meta-llama/Llama-2-7b-chat-hf",
        ...     gpu_ids=[0],
        ...     port=8080,
        ...     max_input_length=4096,
        ...     max_output_length=2048
        ... )
        >>> container = TGIContainer(config)
        """
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
        """Start the TGI container with the configured parameters.

        This method handles the complete container startup process, including:
        1. Configuration of environment variables and volume mounts
        2. Setting up GPU device mapping
        3. Configuring network ports
        4. Starting the container with appropriate resource limits
        5. Performing health checks to ensure container is ready

        The method implements various safety checks and provides detailed
        logging of the container configuration. It will wait for the container
        to become healthy before returning, using the configured number of
        health check retries and intervals.

        The container is started with the following key configurations:
        - Environment variables for Hugging Face authentication
        - Volume mounting for model caching (if configured)
        - GPU device mapping based on specified GPU IDs
        - Port mapping from container port 80 to specified host port
        - Shared memory size of 1GB for model loading
        - Custom max length and token settings

        Returns
        -------
        None

        Raises
        ------
        ContainerError
            If container fails to start, health checks fail, or other runtime
            errors occur
        DockerException
            If Docker daemon encounters errors during container operations

        Notes
        -----
        - Method is idempotent - calling it on an already running container
            has no effect
        - Container logs are captured and displayed if startup or health
            checks fail
        - The method blocks until either:
            a) Container becomes healthy
            b) Health check attempts are exhausted
            c) An error occurs
        - GPU devices must be available and properly configured on the host
            system
        - Port conflicts on the host will cause container startup to fail

        Example
        -------
        >>> container = TGIContainer(config)
        >>> try:
        ...     container.start()
        ...     print("Container started successfully")
        ... except ContainerError as e:
        ...     print(f"Failed to start container: {e}")
        """
        # Substitutes manually removing stale contaienrs by "$ docker rm ID"
        # Errors may result in stale containers
        self.cleanup_stale_containers()

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
            # Start container through the Docker Python SDK
            self._container = self._client.containers.run(
                self.config.tgi_image,
                command=command,
                environment=environment,
                volumes=volumes,
                device_requests=device_requests,
                ports=ports,
                shm_size="1g",
                detach=True)

            logger.debug(f"Started container {self._container.id}")

            if not self._wait_for_healthy():
                # Get final logs before failing
                logger.debug("Final container logs:")
                logger.debug(self._container.logs().decode('utf-8'))
                raise ContainerError("Container failed health check")

            self._is_running = True

        except Exception as e:
            if self._container:
                logger.error("Container logs before error:")
                logger.error(self._container.logs().decode('utf-8'))
            raise ContainerError(
                f"Unexpected error starting container: {str(e)}")

    def stop(self) -> None:
        """Stop and remove the running container.

        Attempts to gracefully stop the container with a 10-second timeout,
        then removes it. Cleans up internal state regardless of success.

        Raises
        ------
        ContainerError
            If container fails to stop or remove cleanly
        """
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
        """Check if the container's health endpoint is responding.

        Sends a GET request to the container's health endpoint and verifies
        the response. Logs detailed information about health check attempts
        and failures.

        Returns
        -------
        bool
            True if container is running and health endpoint returns 200,
            False otherwise
        """
        if not self.is_running:
            logger.warning("Container not running during health check")
            return False

        try:
            url = f"http://localhost:{self.config.port}/health"
            logger.debug(f"Trying health check at: {url}")
            response = requests.get(url, timeout=5)
            success = response.status_code == 200
            if success:
                logger.debug("Health check successful")
            else:
                logger.debug(
                    f'Health check failed with status {response.status_code}: '
                    f'{response.text}')
            return success
        except requests.RequestException as e:
            logger.debug(f"Health check failed: {str(e)}")
            return False

    def _format_gpu_request(self) -> dict:
        """Format GPU device request configuration for Docker.

        Creates the device request dictionary that specifies NVIDIA GPU
        access for the container using configured GPU IDs.

        Returns
        -------
        dict
            Docker device request configuration with NVIDIA driver and
            specified GPU IDs
        """
        return {
            "Driver": "nvidia",
            "DeviceIDs": [str(id) for id in self.config.gpu_ids],
            "Capabilities": [["gpu"]]
        }

    def _calculate_total_tokens(self) -> int:
        """Calculate maximum total tokens for the model.

        Computes the total token limit by summing the maximum input
        and output lengths from the configuration.

        Returns
        -------
        int
            Total token limit for the model
        """
        return self.config.max_input_length + self.config.max_output_length

    def _calculate_prefill_tokens(self) -> int:
        """Calculate maximum batch prefill tokens.

        Sets the prefill token limit as input length plus a 50-token
        buffer for optimal batch processing.

        Returns
        -------
        int
            Maximum number of tokens for batch prefill
        """
        return self.config.max_input_length + 50

    def _wait_for_healthy(self) -> bool:
        """Wait for the container to become healthy and ready for requests.

        This method implements a robust health checking mechanism that combines
        both Docker container state monitoring and HTTP endpoint health checks.
        It provides detailed logging of container status and recent logs to aid
        in troubleshooting startup issues.

        The health check process follows these steps for each attempt:
        1. Reload container state from Docker daemon
        2. Check container running status
        3. Capture and display recent container logs
        4. Check for container exit state and capture exit code
        5. Attempt HTTP health check against container endpoint
        6. Wait for configured interval before next attempt

        The method continues this process until either:
        - Container becomes healthy (returns True)
        - Container exits with error (returns False)
        - Maximum retry attempts are exhausted (returns False)

        Returns
        -------
        bool
            True if container becomes healthy within retry limit,
            False if container fails to become healthy or exits

        Notes
        -----
        - Uses configured health_check_retries and health_check_interval
        - Prints detailed logs and state information for debugging
        - Handles both Docker state checks and HTTP endpoint health checks
        - Terminates early if container exits with error
        - May take significant time depending on model size and system
        """
        logger.debug('Waiting for container health... '
                     f'Retries: {self.config.health_check_retries}')

        for attempt in range(self.config.health_check_retries):
            logger.debug(f"\nHealth check attempt {attempt + 1}")

            # Check container state
            try:
                self._container.reload()
                state = self._container.attrs['State']
                logger.debug(f"Container state: {state['Status']}")

                # Print recent logs
                logger.debug("Recent container logs:")
                logger.debug(self._container.logs(tail=50).decode('utf-8'))

                # If container exited, show exit code and error
                if state['Status'] == 'exited':
                    logger.warning(
                        f"Container exited with code {state['ExitCode']}")
                    logger.warning(
                        f"Error: {state.get('Error', 'No error message')}")
                    return False

            except Exception as e:
                logger.warning(f"Error checking container state: {str(e)}")

            # Connects to the container's HTTP health endpoint
            if self.check_health():
                return True
            time.sleep(self.config.health_check_interval)

        logger.error("Container failed all health check attempts")
        return False

    def __enter__(self):
        """Enable context manager protocol for automatic container lifecycle
        management.

        Starts the container when entering a context manager block and returns
        self for use within the block.

        Returns
        -------
        TGIContainer
            The container instance for use within the context
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Handle context manager exit and container cleanup.

        Ensures container is stopped and removed when exiting the context
        manager block, even if an exception occurred. This prevents container
        resource leaks.

        Parameters
        ----------
        exc_type : type
            The type of the exception that occurred, if any
        exc_val : Exception
            The exception instance that occurred, if any
        exc_tb : traceback
            The traceback of the exception that occurred, if any
        """
        self.stop()

    def _log_container_config(self, command, environment, volumes,
                              device_requests, ports):
        """Log the container configuration in human-readable docker run format.

        Formats the container configuration as an equivalent docker run
        command, making it easier to understand the container setup and
        reproduce it manually if needed. Handles formatting of:
        - GPU device mappings
        - Volume mounts
        - Environment variables
        - Port mappings
        - Container command and arguments

        Parameters
        ----------
        command : List[str]
            Container command and arguments
        environment : Dict[str, str]
            Environment variables to set in container
        volumes : Dict[str, Dict[str, str]]
            Volume mount configurations
        device_requests : List[Dict]
            GPU and device mapping configurations
        ports : Dict[int, int]
            Port mapping configuration
        """

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

        logger.debug(cmd_str)

    def cleanup_stale_containers(self):
        """Remove any stale TGI containers."""
        containers = self._client.containers.list(
            all=True, filters={'ancestor': self.config.tgi_image})
        for container in containers:
            try:
                container.remove(force=True)
            except DockerException as e:
                logger.warning(
                    f"Failed to remove stale container {container.id}: {e}")
