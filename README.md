# TGI Memory Profiler

A framework for empirically determining maximum sequence length capabilities of LLM models deployed via Text Generation Inference (TGI), helping avoid out-of-memory (OOM) errors.

## Features

- **Adaptive Grid Search**: Efficiently explores viable combinations of input and output sequence lengths
- **Boundary Detection**: Automatically identifies and refines the boundary between successful and failed memory configurations
- **Container Management**: Handles Docker container lifecycle for isolated testing
- **Token-Aware**: Ensures accurate sequence length validation using model-specific tokenizers
- **Progress Tracking**: Detailed logging and progress visualization
- **Persistence**: Results are serializable to JSON for analysis or resumption

## Installation

```bash
pip install tgi-profiler
```

## Prerequisites

- Docker with NVIDIA Container Toolkit
- Python 3.8+
- Hugging Face account with API token for accessing models

Set up your environment variables:

```bash
# ~/.bashrc or similar
export HF_TOKEN=YOUR-HF-TOKEN
export HF_DIR=PATH-TO-YOUR-HUGGINGFACE-CACHE-DIR
```

## Quick Start

```python
from tgi_profiler import ProfilerConfig, profile_model

# Configure profiling parameters
config = ProfilerConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    gpu_ids=[0],
    min_input_length=128,
    max_input_length=8192,
    min_output_length=128,
    max_output_length=4096,
    grid_size=10,  # Initial grid resolution
    refinement_rounds=2  # Number of boundary refinement passes
)

# Run profiling
results = profile_model(config)
```

## How It Works

1. **Initial Grid Search**: Tests a uniform grid of input/output length combinations
2. **Boundary Detection**: Identifies transitions between successful and failed regions
3. **Adaptive Refinement**: Focuses additional testing near boundary regions
4. **Result Analysis**: Determines safe operating parameters for your model deployment

## Configuration Options

```python
class ProfilerConfig:
    # Model & hardware settings
    model_id: str  # HuggingFace model identifier
    gpu_ids: List[int]  # GPU devices to use
    port: int = 8080  # TGI container port
    
    # Sequence length bounds
    min_input_length: int 
    max_input_length: int
    min_output_length: int
    max_output_length: int
    
    # Search parameters
    grid_size: int  # Initial sampling density
    refinement_rounds: int  # Boundary refinement passes
    output_tolerance_pct: float = 0.05  # Output length tolerance
    
    # Optional settings
    hf_token: Optional[str]  # HuggingFace API token
    hf_cache_dir: Optional[Path]  # Model cache directory
    output_dir: Path = Path("profiler_results")
```

## Results

Results are saved as JSON files containing:
- Test configurations
- Success/failure status for each point
- Error messages and container logs
- Timestamps for progress tracking

Example visualization:
```python
from tgi_profiler.utils.visualize_mem_profile import plot_results

# Load and visualize results
plot_results("profiler_results/profile_res_20250128_233107.json")
```

## Advanced Usage

### Resuming from Previous Results

```python
config = ProfilerConfig(
    # ... other settings ...
    resume_from_file="profiler_results/previous_run.json"
)
results = profile_model(config)
```

### Custom Boundary Detection

```python
config = ProfilerConfig(
    # ... other settings ...
    k_neighbors=5,  # Nearest neighbors for local boundary detection
    m_random=3,     # Random samples for global exploration
    distance_scale=1000,  # Scale factor for distance-based scoring
    consistency_radius=1000,  # Maximum distance for consistency checks
)
```

## Contributing

Contributions are welcome! Please check our contributing guidelines and feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{tgi_profiler,
  title = {TGI Memory Profiler},
  year = {2024},
  description = {A framework for empirically determining LLM memory limits in TGI deployments}
}
```