# TGI Memory Profiler

A framework for empirically determining the maximum sequence length capabilities of Large Language Models (LLMs) deployed via Text Generation Inference (TGI). This tool helps prevent out-of-memory (OOM) errors by identifying safe operating boundaries for input and output sequence lengths.

## Key Features

- **Adaptive Grid Search**: Systematically explores viable combinations of input and output sequence lengths
- **Intelligent Boundary Detection**: Uses KNN and spatial analysis to identify and refine the boundary between successful and failed memory configurations
- **Automated Container Management**: Handles TGI Docker container lifecycle with robust health checks and cleanup
- **Token-Exact Testing**: Ensures precise sequence length validation using model-specific tokenizers
- **Multimodal Support**: Optional testing for multimodal models with image inputs
- **Resumable Profiling**: Save and resume profiling sessions from previous results
- **Visualization Tools**: Built-in plotting utilities for analyzing memory boundaries
- **Progress Tracking**: Detailed logging with colored console output
- **Configurable Retries**: Robust error handling with configurable retry attempts

## Prerequisites

- Docker with NVIDIA Container Toolkit
- Python 3.7+
- HuggingFace account and API token (for accessing models)
- NVIDIA GPU(s) with compatible drivers

## Installation

```bash
pip install tgi-profiler
```

## Development Installation

Clone the repository:
```bash
git clone https://github.com/robin-karlsson0/tgi_profiler
cd tgi_profiler
```

Install in development mode
```bash
pip install -e .
```

## Environment Setup

```bash
# Add to ~/.bashrc or equivalent
export HF_TOKEN="your-huggingface-token"  # Required for accessing models
export HF_DIR="/path/to/cache"  # Optional: HuggingFace cache directory
```

## Quick Start

### Command Line Usage

Basic profiling:
```bash
tgi-profiler meta-llama/Llama-3.1-8B-Instruct \
    --gpu-ids 0 \
    --min-input-length 128 \
    --max-input-length 32768 \
    --min-output-length 128 \
    --max-output-length 32768 \
    --grid-size 4 \
    --refinement-rounds 6
    --output-dir profiler_result_llama3_1_8b
```

Multimodal model profiling:
```bash
tgi-profiler meta-llama/Llama-3.2-11B-Vision-Instruct \
    --multimodal \
    --dummy-image PATH/TO/IMG \
    --gpu-ids 0 1 \
    --min-input-length 128 \
    --max-input-length 32768 \
    --min-output-length 128 \
    --max-output-length 32768 \
    --grid-size 4 \
    --refinement-rounds 6 \
    --output-dir profiler_results_llama3_2_8b
```

Run `tgi-profiler --help` for full list of input arguments

### Python API Usage

```python
from tgi_profiler import ProfilerConfig, profile_model, plot_results

# Configure profiling parameters
config = ProfilerConfig(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    gpu_ids=[0],
    min_input_length=2048,
    max_input_length=32768,
    min_output_length=2048,
    max_output_length=32768,
    grid_size=4,                # Initial grid resolution
    refinement_rounds=6,        # Boundary refinement passes
    retries_per_point=3,        # Retries for each test point
    output_tolerance_pct=0.05,  # Tolerance for output length variation
)

# Run profiling
results = profile_model(config)

# Visualize results
plot_results("profiler_results/profile_res_20240130_120000.json")
```

## Advanced Configuration

### Boundary Detection Parameters

Fine-tune the boundary detection algorithm:
```python
config = ProfilerConfig(
    # ... basic settings ...
    k_neighbors=5,              # Neighbors for local boundary detection
    m_random=3,                 # Random samples for global exploration
    distance_scale=1000,        # Scale factor for distance-based scoring
    consistency_radius=1000,    # Maximum distance for consistency
    redundancy_weight=0.5,      # Weight for penalizing redundant pairs
    min_refinement_dist=50,     # Minimum distance between refinement points
)
```

### Resuming Previous Runs
```python
config = ProfilerConfig(
    # ... other settings ...
    resume_from_file="profiler_results/previous_run.json"
)
results = profile_model(config)
```

## Output Format

Results are saved as JSON files containing:

- Model and hardware configuration
- Test point results (input length, output length, success/failure)
- Error messages and container logs
- Timestamps for tracking
- Boundary detection parameters

## Visualization

The included visualization tool creates plots showing:

- Success/failure regions
- Estimated memory boundary curve
- Point density heatmap

## Contributing
Contributions are welcome! Please see our contributing guidelines (CONTRIBUTING.md) for details on:

- Code style and formatting
- Testing requirements
- Pull request process

## License

This project is licensed under the GNU General Public License v3 - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{tgi_profiler,
    title = {TGI Memory Profiler},
    year = {2024},
    author = {Karlsson, Robin},
    description = {A framework for empirically determining LLM memory limits in TGI deployments},
    url = {https://github.com/robin-karlsson0/tgi_profiler}
}
```