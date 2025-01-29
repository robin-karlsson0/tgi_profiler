from .config import ProfilerConfig
from .profiler import profile_model
from .utils.colored_logging import ColoredLogger
from .utils.visualize_mem_profile import plot_results

__version__ = "0.1.0"

__all__ = [
    "ProfilerConfig",
    "profile_model",
    "plot_results",
    "ColoredLogger",
]
