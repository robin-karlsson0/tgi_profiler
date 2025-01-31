from importlib.metadata import version

from .config import ProfilerConfig
from .profiler import profile_model
from .utils.colored_logging import ColoredLogger
from .utils.visualize_mem_profile import load_results, plot_results

__version__ = version("tgi-profiler")
__version__ = "0.1.0"

__all__ = [
    "ProfilerConfig",
    "profile_model",
    "plot_results",
    "load_results",
    "ColoredLogger",
]
