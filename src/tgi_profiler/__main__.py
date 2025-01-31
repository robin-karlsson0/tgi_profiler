#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from tgi_profiler import (ColoredLogger, ProfilerConfig, load_results,
                          plot_results, profile_model)


def add_profile_args(parser):
    """Add profiling-specific arguments to parser"""
    # Required arguments
    parser.add_argument(
        "model_id",
        type=str,
        help="HF model identifier (e.g., meta-llama/Llama-3.1-8B-Instruct)")

    # Optional arguments
    parser.add_argument("--gpu-ids",
                        type=int,
                        nargs="+",
                        default=[0],
                        help="Sequence of GPU device IDs to use (ex: 0 1 2 3)")
    parser.add_argument("--min-input-length",
                        type=int,
                        default=2048,
                        help="Minimum input sequence length to test")
    parser.add_argument("--max-input-length",
                        type=int,
                        default=32768,
                        help="Maximum input sequence length to test")
    parser.add_argument("--min-output-length",
                        type=int,
                        default=512,
                        help="Minimum output sequence length to test")
    parser.add_argument("--max-output-length",
                        type=int,
                        default=32768,
                        help="Maximum output sequence length to test")
    parser.add_argument("--grid-size",
                        type=int,
                        default=4,
                        help="Initial grid resolution for testing")
    parser.add_argument("--refinement-rounds",
                        type=int,
                        default=6,
                        help="Number of boundary refinement passes")
    parser.add_argument("--port",
                        type=int,
                        default=8080,
                        help="Port for TGI container")
    parser.add_argument("--output-dir",
                        type=Path,
                        default=Path("./profiler_results"),
                        help="Directory to store results")
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="HuggingFace API token (defaults to HF_TOKEN env variable)")
    parser.add_argument("--retries-per-point",
                        type=int,
                        default=3,
                        help="Number of retries per test point")
    parser.add_argument("--resume-from",
                        type=str,
                        help="Path to previous results file to resume from")

    # Multimodal arguments
    parser.add_argument("--multimodal",
                        action="store_true",
                        help="Enable multimodal profiling mode")
    parser.add_argument("--dummy-image",
                        type=Path,
                        help="Path to dummy image for multimodal profiling")


def add_visualize_args(parser):
    """Add visualization-specific arguments to parser"""
    parser.add_argument('results_file',
                        type=str,
                        help='Path to results JSON file')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='Output path for plot image')
    parser.add_argument('--no-show',
                        action='store_true',
                        help='Do not display the plot')
    parser.add_argument('--no-boundary',
                        action='store_true',
                        help='Do not plot estimated boundary')
    parser.add_argument('--major-tick',
                        type=int,
                        default=5000,
                        help='Interval for major grid lines')
    parser.add_argument('--minor-tick',
                        type=int,
                        default=1000,
                        help='Interval for minor grid lines')


def validate_profile_args(args):
    """Validate profiling arguments"""
    logger = ColoredLogger(name=__name__, level="INFO")

    if not args.hf_token:
        logger.error(
            "HuggingFace token not found. Please set HF_TOKEN environment variable or use --hf-token"  # noqa
        )
        sys.exit(1)

    # Validate multimodal arguments
    if args.multimodal and not args.dummy_image:
        logger.error("--dummy-image is required when using --multimodal mode")
        sys.exit(1)
    if args.dummy_image and not args.multimodal:
        logger.error("--multimodal flag is required when using --dummy-image")
        sys.exit(1)

    # Validate dummy image path if in multimodal mode
    if args.multimodal and not args.dummy_image.exists():
        logger.error(f"Dummy image not found at: {args.dummy_image}")
        sys.exit(1)


def handle_profile(args):
    """Handle the profile subcommand"""
    logger = ColoredLogger(name=__name__, level="INFO")
    validate_profile_args(args)

    # Create config from arguments
    config = ProfilerConfig(
        model_id=args.model_id,
        gpu_ids=args.gpu_ids,
        min_input_length=args.min_input_length,
        max_input_length=args.max_input_length,
        min_output_length=args.min_output_length,
        max_output_length=args.max_output_length,
        grid_size=args.grid_size,
        refinement_rounds=args.refinement_rounds,
        port=args.port,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
        retries_per_point=args.retries_per_point,
        resume_from_file=args.resume_from,
        multimodal=args.multimodal,
        dummy_image_path=args.dummy_image if args.multimodal else None,
    )

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("Starting profiling run")
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Grid size: {config.grid_size}")
    logger.info(f"Refinement rounds: {config.refinement_rounds}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Retries per point: {config.retries_per_point}")
    logger.info(f"Resume from: {config.resume_from_file}")
    logger.info(f"Port: {config.port}")
    logger.info(f"Minimum input length: {config.min_input_length}")
    logger.info(f"Maximum input length: {config.max_input_length}")
    logger.info(f"Minimum output length: {config.min_output_length}")
    logger.info(f"Maximum output length: {config.max_output_length}")
    logger.info(f"GPU IDs: {config.gpu_ids}")
    logger.info(f"HuggingFace token: {'*****' if config.hf_token else 'None'}")
    logger.info(f"HuggingFace cache dir: {config.hf_cache_dir}")
    if config.multimodal:
        logger.info("Mode: Multimodal")
        logger.info(f"Dummy image: {config.dummy_image_path}")
    else:
        logger.info("Mode: Text-only")

    results = profile_model(config)

    # Print summary
    successes = sum(1 for r in results if r.success)
    logger.info(f"Completed {len(results)} test points")
    logger.info(f"Successful points: {successes}")
    logger.info(f"Failed points: {len(results) - successes}")

    # Print maximum successful lengths
    successful_results = [r for r in results if r.success]
    if successful_results:
        max_input = max(r.input_length for r in successful_results)
        max_output = max(r.output_length for r in successful_results)
        logger.info(f"Maximum successful input length: {max_input}")
        logger.info(f"Maximum successful output length: {max_output}")
    else:
        logger.warning("No successful test points found")


def handle_visualize(args):
    """Handle the visualize subcommand"""
    # Load and plot results
    data = load_results(args.results_file)
    plot_results(data,
                 output_path=args.output,
                 show_plot=not args.no_show,
                 fit_boundary=not args.no_boundary,
                 major_tick_interval=args.major_tick,
                 minor_tick_interval=args.minor_tick)


def main():
    """Main entry point for the TGI Memory Profiler CLI"""
    from importlib.metadata import version

    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description=
        "TGI Memory Profiler - Profile and visualize LLM memory usage",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--version',
                        action='version',
                        version=f'%(prog)s {version("tgi-profiler")}')

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command',
                                       help='Command to execute',
                                       required=True)

    # Create parser for the "profile" command
    profile_parser = subparsers.add_parser(
        'profile',
        help='Profile memory usage of a model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_profile_args(profile_parser)

    # Create parser for the "visualize" command
    visualize_parser = subparsers.add_parser(
        'visualize',
        help='Visualize profiling results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_visualize_args(visualize_parser)

    # Parse arguments and call appropriate handler
    args = parser.parse_args()
    if args.command == 'profile':
        handle_profile(args)
    elif args.command == 'visualize':
        handle_visualize(args)


if __name__ == "__main__":
    main()
