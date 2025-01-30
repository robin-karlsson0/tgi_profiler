#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from tgi_profiler import ColoredLogger, ProfilerConfig, profile_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="TGI Memory Profiler - Determine maximum sequence length "
        "capabilities of LLM models on particular machines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
                        help="GPU device IDs to use")
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

    args = parser.parse_args()

    # Validate multimodal arguments
    if args.multimodal and not args.dummy_image:
        parser.error("--dummy-image is required when using --multimodal mode")
    if args.dummy_image and not args.multimodal:
        parser.error("--multimodal flag is required when using --dummy-image")

    return args


def main():
    args = parse_args()

    logger = ColoredLogger(name=__name__, level="INFO")

    if not args.hf_token:
        logger.error(
            "HuggingFace token not found. Please set HF_TOKEN environment variable or use --hf-token"  # noqa
        )
        sys.exit(1)

    # Validate dummy image path if in multimodal mode
    if args.multimodal and not args.dummy_image.exists():
        logger.error(f"Dummy image not found at: {args.dummy_image}")
        sys.exit(1)

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

    # Run profiling
    logger.info("Starting profiling run")
    logger.info(f"Model: {config.model_id}")
    logger.info(f"Grid size: {config.grid_size}")
    logger.info(f"Refinement rounds: {config.refinement_rounds}")
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


if __name__ == "__main__":
    main()
