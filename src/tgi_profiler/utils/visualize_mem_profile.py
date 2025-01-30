#!/usr/bin/env python3
"""
TGI Memory Profile Visualizer

This module provides visualization tools for analyzing Text Generation
Inference (TGI) memory profiling results. It creates detailed plots showing the
relationship between input and output sequence lengths, highlighting successful
and failed configurations.

The visualizer includes:
- Success/failure scatter plot with adaptive point sizing
- Estimated memory boundary curve using monotonic polynomial fitting
- Density heatmap showing test concentration areas
- Customizable grid with K-unit formatting
- Optional boundary and density visualization

Example Usage:
    # Basic usage
    python viz_mem_profile.py results.json

    # Save plot with custom settings
    python viz_mem_profile.py results.json -o plot.png --major-tick 2500

Dependencies:
    - numpy
    - matplotlib
    - scipy
    - sklearn
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.isotonic import IsotonicRegression


def find_boundary_points(x, y, success):
    """
    Find points that lie on the boundary between success and failure regions.
    For each unique x value, takes the midpoint between highest success and
    lowest failure.
    """
    x_unique = np.unique(x)
    boundary_points = []

    for x_val in x_unique:
        mask = (x == x_val)
        success_y = y[mask & success]
        failure_y = y[mask & (~success)]

        if len(success_y) > 0 and len(failure_y) > 0:
            boundary_points.append(
                [x_val, (np.max(success_y) + np.min(failure_y)) / 2])

    return np.array(boundary_points) if boundary_points else None


def fit_monotone_boundary(x, y, success, degree=3):
    """
    Fit a monotonically decreasing polynomial to the boundary between
    success/failure regions.

    Args:
        x: Input lengths
        y: Output lengths
        success: Boolean array indicating success/failure
        degree: Degree of polynomial to fit

    Returns:
        Polynomial coefficients or None if fitting fails
    """
    boundary_points = find_boundary_points(x, y, success)

    if boundary_points is None or len(boundary_points) < degree + 1:
        print("Warning: Not enough boundary points for polynomial fitting")
        return None

    # Apply isotonic regression to ensure monotonicity
    ir = IsotonicRegression(increasing=False)
    y_monotone = ir.fit_transform(boundary_points[:, 0], boundary_points[:, 1])

    # Fit polynomial to monotonic points
    coeffs = np.polyfit(boundary_points[:, 0], y_monotone, degree)

    return coeffs


def plot_results(data,
                 output_path=None,
                 show_plot=True,
                 fit_boundary=True,
                 major_tick_interval=5000,
                 minor_tick_interval=1000):
    """
    Create a scatter plot of the results with success/failure points and
    estimated boundary.

    Args:
        data: Dictionary containing profiling results
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        fit_boundary: Whether to estimate and plot the boundary
        show_density: Whether to show density heatmap
        major_tick_interval: Interval for major grid lines (in tokens)
        minor_tick_interval: Interval for minor grid lines (in tokens)
    """
    # Extract points and their success status
    points = [(r['input_length'], r['output_length'], r['success'])
              for r in data['results']]

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    success = np.array([p[2] for p in points])

    # Create the plot
    plt.figure(figsize=(12, 9))

    # Plot points
    success_mask = success
    failure_mask = ~success

    # Compute point sizes based on local density
    if len(x) > 1:
        xy = np.vstack([np.log10(x), np.log10(y)])
        kde = gaussian_kde(xy)
        local_density = kde(xy)
        point_sizes = 100 + 200 * (local_density - local_density.min()) / (
            local_density.max() - local_density.min())
    else:
        point_sizes = 100

    if np.any(success_mask):
        plt.scatter(x[success_mask],
                    y[success_mask],
                    c='green',
                    label='Success',
                    alpha=0.6,
                    s=point_sizes[success_mask],
                    zorder=3)
    if np.any(failure_mask):
        plt.scatter(x[failure_mask],
                    y[failure_mask],
                    c='red',
                    label='Failure',
                    alpha=0.6,
                    s=point_sizes[failure_mask],
                    zorder=3)

    # Fit and plot boundary if requested
    if fit_boundary and len(x) > 4:  # Need at least 4 points for cubic fit
        coeffs = fit_monotone_boundary(x, y, success)
        if coeffs is not None:
            x_smooth = np.geomspace(min(x), max(x), 100)
            y_smooth = np.polyval(coeffs, x_smooth)
            plt.plot(x_smooth,
                     y_smooth,
                     'b-',
                     label='Estimated Boundary',
                     linewidth=2,
                     zorder=2)

    # Add labels and title
    plt.xlabel('Input Length (#tokens)')
    plt.ylabel('Output Length (#tokens)')
    plt.title(f'TGI Memory Profile: {data["config"]["model_id"]}\n'
              f'GPU(s): {data["config"]["gpu_ids"]}')

    # Define K-units formatter
    def format_k(x, p):
        """Format with K units"""
        if x >= 1000:
            return f'{int(x/1000)}K'
        return str(int(x))

    # Set up grid with more density
    ax = plt.gca()

    # Set major and minor ticks
    major_ticks_x = np.arange(0, np.max(x) * 1.1, major_tick_interval)
    minor_ticks_x = np.arange(0, np.max(x) * 1.1, minor_tick_interval)
    major_ticks_y = np.arange(0, np.max(y) * 1.1, major_tick_interval)
    minor_ticks_y = np.arange(0, np.max(y) * 1.1, minor_tick_interval)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # Format ticks with K units
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_k))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_k))

    # Customize grid
    plt.grid(True, which='major', linestyle='-', alpha=0.3)
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)

    # Set axis properties
    plt.axis('equal')
    plt.xlim(0, np.max(x) * 1.1)
    plt.ylim(0, np.max(y) * 1.1)

    # Add legend
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plot if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_results(filepath):
    """Load and parse the results JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def main():
    """
    Command-line interface for the TGI Memory Profile visualizer.

    This function handles command-line argument parsing and drives the
    visualization process. It supports various customization options including:
    - Output file path specification
    - Display control (show/hide plot)
    - Visualization features (boundary curve, density heatmap)
    - Grid customization (major and minor tick intervals)

    The function expects a JSON file containing TGI memory profiling results
    structured as:
    {
        "config": {
            "model_id": str,
            "gpu_ids": List[int],
            ...
        },
        "results": [
            {
                "input_length": int,
                "output_length": int,
                "success": bool,
                ...
            },
            ...
        ]
    }

    Returns:
        None. The function either displays the plot, saves it to a file,
        or both, depending on the provided arguments.
    """
    parser = argparse.ArgumentParser(
        description='Visualize TGI Memory Profile results')

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
                        help='Interval for major grid lines (default: 5000)')
    parser.add_argument('--minor-tick',
                        type=int,
                        default=1000,
                        help='Interval for minor grid lines (default: 1000)')

    args = parser.parse_args()

    # Load and plot results
    data = load_results(args.results_file)
    plot_results(data,
                 output_path=args.output,
                 show_plot=not args.no_show,
                 fit_boundary=not args.no_boundary,
                 major_tick_interval=args.major_tick,
                 minor_tick_interval=args.minor_tick)


if __name__ == "__main__":
    main()
