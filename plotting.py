from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_over_time(accuracies_by_method):
    plt.figure(figsize=(12, 5))
    for method, acc in accuracies_by_method.items():
        plt.plot(acc, label=method)
    plt.xlabel("Time step")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_detection_delays(delays_dict):
    import matplotlib.pyplot as plt

    methods = list(delays_dict.keys())
    delays = list(delays_dict.values())
    plt.figure(figsize=(8, 5))
    plt.bar(methods, delays, color="orange")
    plt.ylabel("Detection Delay (steps)")
    plt.title("Detection Delay by Method")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()


def plot_combined_martingale_accuracy_severity(
    log_sj_dict: Dict[str, List[float]],
    epsilons_dict: Dict[str, List[float]],
    accuracies_dict: Dict[str, List[float]],
    entropy_dict: Dict[str, List[float]],
    batch_size: int,
    title: str = "Martingale, Epsilon, Accuracy, and Entropy Over Time",
):
    """
    Plot log(Sj), epsilon, batch-wise accuracy, and entropy for multiple severity levels.

    Parameters:
        log_sj_dict (Dict[str, List[float]]): Dict of martingale log-wealth per sample.
        epsilons_dict (Dict[str, List[float]]): Dict of epsilon values per sample.
        accuracies_dict (Dict[str, List[float]]): Dict of accuracy values per batch.
        entropy_dict (Dict[str, List[float]]): Dict of entropy values per sample.
        batch_size (int): Number of samples per accuracy point.
        title (str): Title of the figure.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    shift_start = 3500
    shift_end = 5000
    axs[0].set_xlim(shift_start, shift_end)
    axs[1].set_xlim(shift_start, shift_end)
    axs[2].set_xlim(shift_start, shift_end)
    axs[3].set_xlim(shift_start, shift_end)

    for severity in sorted(log_sj_dict.keys()):
        log_sj = np.array(log_sj_dict[severity])
        epsilons = np.array(epsilons_dict[severity])
        accuracies = np.array(accuracies_dict[severity])
        entropy = np.array(entropy_dict[severity])

        steps = np.arange(len(log_sj))
        acc_steps = np.arange(0, len(log_sj), batch_size)[: len(accuracies)]

        axs[0].plot(steps, log_sj, label=f"Severity {severity}")
        axs[1].plot(steps, epsilons, label=f"Severity {severity}")
        axs[2].plot(acc_steps, accuracies, marker="o", linestyle="-", label=f"Severity {severity}")

        # Smooth entropy for clarity
        smooth_entropy = np.convolve(entropy, np.ones(50) / 50, mode="valid")
        smooth_steps = np.arange(len(smooth_entropy)) + 25
        axs[3].plot(smooth_steps, smooth_entropy, label=f"Severity {severity}")

    axs[0].axhline(np.log(100), linestyle="--", color="red", label="log(Sj)=log(100)")
    axs[0].set_ylabel("log(Sj)")
    axs[0].set_ylim(-5, 80)
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_ylabel("Epsilon")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].set_ylabel("Accuracy")
    axs[2].set_xlabel("Time step (samples)")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].set_ylabel("Entropy")
    axs[3].set_xlabel("Time step (samples)")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # Suppress individual legends
    for ax in axs:
        ax.get_legend().remove()
        ax.axvline(4000, color="gray", linestyle="-", alpha=0.5)

    # Collect unique severity levels and relabel them
    severity_labels = sorted(log_sj_dict.keys())

    # Get handles and labels from the first subplot
    handles, labels = axs[0].get_legend_handles_labels()

    # Create mapping between labels and desired display names
    display_labels = [f"Severity {severity.split('_s')[-1]}" for severity in severity_labels]

    # Create filtered handles and labels
    filtered = list(zip(handles, display_labels))
    handles, labels = zip(*filtered)

    # Add unified legend at bottom
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))

    # Adjust layout to make room
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Adjust layout to make room for bottom legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def plot_combined_martingale_accuracy_corruption(
    log_sj_dict: Dict[str, List[float]],
    epsilons_dict: Dict[str, List[float]],
    accuracies_dict: Dict[str, List[float]],
    entropy_dict: Dict[str, List[float]],
    batch_size: int,
    title: str = "Martingale, Epsilon, Accuracy, and Entropy Over Time",
):
    """
    Plot log(Sj), epsilon, batch-wise accuracy, and entropy for multiple corruption types.

    Parameters:
        log_sj_dict (Dict[str, List[float]]): Dict of martingale log-wealth per sample.
        epsilons_dict (Dict[str, List[float]]): Dict of epsilon values per sample.
        accuracies_dict (Dict[str, List[float]]): Dict of accuracy values per batch.
        entropy_dict (Dict[str, List[float]]): Dict of entropy values per sample.
        batch_size (int): Number of samples per accuracy point.
        title (str): Title of the figure.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    shift_start = 3500
    shift_end = 5000
    for ax in axs:
        ax.set_xlim(shift_start, shift_end)

    all_labels = []
    for corruption_key in sorted(log_sj_dict.keys()):
        corruption_name = corruption_key.rsplit("_", 1)[0]  # e.g., 'gaussian_noise'
        label = corruption_name.replace("_", " ").title()
        all_labels.append(label)

        log_sj = np.array(log_sj_dict[corruption_key])
        epsilons = np.array(epsilons_dict[corruption_key])
        accuracies = np.array(accuracies_dict[corruption_key])
        entropy = np.array(entropy_dict[corruption_key])

        steps = np.arange(len(log_sj))
        acc_steps = np.arange(0, len(log_sj), batch_size)[: len(accuracies)]

        # Smooth entropy for clarity
        smooth_entropy = np.convolve(entropy, np.ones(50) / 50, mode="valid")
        smooth_steps = np.arange(len(smooth_entropy)) + 25
        axs[3].plot(smooth_steps, smooth_entropy, label="Severity 5")

        axs[0].plot(steps, log_sj, label=label)
        axs[1].plot(steps, epsilons, label=label)
        axs[2].plot(acc_steps, accuracies, marker="o", linestyle="-", label=label)
        # axs[3].plot(steps, entropy, label=label)

    axs[0].axhline(np.log(100), linestyle="--", color="red", label="log(Sj)=log(100)")
    axs[0].set_ylabel("log(Sj)")
    axs[0].set_title(title)
    axs[0].grid(True)

    axs[1].set_ylabel("Epsilon")
    axs[1].grid(True)

    axs[2].set_ylabel("Accuracy")
    axs[2].set_xlabel("Time step (samples)")
    axs[2].grid(True)

    axs[3].set_ylabel("Entropy")
    axs[3].set_xlabel("Time step (samples)")
    axs[3].grid(True)

    # Unified legend at bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

def plot_combined_noise_shift(
    log_sj_dict: Dict[str, List[float]],
    epsilons_dict: Dict[str, List[float]],
    accuracies_dict: Dict[str, List[float]],
    entropy_dict: Dict[str, List[float]],
    batch_size: int,
    title: str = "Comparison for Label Noise, Prior Shift, and Combined Shift",
):

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    shift_start = 0
    shift_end = 5000
    for ax in axs:
        ax.set_xlim(shift_start, shift_end)

    all_labels = []
    for corruption_key in sorted(log_sj_dict.keys()):
        corruption_name = corruption_key.rsplit("_", 1)[0]  # e.g., 'gaussian_noise'
        label = corruption_name.replace("_", " ").title()
        all_labels.append(label)

        log_sj = np.array(log_sj_dict[corruption_key])
        epsilons = np.array(epsilons_dict[corruption_key])
        accuracies = np.array(accuracies_dict[corruption_key])
        entropy = np.array(entropy_dict[corruption_key])

        steps = np.arange(len(log_sj))
        acc_steps = np.arange(0, len(log_sj), batch_size)[: len(accuracies)]

        # Smooth entropy for clarity
        smooth_entropy = np.convolve(entropy, np.ones(50) / 50, mode="valid")
        smooth_steps = np.arange(len(smooth_entropy)) + 25
        axs[3].plot(smooth_steps, smooth_entropy, label="Severity 5")

        axs[0].plot(steps, log_sj, label=label)
        axs[1].plot(steps, epsilons, label=label)
        axs[2].plot(acc_steps, accuracies, marker="o", linestyle="-", label=label)
        # axs[3].plot(steps, entropy, label=label)

    axs[0].axhline(np.log(100), linestyle="--", color="red", label="log(Sj)=log(100)")
    axs[0].set_ylabel("log(Sj)")
    axs[0].set_title(title)
    axs[0].grid(True)

    axs[1].set_ylabel("Epsilon")
    axs[1].grid(True)

    axs[2].set_ylabel("Accuracy")
    axs[2].set_xlabel("Time step (samples)")
    axs[2].grid(True)

    axs[3].set_ylabel("Entropy")
    axs[3].set_xlabel("Time step (samples)")
    axs[3].grid(True)

    # Unified legend at bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()


def plot_detection_delays(detection_delays):
    """
    Plot detection delays for different corruption types.

    Args:
        detection_delays (dict): Dictionary mapping corruption types to their detection delays
    """
    # Sort for better readability
    sorted_delays = dict(sorted(detection_delays.items(), key=lambda x: x[1]))

    # Plotting
    plt.figure(figsize=(12, 6))
    bars = plt.barh(list(sorted_delays.keys()), list(sorted_delays.values()), color="skyblue")

    # Add value labels to the end of each bar
    for bar in bars:
        plt.text(
            bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.0f}", va="center", fontsize=9
        )

    plt.xlabel("Detection Delay (samples after shift)")
    plt.title("Detection Delay of Martingale Approach by Corruption Type (Severity 5)")
    plt.grid(True, axis="x", linestyle=":")
    plt.tight_layout()
    plt.show()


def plot_dynamic_stream_analysis(
    log_sj_dict: Dict[str, List[float]],
    epsilons_dict: Dict[str, List[float]],
    accuracies_dict: Dict[str, List[float]],
    entropy_dict: Dict[str, List[float]],
    segment_schedule: List[str],
    segment_size: int,
    batch_size: int,
    title: str = "Dynamic Stream Analysis",
):
    """
    Plot log(Sj), epsilon, batch-wise accuracy, and entropy for a dynamic stream with changing corruption levels.

    Parameters:
        log_sj_dict (Dict[str, List[float]]): Dict of martingale log-wealth per sample.
        epsilons_dict (Dict[str, List[float]]): Dict of epsilon values per sample.
        accuracies_dict (Dict[str, List[float]]): Dict of accuracy values per batch.
        entropy_dict (Dict[str, List[float]]): Dict of entropy values per sample.
        segment_schedule (List[str]): List of segment types in order (e.g. ["clean", "s2", "s5", "clean", "s4"]).
        segment_size (int): Number of samples in each segment.
        batch_size (int): Number of samples per accuracy point.
        title (str): Title of the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    # Create figure with 4 vertically stacked subplots sharing x-axis
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # Define colors for segments and their boundaries
    segment_colors = {
        "clean": "#98FB98",  # Pale green
        "s1": "#ADD8E6",  # Light blue
        "s2": "#FFCC99",  # Peach
        "s3": "#DDA0DD",  # Plum
        "s4": "#FFA07A",  # Light salmon
        "s5": "#9370DB",  # Medium purple
    }

    # Full range of data for x-axis
    total_length = len(list(log_sj_dict.values())[0])
    steps = np.arange(total_length)

    # Calculate segment boundaries
    segment_boundaries = [i * segment_size for i in range(len(segment_schedule) + 1)]

    # Add colored background for each segment
    for i, segment in enumerate(segment_schedule):
        start = segment_boundaries[i]
        end = segment_boundaries[i + 1]
        for ax in axs:
            ax.axvspan(start, end, alpha=0.2, color=segment_colors.get(segment, "lightgray"))
            ax.axvline(x=start, color="gray", linestyle="--", alpha=0.7)

    # Plot data for each metric
    for key in log_sj_dict.keys():
        # Extract data
        log_sj = np.array(log_sj_dict[key])
        epsilons = np.array(epsilons_dict[key])
        accuracies = np.array(accuracies_dict[key])
        entropy = np.array(entropy_dict[key])

        # Calculate x positions for accuracy points
        acc_steps = np.arange(0, total_length, batch_size)[: len(accuracies)]

        # Plot log(Sj) in the first subplot
        axs[0].plot(steps, log_sj, label=key, linewidth=2)

        # Plot epsilon in the second subplot
        axs[1].plot(steps, epsilons, label=key, linewidth=2)

        # Plot accuracy in the third subplot
        axs[2].plot(acc_steps, accuracies, marker="o", linestyle="-", label=key, markersize=4)

        # Smooth entropy for clarity and plot in the fourth subplot
        smooth_entropy = np.convolve(entropy, np.ones(50) / 50, mode="valid")
        smooth_steps = np.arange(len(smooth_entropy)) + 25
        axs[3].plot(smooth_steps, smooth_entropy, label=key, linewidth=2)

    # Add threshold line for log(Sj)
    axs[0].axhline(np.log(100), linestyle="--", color="red", label="Detection threshold (log(100))")

    # Add segment labels
    for i, segment in enumerate(segment_schedule):
        mid_point = segment_boundaries[i] + segment_size / 2
        axs[0].text(
            mid_point,
            axs[0].get_ylim()[1] * 0.9,
            segment,
            horizontalalignment="center",
            fontsize=12,
            fontweight="bold",
        )

    # Configure subplot titles and labels
    axs[0].set_title(title, fontsize=14, fontweight="bold")
    axs[0].set_ylabel("log(Sj)")
    axs[1].set_ylabel("Epsilon")
    axs[2].set_ylabel("Accuracy")
    axs[3].set_ylabel("Entropy")
    axs[3].set_xlabel("Time step (samples)")

    # Add grid to all subplots
    for ax in axs:
        ax.grid(True, alpha=0.3)

    # Create legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))

    # Create a custom legend for the segment colors
    segment_legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=segment_colors.get(s, "lightgray"), alpha=0.2)
        for s in sorted(set(segment_schedule))
    ]
    segment_labels = [s.capitalize() for s in sorted(set(segment_schedule))]

    # Add the custom legend to the figure
    leg2 = fig.legend(
        segment_legend_handles,
        segment_labels,
        loc="lower center",
        ncol=len(set(segment_schedule)),
        bbox_to_anchor=(0.5, -0.02),
        title="Segment Types",
    )
    fig.add_artist(leg2)

    # Adjust layout to make room for legends
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.subplots_adjust(hspace=0.15)

    # Show the plot
    plt.show()
