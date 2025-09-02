# regrank/utils/plotting.py

import matplotlib.pyplot as plt
import numpy as np

from .colors import generate_adjacent_colors, generate_complementary_colors


def plot_hist(summary, bin_count=30, legend=True, saveto=None):
    """
    Generates and displays a histogram of rankings, grouped by clusters.
    """
    plt.style.use("seaborn-v0_8-whitegrid")  # A modern and clean style
    fig, ax = plt.subplots(figsize=(8, 5))

    data_goi = summary["goi"]
    cluster_colors = generate_complementary_colors(len(summary["avg_clusters"]))
    _cluster_colors = {}

    for idx, cluster in enumerate(summary["avg_clusters"]):
        ax.axvline(
            np.mean(cluster),
            label=f"Group {idx + 1} Mean",
            color=cluster_colors[idx],
            linestyle="--",
            linewidth=2,
        )
        adj_colors = generate_adjacent_colors(cluster_colors[idx], k=len(cluster) + 1)
        _cluster_colors[idx] = iter(adj_colors)

    bins = np.histogram(summary["rankings"], bins=bin_count)[1]

    for idx, key in enumerate(data_goi.keys()):
        data = data_goi[key]
        cluster_id = summary["keyid2clusterid"][idx]
        color = next(_cluster_colors[cluster_id])
        ax.hist(
            data, bins, alpha=0.75, label=f"Tag: {key}", color=color, edgecolor="white"
        )

    ax.set_ylabel("Frequency")
    ax.set_xlabel("Rankings")
    ax.set_title("Distribution of Rankings by Group")

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")

    plt.tight_layout()

    if saveto:
        plt.savefig(saveto, bbox_inches="tight", dpi=300)

    plt.show()
