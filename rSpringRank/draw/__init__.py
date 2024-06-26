__all__ = ["compute_data_goi", "plot_hist"]

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import random
import randomcolor
from .utils import cluster_1d_array


def compute_data_goi(g, goi, sslc=None, dual_v=None, primal_s=None):
    if dual_v is not None and primal_s is not None:
        raise AttributeError("Only use either dual_v or primal_s.")
    elif dual_v is None and primal_s is None:
        raise AttributeError("You need some input data.")
    elif dual_v is not None:
        # We take firstOrderMethods.py output directly
        dual_v = np.array(dual_v).reshape(-1, 1)
        output = sslc.dual2primal(dual_v)
    else:
        output = primal_s
    node_metadata = np.array(list(g.vp[goi]))
    data_goi = defaultdict(list)
    for idx, _c in enumerate(node_metadata):
        data_goi[_c].append(output[idx])
    return data_goi


def plot_hist(data_goi, bin_count=20, goi2text=None, legend=True, saveto=None):
    rand_color = randomcolor.RandomColor()
    if goi2text is None:
        goi2text = {}
        for key in data_goi.keys():
            goi2text[key] = [str(key), "#" + "%06x" % random.randint(0, 0xFFFFFF)]
    hstack = []
    for key in data_goi.keys():
        hstack.append(data_goi[key])

    bins = np.histogram(np.concatenate(hstack), bins=bin_count)[1]

    diff_avgs = []
    for key in data_goi.keys():
        diff_avgs.append(np.mean(data_goi[key]))
    # print(diff_avgs)
    avg_clusters, _dummy_map = cluster_1d_array(diff_avgs)
    cluster_colors = rand_color.generate(count=len(avg_clusters))
    _cluster_colors = dict()
    for idx, cluster in enumerate(avg_clusters):
        plt.axvline(
            np.mean(cluster),
            label=f"Size: {len(cluster)}, avg: {np.mean(cluster):.2f}, var: {np.var(cluster):.1e}",
            color=cluster_colors[idx],
        )
        adj_colors = rand_color.generate(hue=cluster_colors[idx], count=len(cluster))
        # print(f"adj_colors = {adj_colors}")
        _cluster_colors[idx] = iter(adj_colors)
    # print(_cluster_colors)

    for idx, key in enumerate(data_goi.keys()):
        data = data_goi[key]
        # print(np.mean(data))
        # plt.axvline(
        #     np.mean(data),
        #     label=goi2text[str(key)][0],
        #     color=goi2text[str(key)][1],
        # )
        # diff_avgs.append(np.mean(data))
        # kernel = gaussian_kde(data)
        # plt.plot(bins, kernel(bins), color=c18toMEANING[str(key)][1])
        c = next(_cluster_colors[_dummy_map[idx]])
        # print(c)
        plt.hist(
            data,
            bins,
            alpha=0.8,
            edgecolor="white",
            linewidth=1.2,
            color=c,
            zorder=np.mean(data),
            density=False,
        )

    plt.rcParams["figure.figsize"] = [7, 4]  # or 7, 4 or 10,8
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 4
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({"font.size": 14})
    plt.rcParams["font.family"] = "Helvetica"

    plt.tick_params(axis="y", direction="in", length=5, which="both")
    plt.tick_params(axis="x", direction="in", length=5, which="both")

    plt.ylabel("Frequency")
    plt.xlabel("SpringRank")
    if legend:
        plt.legend()
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
