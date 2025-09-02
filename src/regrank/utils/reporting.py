# regrank/utils/reporting.py

import textwrap

import numpy as np

from .colors import reverse_dict  # Note the relative import


def print_summary_table(summary, max_width=40):
    """
    Prints a formatted summary table of ranking clusters to the console.

    Since prettytable is not a standard dependency, this uses f-strings.
    """
    rev_key_id2clsr_id = reverse_dict(summary["keyid2clusterid"])

    # Define table headers and separators
    header = f"| {'Group':<7} | {'#Tags':>5} | {'#Nodes':>7} | {'Members':<{max_width}} | {'Mean':>8} | {'Std':>10} |"
    separator = "-" * len(header)

    print(separator)
    print(header)
    print(separator)

    sorted_cluster_ids = sorted(rev_key_id2clsr_id.keys())

    for cluster_id in sorted_cluster_ids:
        key_ids = rev_key_id2clsr_id[cluster_id]
        members = ", ".join([summary[key_id][0] for key_id in key_ids])
        counts = sum([summary[key_id][1] for key_id in key_ids])
        sizes = len(key_ids)

        # Get the correct cluster data for mean/std calculation
        cluster_data = summary["avg_clusters"][cluster_id]
        m = np.mean(cluster_data)
        s = np.std(cluster_data)

        # Wrap member text for clean display
        wrapped_members = textwrap.wrap(members, width=max_width)

        # Print first line with all data
        first_line = wrapped_members[0] if wrapped_members else ""
        print(
            f"| {cluster_id + 1:<7} | {sizes:5d} | {counts:7d} | {first_line:<{max_width}} | {m:8.3f} | {s:10.1e} |"
        )

        # Print subsequent lines for wrapped members
        for line in wrapped_members[1:]:
            print(
                f"| {'':<7} | {'':>5} | {'':>7} | {line:<{max_width}} | {'':>8} | {'':>10} |"
            )

    print(separator)
