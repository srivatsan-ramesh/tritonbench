import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--op", type=str, default="fp8_gemm")
parser.add_argument("--threadblock", type=int, default=0)
parser.add_argument("--warp", type=int, default=0)

args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
traces_dir = os.path.join(script_dir, args.op, "traces")

data = {}

for i, filename in enumerate(os.listdir(traces_dir)):
    with open(os.path.join(traces_dir, filename), "r") as f:
        d = json.load(f)
        for event in d["traceEvents"]:
            if (
                event["pid"] == f"threadblock {args.threadblock}"
                and event["tid"] == f"warpgroup {args.warp}"
            ):
                if event["name"] in data:
                    if i >= len(data[event["name"]]):
                        data[event["name"]].append([event["dur"]])
                    else:
                        data[event["name"]][i].append(event["dur"])
                else:
                    data[event["name"]] = [[event["dur"]]]

region_names = list(data.keys())
data_values = [np.sum(data[region], axis=1) for region in region_names]

fig, ax = plt.subplots(figsize=(8, 5))

# Create a boxplot for each region (whiskers go to min and max)
bp = ax.boxplot(
    data_values,
    tick_labels=["prologue", "stable", "epilogue"],
    whis=[0, 100],
    patch_artist=True,
)

for box in bp["boxes"]:
    box.set(facecolor="lightgray")

# Determine global min and max across all data
global_min = min(np.min(vals) for vals in data_values)
global_max = max(np.max(vals) for vals in data_values)

# Set the plot’s y-limit slightly above the global max so annotations don’t go off the top
upper_margin_factor = 1.1
ax.set_ylim(global_min * 0.9, global_max * upper_margin_factor)

for i, region in enumerate(region_names):
    values = np.array(np.sum(data[region], axis=1))
    mean_val = np.mean(values)
    std_val = np.std(values)

    # Coefficient of Variation (std / mean) * 100; handle zero-mean case
    if mean_val != 0:
        var_percent = (std_val / mean_val) * 100
    else:
        var_percent = 0.0

    # Position the text slightly above the region's maximum value
    max_val = np.max(values)
    offset = 5  # small vertical offset
    var_text_y = max_val + offset

    ax.text(
        i + 1,
        var_text_y,
        f"{var_percent:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        color="red",
    )

ax.set_xlabel("(M, N, K) = (1024, 1024, 512)")
ax.set_ylabel("Cycles")
ax.set_title("Boxplot per Region with Variance % (Std as % of Mean)")
ax.grid(True, axis="y")

plt.savefig(f"{script_dir}/{args.op}/clock_cycles_box_plot.png")
