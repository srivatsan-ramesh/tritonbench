import os

import matplotlib.pyplot as plt

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_values(filename):
    with open(f"{script_dir}/util_results/{filename}.txt", "r") as f:
        lines = f.read().split("\n")
        values = [
            [float(val) for val in line.split(",")] for line in lines if line != ""
        ]
        return values


values = get_values("warp_4")
# Example data (blue => x, orange => y)
blue_values_1 = values[0]
orange_values_1 = values[1]

values = get_values("warp_8")
blue_values_2 = values[0]
orange_values_2 = values[1]

values = get_values("warp_16")
blue_values_3 = values[0]
orange_values_3 = values[1]

values = get_values("warp_32")
blue_values_4 = values[0]
orange_values_4 = values[1]

plt.figure(figsize=(6, 6))

# Plot each pair (blue[i], orange[i]) on a 100x100 grid
plt.scatter(blue_values_1, orange_values_1, color="black", label="Warp 4")
plt.scatter(blue_values_2, orange_values_2, color="green", label="Warp 8")
plt.scatter(blue_values_3, orange_values_3, color="orange", label="Warp 16")
plt.scatter(blue_values_4, orange_values_4, color="blue", label="Warp 32")

x = np.linspace(0, 100, 100)
plt.plot(x, x, "r--", label="x = y")

# Constrain the axes to 0..100
plt.xlim(0, 100)
plt.ylim(0, 100)

plt.legend()
plt.xlabel("NCU")
plt.ylabel("Intra Kernel Profiler")

plt.tight_layout()
plt.savefig(f"{script_dir}/all_tensor_utils_comp.png")
