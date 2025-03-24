import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from loop_experiment import main, RunType


script_dir = os.path.dirname(os.path.abspath(__file__))


def get_all_data_as_dict(traces_dir_name="traces"):
    """
    Returns a dictionary of the form: {event_name: list with shape (n_iter, n_tid, n_wid, n_val)}
    """
    traces_dir = os.path.join(script_dir, traces_dir_name)

    data = {}

    for i, filename in enumerate(os.listdir(traces_dir)):
        with open(os.path.join(traces_dir, filename), "r") as f:
            d = json.load(f)
            for event in d["traceEvents"]:
                if event["name"] not in data:
                    data[event["name"]] = []
                if i >= len(data[event["name"]]):
                    data[event["name"]].append([])
                tid = int(event["pid"].split(" ")[1])
                wid = int(event["tid"].split(" ")[1])
                if len(data[event["name"]][i]) <= tid:
                    data[event["name"]][i].extend(
                        [[]] * (tid - len(data[event["name"]][i]) + 1)
                    )
                if len(data[event["name"]][i][tid]) <= wid:
                    data[event["name"]][i][tid].extend(
                        [[]] * (wid - len(data[event["name"]][i][tid]) + 1)
                    )
                data[event["name"]][i][tid][wid].append(event["dur"])
    return data


def get_data_for_region(traces_dir="traces", region_name="region_0"):
    """
    Returns a numpy array of the form (n_iter, n_tid, n_wid, n_val)
    """
    return np.array(get_all_data_as_dict(traces_dir)[region_name])


def get_mean_of_sum(traces_dir="traces", region_name="region_0"):
    """
    Returns a value that sums up the values corresponding to a region and then takes the mean
    """
    return np.mean(np.sum(get_data_for_region(traces_dir, region_name), axis=3))


def get_tensor_core_util_from_proton(m, k, n):
    main(m, k, n, RunType.PASS1)
    main(m, k, n, RunType.PASS2)

    return get_mean_of_sum("traces", "region_0") / get_mean_of_sum(
        "traces_pass2", "region_0"
    )


def get_tensor_core_util_from_ncu(m, k, n):
    pass
