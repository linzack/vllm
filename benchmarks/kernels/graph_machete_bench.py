# SPDX-License-Identifier: Apache-2.0

import math
import pickle
import re
from collections import defaultdict
<<<<<<< HEAD
from typing import List
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torch.utils.benchmark import Measurement as TMeasurement

from vllm.utils import FlexibleArgumentParser

if __name__ == "__main__":
    parser = FlexibleArgumentParser(
<<<<<<< HEAD
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('filename', type=str)

    args = parser.parse_args()

    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
        raw_results: List[TMeasurement] = data["results"]
=======
        description="Benchmark the latency of processing a single batch of "
        "requests till completion."
    )
    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    with open(args.filename, "rb") as f:
        data = pickle.load(f)
        raw_results: list[TMeasurement] = data["results"]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    results = defaultdict(lambda: list())
    for v in raw_results:
        result = re.search(r"MKN=\(\d+x(\d+x\d+)\)", v.task_spec.sub_label)
        if result is not None:
            KN = result.group(1)
        else:
            raise Exception("MKN not found")
        result = re.search(r"MKN=\((\d+)x\d+x\d+\)", v.task_spec.sub_label)
        if result is not None:
            M = result.group(1)
        else:
            raise Exception("MKN not found")

        kernel = v.task_spec.description
<<<<<<< HEAD
        results[KN].append({
            "kernel": kernel,
            "batch_size": M,
            "median": v.median
        })
=======
        results[KN].append({"kernel": kernel, "batch_size": M, "median": v.median})
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    rows = int(math.ceil(len(results) / 2))
    fig, axs = plt.subplots(rows, 2, figsize=(12, 5 * rows))
    axs = axs.flatten()
    for axs_idx, (shape, data) in enumerate(results.items()):
        plt.sca(axs[axs_idx])
        df = pd.DataFrame(data)
<<<<<<< HEAD
        sns.lineplot(data=df,
                     x="batch_size",
                     y="median",
                     hue="kernel",
                     style="kernel",
                     markers=True,
                     dashes=False,
                     palette="Dark2")
=======
        sns.lineplot(
            data=df,
            x="batch_size",
            y="median",
            hue="kernel",
            style="kernel",
            markers=True,
            dashes=False,
            palette="Dark2",
        )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        plt.title(f"Shape: {shape}")
        plt.ylabel("time (median, s)")
    plt.tight_layout()
    plt.savefig("graph_machete_bench.pdf")
