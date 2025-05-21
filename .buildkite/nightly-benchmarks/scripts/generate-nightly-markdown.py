# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate


def parse_arguments():
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description=
        'Parse command line arguments for summary-nightly-results script.')
    parser.add_argument('--results-folder',
                        type=str,
                        required=True,
                        help='The folder where the results are stored.')
    parser.add_argument('--description',
                        type=str,
                        required=True,
                        help='Description of the results.')
=======
        description="Parse command line arguments for summary-nightly-results script."
    )
    parser.add_argument(
        "--results-folder",
        type=str,
        required=True,
        help="The folder where the results are stored.",
    )
    parser.add_argument(
        "--description", type=str, required=True, help="Description of the results."
    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    args = parser.parse_args()
    return args


def get_perf(df, method, model, metric):
<<<<<<< HEAD

    means = []

    for qps in [2, 4, 8, 16, "inf"]:
        target = df['Test name'].str.contains(model)
        target = target & df['Engine'].str.contains(method)
        target = target & df['Test name'].str.contains("qps_" + str(qps))
        filtered_df = df[target]

        if filtered_df.empty:
            means.append(0.)
=======
    means = []

    for qps in [2, 4, 8, 16, "inf"]:
        target = df["Test name"].str.contains(model)
        target = target & df["Engine"].str.contains(method)
        target = target & df["Test name"].str.contains("qps_" + str(qps))
        filtered_df = df[target]

        if filtered_df.empty:
            means.append(0.0)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        else:
            means.append(filtered_df[metric].values[0])

    return np.array(means)


def get_perf_w_std(df, method, model, metric):
<<<<<<< HEAD

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if metric in ["TTFT", "ITL"]:
        mean = get_perf(df, method, model, "Mean " + metric + " (ms)")
        mean = mean.tolist()
        std = get_perf(df, method, model, "Std " + metric + " (ms)")
        if std.mean() == 0:
            std = None
        success = get_perf(df, method, model, "Successful req.")
        if std is not None:
            std = std / np.sqrt(success)
            std = std.tolist()

    else:
        assert metric == "Tput"
        mean = get_perf(df, method, model, "Input Tput (tok/s)") + get_perf(
<<<<<<< HEAD
            df, method, model, "Output Tput (tok/s)")
=======
            df, method, model, "Output Tput (tok/s)"
        )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        mean = mean.tolist()
        std = None

    return mean, std


def main(args):
    results_folder = Path(args.results_folder)

    results = []

    # collect results
    for test_file in results_folder.glob("*_nightly_results.json"):
        with open(test_file) as f:
            results = results + json.loads(f.read())

    # generate markdown table
    df = pd.DataFrame.from_dict(results)

<<<<<<< HEAD
    md_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
=======
    md_table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    with open(args.description) as f:
        description = f.read()

<<<<<<< HEAD
    description = description.format(
        nightly_results_benchmarking_table=md_table)
=======
    description = description.format(nightly_results_benchmarking_table=md_table)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    with open("nightly_results.md", "w") as f:
        f.write(description)


<<<<<<< HEAD
if __name__ == '__main__':
=======
if __name__ == "__main__":
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    args = parse_arguments()
    main(args)
