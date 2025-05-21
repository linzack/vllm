# SPDX-License-Identifier: Apache-2.0

import os
import sys
import zipfile

# Read the VLLM_MAX_SIZE_MB environment variable, defaulting to 400 MiB
# Note that we have 400 MiB quota, please use it wisely.
# See https://github.com/pypi/support/issues/3792 .
# Please also sync the value with the one in Dockerfile.
<<<<<<< HEAD
VLLM_MAX_SIZE_MB = int(os.environ.get('VLLM_MAX_SIZE_MB', 400))
=======
VLLM_MAX_SIZE_MB = int(os.environ.get("VLLM_MAX_SIZE_MB", 400))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


def print_top_10_largest_files(zip_file):
    """Print the top 10 largest files in the given zip file."""
<<<<<<< HEAD
    with zipfile.ZipFile(zip_file, 'r') as z:
=======
    with zipfile.ZipFile(zip_file, "r") as z:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        file_sizes = [(f, z.getinfo(f).file_size) for f in z.namelist()]
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        for f, size in file_sizes[:10]:
            print(f"{f}: {size / (1024 * 1024):.2f} MBs uncompressed.")


def check_wheel_size(directory):
    """Check the size of .whl files in the given directory."""
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(".whl"):
                wheel_path = os.path.join(root, file_name)
                wheel_size_mb = os.path.getsize(wheel_path) / (1024 * 1024)
                if wheel_size_mb > VLLM_MAX_SIZE_MB:
<<<<<<< HEAD
                    print(f"Not allowed: Wheel {wheel_path} is larger "
                          f"({wheel_size_mb:.2f} MB) than the limit "
                          f"({VLLM_MAX_SIZE_MB} MB).")
                    print_top_10_largest_files(wheel_path)
                    return 1
                else:
                    print(f"Wheel {wheel_path} is within the allowed size "
                          f"({wheel_size_mb:.2f} MB).")
=======
                    print(
                        f"Not allowed: Wheel {wheel_path} is larger "
                        f"({wheel_size_mb:.2f} MB) than the limit "
                        f"({VLLM_MAX_SIZE_MB} MB)."
                    )
                    print_top_10_largest_files(wheel_path)
                    return 1
                else:
                    print(
                        f"Wheel {wheel_path} is within the allowed size "
                        f"({wheel_size_mb:.2f} MB)."
                    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check-wheel-size.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
<<<<<<< HEAD
    sys.exit(check_wheel_size(directory))
=======
    sys.exit(check_wheel_size(directory))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
