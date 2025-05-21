# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
<<<<<<< HEAD
from typing import Tuple
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@dataclass
class AdapterMapping:
    # Per every token in input_ids:
<<<<<<< HEAD
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]
=======
    index_mapping: tuple[int, ...]
    # Per sampled token:
    prompt_mapping: tuple[int, ...]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)