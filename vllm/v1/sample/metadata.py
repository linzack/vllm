# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
<<<<<<< HEAD
from typing import Dict, List, Optional, Set, Tuple
=======
from typing import Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch


@dataclass
class SamplingMetadata:

    temperature: Optional[torch.Tensor]
    all_greedy: bool
    all_random: bool

<<<<<<< HEAD
    # None when there are no speculated tokens.
    spec_token_ids: Optional[List[List[int]]]

=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    top_p: Optional[torch.Tensor]
    top_k: Optional[torch.Tensor]
    min_p: Optional[torch.Tensor]

<<<<<<< HEAD
    generators: Dict[int, torch.Generator]
=======
    generators: dict[int, torch.Generator]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: Optional[int]

    no_penalties: bool
    prompt_token_ids: Optional[torch.Tensor]
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor

<<<<<<< HEAD
    output_token_ids: List[List[int]]

    # req_index -> (min_tokens, stop_token_ids)
    min_tokens: Dict[int, Tuple[int, Set[int]]]

    logit_bias: List[Optional[Dict[int, float]]]
=======
    output_token_ids: list[list[int]]

    # req_index -> (min_tokens, stop_token_ids)
    min_tokens: dict[int, tuple[int, set[int]]]

    logit_bias: list[Optional[dict[int, float]]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    # `allowed_token_ids_mask` is a 2D bool tensor of shape (max batch size,
    # vocab size).
    allowed_token_ids_mask: Optional[torch.Tensor]
<<<<<<< HEAD
=======

    # req_index -> bad_words_token_ids
    bad_words_token_ids: dict[int, list[list[int]]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
