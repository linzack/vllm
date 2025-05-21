# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
<<<<<<< HEAD
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
=======
from typing import TYPE_CHECKING, Any, Optional, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.sequence import Logprob

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens includes the prompt.
<<<<<<< HEAD
    tokens: List[int]
    logprobs: List[Dict[int, Logprob]]
=======
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    cum_logprob: float = 0.0
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
<<<<<<< HEAD
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
=======
    mm_processor_kwargs: Optional[dict[str, Any]] = None
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
<<<<<<< HEAD
    sequences: List[BeamSearchSequence]
=======
    sequences: list[BeamSearchSequence]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class BeamSearchInstance:

<<<<<<< HEAD
    def __init__(self, prompt_tokens: List[int]):
        self.beams: List[BeamSearchSequence] = [
            BeamSearchSequence(tokens=prompt_tokens, logprobs=[])
        ]
        self.completed: List[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: List[int],
=======
    def __init__(
        self,
        prompt_tokens: list[int],
        logprobs: Optional[list[dict[int, Logprob]]] = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []


def get_beam_search_score(
    tokens: list[int],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score with length penalty.

    Adapted from

    https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
    """
    seq_len = len(tokens)
    if tokens[-1] == eos_token_id:
        seq_len -= 1

    return cumulative_logprob / (seq_len**length_penalty)


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):

    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(x.tokens, x.cum_logprob, eos_token_id,
                                     length_penalty)

    return sort_beams_key
