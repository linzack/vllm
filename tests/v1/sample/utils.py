# SPDX-License-Identifier: Apache-2.0

import re
<<<<<<< HEAD
from typing import List, Tuple
=======
from enum import Enum
from typing import Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm import CompletionOutput


<<<<<<< HEAD
def get_test_batch(batch_logprobs_composition: str) -> List[Tuple]:
=======
class BatchLogprobsComposition(Enum):
    """Types of logprobs configs to include in test batch"""
    NONE = 0
    SAMPLE = 1
    PROMPT = 2
    SAMPLE_PROMPT = 3


BatchLogprobsSpecType = list[tuple[Optional[int], Optional[int]]]


def get_test_batch(
    batch_logprobs_composition: BatchLogprobsComposition
) -> BatchLogprobsSpecType:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Generate logprobs configs for a batch of requests
    
    A given request's logprobs configuration is (1) num_sample_logprobs and (2)
    num_prompt_logprobs. The batch logprobs configuration is the list of request
    logprobs configs.

<<<<<<< HEAD
    batch_logprobs_composition == "NONE" yields a batch with no sample or prompt
    logprobs

    batch_logprobs_composition == "SAMPLE" yields a batch with some requests
    configured for sample logprobs only, and others configured for no logprobs

    batch_logprobs_composition == "PROMPT" yields a batch with some requests
    configured for prompt logprobs only, and others configured for no logprobs

    batch_logprobs_composition == "SAMPLE_PROMPT" yields a batch with some
=======
    batch_logprobs_composition == NONE yields a batch with no sample or prompt
    logprobs

    batch_logprobs_composition == SAMPLE yields a batch with some requests
    configured for sample logprobs only, and others configured for no logprobs

    batch_logprobs_composition == PROMPT yields a batch with some requests
    configured for prompt logprobs only, and others configured for no logprobs

    batch_logprobs_composition == SAMPLE_PROMPT yields a batch with some
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    requests configured for sample logprobs and prompt logprobs, some configured
    for only sample logprobs or only prompt logprobs, and some configured for
    no logprobs

    Args:
      batch_logprobs_composition: types of logprobs configs to include in batch

    Returns:

<<<<<<< HEAD
      List of (Optional[num_sample_logprobs], Optional[num_prompt_logprobs])
      tuples
    """
    if batch_logprobs_composition == "NONE":
        # No requests with sample or prompt logprobs
        return [(None, None)]
    elif batch_logprobs_composition == "SAMPLE":
=======
      list of (Optional[num_sample_logprobs], Optional[num_prompt_logprobs])
      tuples
    """
    if batch_logprobs_composition == BatchLogprobsComposition.NONE:
        # No requests with sample or prompt logprobs
        return [(None, None)]
    elif batch_logprobs_composition == BatchLogprobsComposition.SAMPLE:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # Requests requiring sample logprobs or no logprobs
        return [
            (None, None),
            (0, None),
            (5, None),
            (3, None),
        ]
<<<<<<< HEAD
    elif batch_logprobs_composition == "PROMPT":
=======
    elif batch_logprobs_composition == BatchLogprobsComposition.PROMPT:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # Requests requiring prompt logprobs or no logprobs
        return [
            (None, None),
            (None, 0),
            (None, 6),
            (None, 5),
        ]
<<<<<<< HEAD
    elif batch_logprobs_composition == "SAMPLE_PROMPT":
=======
    elif batch_logprobs_composition == BatchLogprobsComposition.SAMPLE_PROMPT:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        # Requests requiring either no logprobs, just
        # sample logprobs, just prompt logprobs, or
        # both sample and prompt logprobs
        return [
            (None, None),
            (0, None),
            (5, None),
            (3, None),
            (0, 3),
            (6, 0),
            (6, 3),
            (None, 6),
            (None, 5),
            (None, 0),
        ]
    else:
        raise ValueError("Invalid logprobs batch configuration for test.")


def assert_incr_detok_str_matches_non_incr_detok_str(
    incremental_detokenization_str: str,
    non_incremental_detokenization_str: str,
    msg: str,
) -> None:
    """Compare incrementally detok. text to non-incrementally detok. text
    
    Fail if the strings mismatch after non-alphanumeric characters are stripped
    out.

    Rationale: incremental detokenization in the text generation process allows
    the tokenizer to adjust the next token text output based on the token's
    context in the string. However, logprobs detokenization detokenizes each
    token individually, and the resultant strings may include some
    non-alphanumeric placeholder characters where there could be i.e.
    whitespace. So, this function compares only the alphanumeric text
    between two strings and fails if there is a mismatch, which helps
    with validating logprobs detokenization.

    Args:
      incremental_detokenization_str: incrementally-detokenized generated text
      non_incremental_detokenization_str: non-incrementally-detokenized logprob
                                          tokens
      msg: error message if `assert` fails
    """
    rgx = r'[^a-zA-Z0-9]+'
    assert (re.sub(rgx, '', incremental_detokenization_str) == re.sub(
        rgx, '', non_incremental_detokenization_str)), (msg)


def compute_correct_cumulative_logprob(
        completion_output: CompletionOutput) -> float:
    """Compute known-good value for evaluating cumulative logprob
    
    Args:
      completion_output: completion output from engine

    Returns:
      Known-good cumulative logprob value
    """
    token_ids = completion_output.token_ids
    logprobs = completion_output.logprobs
    assert logprobs is not None
    return sum([lp[tok_id].logprob for tok_id, lp in zip(token_ids, logprobs)])
