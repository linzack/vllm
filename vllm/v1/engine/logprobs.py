# SPDX-License-Identifier: Apache-2.0

import itertools
<<<<<<< HEAD
from dataclasses import dataclass
from typing import Dict, List, Optional
=======
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.logger import init_logger
from vllm.sequence import Logprob, PromptLogprobs, SampleLogprobs
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_ids_list_to_tokens)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

logger = init_logger(__name__)

<<<<<<< HEAD
=======
NONES = itertools.repeat(None)

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

@dataclass
class LogprobsProcessor:

<<<<<<< HEAD
    # Tokenizer for this request
    tokenizer: AnyTokenizer
=======
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Optional[AnyTokenizer]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    # Logprobs for this request
    logprobs: Optional[SampleLogprobs]
    prompt_logprobs: Optional[PromptLogprobs]
    cumulative_logprob: Optional[float]
    num_logprobs: Optional[int]
    num_prompt_logprobs: Optional[int]

    @classmethod
    def from_new_request(
        cls,
<<<<<<< HEAD
        tokenizer: AnyTokenizer,
=======
        tokenizer: Optional[AnyTokenizer],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        num_logprobs = request.sampling_params.logprobs
        num_prompt_logprobs = request.sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.),
            logprobs=(None if num_logprobs is None else []),
            # NOTE: logprob of first prompt token is None.
            prompt_logprobs=(None if num_prompt_logprobs is None else [None]),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst,
                                             token_ids_lst):

            # Detokenize (non-incrementally).
<<<<<<< HEAD
            decoded_tokens = convert_ids_list_to_tokens(
                self.tokenizer, token_ids)
=======
            decoded_tokens = NONES if self.tokenizer is None else (
                convert_ids_list_to_tokens(self.tokenizer, token_ids))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob dictionary for this pos.
            self.logprobs.append(
                self._make_logprob_dict(
                    logprobs,
                    token_ids,
                    decoded_tokens,
                    rank,
                    self.num_logprobs,
                ))

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
<<<<<<< HEAD
        decoded_tokens = convert_ids_list_to_tokens(
            self.tokenizer,
            token_ids.flatten().tolist())
=======
        decoded_tokens = None if self.tokenizer is None else (
            convert_ids_list_to_tokens(self.tokenizer,
                                       token_ids.flatten().tolist()))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
<<<<<<< HEAD
        # TODO(rob): experiment with doing this in EngineCore?
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
<<<<<<< HEAD
            decoded_tokens_for_pos = decoded_tokens[offset:offset_end]
=======
            decoded_tokens_for_pos = NONES \
            if decoded_tokens is None else decoded_tokens[offset:offset_end]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

            # Update with the Logprob dictionary for this pos.
            self.prompt_logprobs.append(
                self._make_logprob_dict(prompt_logprobs[pos], token_ids[pos],
                                        decoded_tokens_for_pos,
                                        prompt_token_ranks[pos],
                                        self.num_prompt_logprobs))

    def pop_prompt_logprobs(self) -> Optional[PromptLogprobs]:
        """Pop and return all request prompt logprobs
        
        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _make_logprob_dict(
<<<<<<< HEAD
        logprobs: List[float],
        logprob_token_ids: List[int],
        decoded_tokens: List[str],
        rank: int,
        num_logprobs: int,
    ) -> Dict[int, Logprob]:
=======
        logprobs: list[float],
        logprob_token_ids: list[int],
        decoded_tokens: Iterable[Optional[str]],
        rank: int,
        num_logprobs: int,
    ) -> dict[int, Logprob]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        """Make a Logprob dictionary for a position.

        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)

        Returns:
<<<<<<< HEAD
          Dict[token id, Logprob]
=======
          dict[token id, Logprob]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        """

        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank, ), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(
                logprob_token_ids, logprobs, ranks, decoded_tokens)
        }

    def update_from_output(self, output: EngineCoreOutput) -> None:
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)
