# SPDX-License-Identifier: Apache-2.0
"""Make sure bad_words works.

Run `pytest tests/samplers/test_no_bad_words.py`.

"""
<<<<<<< HEAD
from typing import List, Optional

=======
from typing import Optional

import pytest
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


<<<<<<< HEAD
=======
@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    """We can run both engines for this test."""
    pass


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def _generate(
    model: LLM,
    prompt: str,
    num_prompt_tokens: int,
    temperature: float = 0,
<<<<<<< HEAD
    bad_words: Optional[List[str]] = None,
) -> List[int]:
=======
    bad_words: Optional[list[str]] = None,
) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    sampling_params = SamplingParams(
        temperature=temperature,
        bad_words=bad_words,
    )

    # [([output_token_ids, ], [output_text, ]), ]
    output = model.generate([prompt], sampling_params=sampling_params)

    output_token_ids = output[0][0][0][num_prompt_tokens:]
    # [0] first (and only) request output
    # [0] token_ids (not text)
    # [0] first (and only) output completion

    return output_token_ids


class TestOneTokenBadWord:
    MODEL = "TheBloke/Llama-2-7B-fp16"

    PROMPT = "Hi! How are"
    TARGET_TOKEN = "you"

    def setup_method(self, method):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL,
                                                       add_prefix_space=True)

        self.num_prompt_tokens = len(self._encode(self.PROMPT))
        self.target_token_id = self._encode(self.TARGET_TOKEN,
                                            add_special_tokens=False)[0]

    def test_one_token_bad_word(self, vllm_runner):
        with vllm_runner(self.MODEL) as llm:
            output_token_ids = self._generate(llm)
            assert output_token_ids[0] == self.target_token_id

            output_token_ids = self._generate(llm,
                                              bad_words=[self.TARGET_TOKEN])
            assert self.target_token_id not in output_token_ids

    def _generate(self,
                  model: LLM,
<<<<<<< HEAD
                  bad_words: Optional[List[str]] = None) -> List[int]:
=======
                  bad_words: Optional[list[str]] = None) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return _generate(
            model=model,
            prompt=self.PROMPT,
            num_prompt_tokens=self.num_prompt_tokens,
            bad_words=bad_words,
        )

    def _encode(self,
                prompt: str,
<<<<<<< HEAD
                add_special_tokens: bool = True) -> List[int]:
=======
                add_special_tokens: bool = True) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return self.tokenizer(prompt,
                              add_special_tokens=add_special_tokens).input_ids


class TestTwoTokenBadWord:
    # Another model (with a different tokenizer behaviour)
    MODEL = "distilbert/distilgpt2"

    PROMPT = "How old are you? I am 10"
    TARGET_TOKEN1 = "years"
    TARGET_TOKEN2 = "old"
    NEIGHBOUR_TOKEN2 = "older"

    def setup_method(self, method):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL,
                                                       add_prefix_space=True)

        self.num_prompt_tokens = len(self._encode(self.PROMPT))
        self.target_token_id1 = self._encode(self.TARGET_TOKEN1,
                                             add_special_tokens=False)[0]
        self.target_token_id2 = self._encode(self.TARGET_TOKEN2,
                                             add_special_tokens=False)[0]
        self.neighbour_token_id2 = self._encode(self.NEIGHBOUR_TOKEN2,
                                                add_special_tokens=False)[0]

    def test_two_token_bad_word(self, vllm_runner):
        with vllm_runner(self.MODEL) as llm:
            output_token_ids = self._generate(llm)
            assert output_token_ids[:2] == [
                self.target_token_id1, self.target_token_id2
            ]

            output_token_ids = self._generate(llm,
                                              bad_words=[self.TARGET_TOKEN1])
            assert self.target_token_id1 not in output_token_ids

            output_token_ids = self._generate(llm,
                                              bad_words=[self.TARGET_TOKEN2])
            assert output_token_ids[0] == self.target_token_id1
            assert self.target_token_id2 not in output_token_ids

            output_token_ids = self._generate(
                llm, bad_words=[f'{self.TARGET_TOKEN1} {self.TARGET_TOKEN2}'])
            assert output_token_ids[0] == self.target_token_id1
            assert output_token_ids[:2] != [
                self.target_token_id1, self.target_token_id2
            ]
            assert not self._contains(
                output_token_ids,
                [self.target_token_id1, self.target_token_id2])
            # Model dependent behaviour
            assert output_token_ids[:2] == [
                self.target_token_id1, self.neighbour_token_id2
            ]

            output_token_ids = self._generate(
                llm,
                bad_words=[
                    f'{self.TARGET_TOKEN1} {self.TARGET_TOKEN2}',
                    f'{self.TARGET_TOKEN1} {self.NEIGHBOUR_TOKEN2}'
                ])
            assert output_token_ids[0] == self.target_token_id1
            assert output_token_ids[:2] != [
                self.target_token_id1, self.target_token_id2
            ]
            assert not self._contains(
                output_token_ids,
                [self.target_token_id1, self.target_token_id2])
            assert output_token_ids[:2] != [
                self.target_token_id1, self.neighbour_token_id2
            ]
            assert not self._contains(
                output_token_ids,
                [self.target_token_id1, self.neighbour_token_id2])
            assert ((self.target_token_id2 in output_token_ids)
                    or (self.neighbour_token_id2 in output_token_ids))

    def _generate(self,
                  model: LLM,
<<<<<<< HEAD
                  bad_words: Optional[List[str]] = None) -> List[int]:
=======
                  bad_words: Optional[list[str]] = None) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return _generate(
            model=model,
            prompt=self.PROMPT,
            num_prompt_tokens=self.num_prompt_tokens,
            bad_words=bad_words,
        )

    @staticmethod
<<<<<<< HEAD
    def _contains(sequence: List[int], subsequence: List[int]) -> bool:
=======
    def _contains(sequence: list[int], subsequence: list[int]) -> bool:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        searched = False

        for start in range(len(sequence)):
            end = start + len(subsequence)
            current_subsequence = sequence[start:end]

            if len(current_subsequence) < len(subsequence):
                continue

            searched = True

            assert len(current_subsequence) == len(subsequence)

            if current_subsequence == subsequence:
                return True

        assert searched, "All subsequences did not match in length..."

        return False

    def _encode(self,
                prompt: str,
<<<<<<< HEAD
                add_special_tokens: bool = True) -> List[int]:
=======
                add_special_tokens: bool = True) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        return self.tokenizer(prompt,
                              add_special_tokens=add_special_tokens).input_ids
