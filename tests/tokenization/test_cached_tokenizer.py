# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD

from copy import deepcopy

from transformers import AutoTokenizer

from vllm.transformers_utils.tokenizer import get_cached_tokenizer


def test_cached_tokenizer():
    reference_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reference_tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    reference_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<SEP>"]})
    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))

    assert reference_tokenizer.encode("prompt") == cached_tokenizer.encode(
        "prompt")
    assert set(reference_tokenizer.all_special_ids) == set(
        cached_tokenizer.all_special_ids)
    assert set(reference_tokenizer.all_special_tokens) == set(
        cached_tokenizer.all_special_tokens)
    assert set(reference_tokenizer.all_special_tokens_extended) == set(
        cached_tokenizer.all_special_tokens_extended)
=======
import pickle
from copy import deepcopy

import pytest
from transformers import AutoTokenizer

from vllm.transformers_utils.tokenizer import (AnyTokenizer,
                                               get_cached_tokenizer)


@pytest.mark.parametrize("model_id", ["gpt2", "THUDM/chatglm3-6b"])
def test_cached_tokenizer(model_id: str):
    reference_tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                        trust_remote_code=True)
    reference_tokenizer.add_special_tokens({"cls_token": "<CLS>"})
    reference_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<SEP>"]})

    cached_tokenizer = get_cached_tokenizer(deepcopy(reference_tokenizer))
    _check_consistency(cached_tokenizer, reference_tokenizer)

    pickled_tokenizer = pickle.dumps(cached_tokenizer)
    unpickled_tokenizer = pickle.loads(pickled_tokenizer)
    _check_consistency(unpickled_tokenizer, reference_tokenizer)


def _check_consistency(target: AnyTokenizer, expected: AnyTokenizer):
    assert isinstance(target, type(expected))

    # Cached attributes
    assert target.all_special_ids == expected.all_special_ids
    assert target.all_special_tokens == expected.all_special_tokens
    assert (target.all_special_tokens_extended ==
            expected.all_special_tokens_extended)
    assert target.get_vocab() == expected.get_vocab()
    assert len(target) == len(expected)

    # Other attributes
    assert getattr(target, "padding_side",
                   None) == getattr(expected, "padding_side", None)

    assert target.encode("prompt") == expected.encode("prompt")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
