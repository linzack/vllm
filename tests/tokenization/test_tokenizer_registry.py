# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
=======
from typing import TYPE_CHECKING, Any, Optional, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.transformers_utils.tokenizer_base import (TokenizerBase,
                                                    TokenizerRegistry)

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class TestTokenizer(TokenizerBase):

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "TestTokenizer":
        return TestTokenizer()

    @property
<<<<<<< HEAD
    def all_special_tokens_extended(self) -> List[str]:
        raise NotImplementedError()

    @property
    def all_special_tokens(self) -> List[str]:
        raise NotImplementedError()

    @property
    def all_special_ids(self) -> List[int]:
=======
    def all_special_tokens_extended(self) -> list[str]:
        raise NotImplementedError()

    @property
    def all_special_tokens(self) -> list[str]:
        raise NotImplementedError()

    @property
    def all_special_ids(self) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @property
    def bos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

    @property
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    def is_fast(self) -> bool:
        raise NotImplementedError()

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError()

    @property
    def max_token_id(self) -> int:
        raise NotImplementedError()

    def __call__(
        self,
<<<<<<< HEAD
        text: Union[str, List[str], List[int]],
=======
        text: Union[str, list[str], list[int]],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        text_pair: Optional[str] = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ):
        raise NotImplementedError()

<<<<<<< HEAD
    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    def get_added_vocab(self) -> Dict[str, int]:
=======
    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    def get_added_vocab(self) -> dict[str, int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    def encode_one(
        self,
        text: str,
        truncation: bool = False,
        max_length: Optional[int] = None,
<<<<<<< HEAD
    ) -> List[int]:
=======
    ) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    def encode(self,
               text: str,
<<<<<<< HEAD
               add_special_tokens: Optional[bool] = None) -> List[int]:
        raise NotImplementedError()

    def apply_chat_template(self,
                            messages: List["ChatCompletionMessageParam"],
                            tools: Optional[List[Dict[str, Any]]] = None,
                            **kwargs) -> List[int]:
        raise NotImplementedError()

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        raise NotImplementedError()

    def decode(self,
               ids: Union[List[int], int],
=======
               add_special_tokens: Optional[bool] = None) -> list[int]:
        raise NotImplementedError()

    def apply_chat_template(self,
                            messages: list["ChatCompletionMessageParam"],
                            tools: Optional[list[dict[str, Any]]] = None,
                            **kwargs) -> list[int]:
        raise NotImplementedError()

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        raise NotImplementedError()

    def decode(self,
               ids: Union[list[int], int],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
               skip_special_tokens: bool = True) -> str:
        raise NotImplementedError()

    def convert_ids_to_tokens(
        self,
<<<<<<< HEAD
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> List[str]:
=======
        ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()


def test_customized_tokenizer():
    TokenizerRegistry.register("test_tokenizer",
                               "tests.tokenization.test_tokenizer_registry",
                               "TestTokenizer")

    tokenizer = TokenizerRegistry.get_tokenizer("test_tokenizer")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1

    tokenizer = get_tokenizer("test_tokenizer", tokenizer_mode="custom")
    assert isinstance(tokenizer, TestTokenizer)
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 1
