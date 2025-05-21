# SPDX-License-Identifier: Apache-2.0

import importlib
from abc import ABC, abstractmethod
<<<<<<< HEAD
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
=======
from typing import TYPE_CHECKING, Any, Optional, Union
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

if TYPE_CHECKING:
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam


class TokenizerBase(ABC):

    @property
    @abstractmethod
<<<<<<< HEAD
    def all_special_tokens_extended(self) -> List[str]:
=======
    def all_special_tokens_extended(self) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @property
    @abstractmethod
<<<<<<< HEAD
    def all_special_tokens(self) -> List[str]:
=======
    def all_special_tokens(self) -> list[str]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @property
    @abstractmethod
<<<<<<< HEAD
    def all_special_ids(self) -> List[int]:
=======
    def all_special_ids(self) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def sep_token(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def pad_token(self) -> str:
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_fast(self) -> bool:
        raise NotImplementedError()

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def max_token_id(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.vocab_size

    @abstractmethod
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

    @abstractmethod
<<<<<<< HEAD
    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_added_vocab(self) -> Dict[str, int]:
=======
    def get_vocab(self) -> dict[str, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_added_vocab(self) -> dict[str, int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @abstractmethod
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

    @abstractmethod
    def encode(self,
               text: str,
<<<<<<< HEAD
               add_special_tokens: Optional[bool] = None) -> List[int]:
=======
               truncation: Optional[bool] = None,
               max_length: Optional[int] = None,
               add_special_tokens: Optional[bool] = None) -> list[int]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @abstractmethod
    def apply_chat_template(self,
<<<<<<< HEAD
                            messages: List["ChatCompletionMessageParam"],
                            tools: Optional[List[Dict[str, Any]]] = None,
                            **kwargs) -> List[int]:
        raise NotImplementedError()

    @abstractmethod
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
=======
                            messages: list["ChatCompletionMessageParam"],
                            tools: Optional[list[dict[str, Any]]] = None,
                            **kwargs) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        raise NotImplementedError()

    @abstractmethod
    def decode(self,
<<<<<<< HEAD
               ids: Union[List[int], int],
=======
               ids: Union[list[int], int],
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
               skip_special_tokens: bool = True) -> str:
        raise NotImplementedError()

    @abstractmethod
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


class TokenizerRegistry:
    # Tokenizer name -> (tokenizer module, tokenizer class)
<<<<<<< HEAD
    REGISTRY: Dict[str, Tuple[str, str]] = {}
=======
    REGISTRY: dict[str, tuple[str, str]] = {}
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    @staticmethod
    def register(name: str, module: str, class_name: str) -> None:
        TokenizerRegistry.REGISTRY[name] = (module, class_name)

    @staticmethod
    def get_tokenizer(
        tokenizer_name: str,
        *args,
        **kwargs,
    ) -> TokenizerBase:
        tokenizer_cls = TokenizerRegistry.REGISTRY.get(tokenizer_name)
        if tokenizer_cls is None:
            raise ValueError(f"Tokenizer {tokenizer_name} not found.")

        tokenizer_module = importlib.import_module(tokenizer_cls[0])
        class_ = getattr(tokenizer_module, tokenizer_cls[1])
        return class_.from_pretrained(*args, **kwargs)
