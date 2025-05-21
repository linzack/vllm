# SPDX-License-Identifier: Apache-2.0

import pickle
<<<<<<< HEAD
from typing import TYPE_CHECKING, Iterable, Mapping, Optional
=======
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import numpy as np
import torch
from blake3 import blake3
from PIL import Image

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.inputs import TokensPrompt

logger = init_logger(__name__)

MultiModalHashDict = Mapping[str, list[str]]
"""
A dictionary containing hashes for items in each modality.
"""


class MultiModalHasher:

    @classmethod
    def serialize_item(cls, obj: object) -> bytes:
        # Simple cases
        if isinstance(obj, str):
            return obj.encode("utf-8")
        if isinstance(obj, bytes):
            return obj
<<<<<<< HEAD
        if isinstance(obj, Image.Image):
            return obj.tobytes()

        # Convertible to NumPy arrays
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()
        if isinstance(obj, (int, float)):
            obj = np.array(obj)
        if isinstance(obj, np.ndarray):
            return obj.tobytes()
=======
        if isinstance(obj, (int, float)):
            return np.array(obj).tobytes()

        if isinstance(obj, Image.Image):
            return cls.item_to_bytes("image", np.array(obj.convert("RGBA")))
        if isinstance(obj, torch.Tensor):
            return cls.item_to_bytes("tensor", obj.numpy())
        if isinstance(obj, np.ndarray):
            return cls.item_to_bytes(
                "ndarray", {
                    "dtype": obj.dtype.str,
                    "shape": obj.shape,
                    "data": obj.data.tobytes(),
                })
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        logger.warning(
            "No serialization method found for %s. "
            "Falling back to pickle.", type(obj))

        return pickle.dumps(obj)

    @classmethod
    def item_to_bytes(
        cls,
        key: str,
        obj: object,
<<<<<<< HEAD
=======
    ) -> bytes:
        return b''.join(kb + vb for kb, vb in cls.iter_item_to_bytes(key, obj))

    @classmethod
    def iter_item_to_bytes(
        cls,
        key: str,
        obj: object,
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    ) -> Iterable[tuple[bytes, bytes]]:
        # Recursive cases
        if isinstance(obj, (list, tuple)):
            for i, elem in enumerate(obj):
<<<<<<< HEAD
                yield from cls.item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.item_to_bytes(f"{key}.{k}", v)
=======
                yield from cls.iter_item_to_bytes(f"{key}.{i}", elem)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                yield from cls.iter_item_to_bytes(f"{key}.{k}", v)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        else:
            key_bytes = cls.serialize_item(key)
            value_bytes = cls.serialize_item(obj)
            yield key_bytes, value_bytes

    @classmethod
    def hash_kwargs(cls, **kwargs: object) -> str:
        hasher = blake3()

        for k, v in kwargs.items():
<<<<<<< HEAD
            for k_bytes, v_bytes in cls.item_to_bytes(k, v):
=======
            for k_bytes, v_bytes in cls.iter_item_to_bytes(k, v):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
                hasher.update(k_bytes)
                hasher.update(v_bytes)

        return hasher.hexdigest()

    @classmethod
    def hash_prompt_mm_data(
            cls, prompt: "TokensPrompt") -> Optional["MultiModalHashDict"]:
        """Hash multimodal data in the user input prompt if they exist."""

        if "multi_modal_data" not in prompt:
            return None

        mm_data = prompt["multi_modal_data"]
        if not mm_data:
            # mm_data can be None or an empty dict.
            return None

        mm_items = {
            modality: items if isinstance(items, list) else [items]
            for modality, items in mm_data.items()
        }

        mm_hashes = {
            modality: [cls.hash_kwargs(**{modality: item}) for item in items]
            for modality, items in mm_items.items()
        }

        return mm_hashes
