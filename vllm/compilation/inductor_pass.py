# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
<<<<<<< HEAD
import types
from abc import ABC, abstractmethod
=======
import json
import types
from contextlib import contextmanager
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from typing import Any, Callable, Optional, Union

import torch
from torch import fx

<<<<<<< HEAD

class InductorPass(ABC):
    """
    General custom inductor pass interface.
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.Graph):
        """
        Execute the pass on the given graph.
        """
        raise NotImplementedError
=======
from vllm.utils import is_torch_equal_or_newer

if is_torch_equal_or_newer("2.6"):
    from torch._inductor.custom_graph_pass import CustomGraphPass
else:
    # CustomGraphPass is not present in 2.5 or lower, import our version
    from .torch25_custom_graph_pass import (  # noqa: E501
        Torch25CustomGraphPass as CustomGraphPass)

_pass_context = None


class PassContext:

    def __init__(self, runtime_shape: Optional[int]):
        self.runtime_shape = runtime_shape


def get_pass_context() -> PassContext:
    """Get the current pass context."""
    assert _pass_context is not None
    return _pass_context


@contextmanager
def pass_context(runtime_shape: Optional[int]):
    """A context manager that stores the current pass context,
    usually it is a list of sizes to specialize.
    """
    global _pass_context
    prev_context = _pass_context
    _pass_context = PassContext(runtime_shape)
    try:
        yield
    finally:
        _pass_context = prev_context


class InductorPass(CustomGraphPass):
    """
    A custom graph pass that uses a hash of its source as the UUID.
    This is defined as a convenience and should work in most cases.
    """
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation.
        By default, the object source is hashed.
        """
        return InductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: Union[str, Any]):
        """
        Utility method to hash the sources of functions or objects.
        :param srcs: strings or objects to add to the hash.
        Objects and functions have their source inspected.
        :return:
        """
        hasher = hashlib.sha256()
        for src in srcs:
            if isinstance(src, str):
                src_str = src
            elif isinstance(src, types.FunctionType):
                src_str = inspect.getsource(src)
            else:
                src_str = inspect.getsource(src.__class__)
            hasher.update(src_str.encode("utf-8"))
<<<<<<< HEAD
        return hasher.digest()
=======
        return hasher.hexdigest()

    @staticmethod
    def hash_dict(dict_: dict[Any, Any]):
        """
        Utility method to hash a dictionary, can alternatively be used for uuid.
        :return: A sha256 hash of the json rep of the dictionary.
        """
        encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def is_applicable_for_shape(self, shape: Optional[int]):
        return True
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class CallableInductorPass(InductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(self,
                 callable: Callable[[fx.Graph], None],
                 uuid: Optional[Any] = None):
        self.callable = callable
<<<<<<< HEAD
        if uuid is None:
            uuid = InductorPass.hash_source(callable)
        self._uuid = uuid
=======
        self._uuid = self.hash_source(callable) if uuid is None else uuid
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def __call__(self, graph: torch.fx.Graph):
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid
<<<<<<< HEAD

    def __getstate__(self):
        """
        Pickling occurs in the Inductor code cache if a pass is not given to
        the pass manager but is instead directly added to config as a pass.
        See PostGradPassManager for more.

        TODO(torch==2.6), use the `uuid` method in CustomGraphPass instead.
        """
        return self._uuid

    def __setstate__(self, state):
        raise ValueError("Cannot unpickle CallableInductorPass")
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
