# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from importlib.util import find_spec

from vllm.logger import init_logger
from vllm.platforms import current_platform
=======
import types
from importlib.util import find_spec

from vllm.logger import init_logger
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

logger = init_logger(__name__)

HAS_TRITON = (
    find_spec("triton") is not None
<<<<<<< HEAD
    and not current_platform.is_xpu()  # Not compatible
=======
    or find_spec("pytorch-triton-xpu") is not None  # Not compatible
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
)

if not HAS_TRITON:
    logger.info("Triton not installed or not compatible; certain GPU-related"
                " functions will not be available.")
<<<<<<< HEAD
=======


class TritonPlaceholder(types.ModuleType):

    def __init__(self):
        super().__init__("triton")
        self.jit = self._dummy_decorator("jit")
        self.autotune = self._dummy_decorator("autotune")
        self.heuristics = self._dummy_decorator("heuristics")
        self.language = TritonLanguagePlaceholder()
        logger.warning_once(
            "Triton is not installed. Using dummy decorators. "
            "Install it via `pip install triton` to enable kernel"
            " compilation.")

    def _dummy_decorator(self, name):

        def decorator(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda f: f

        return decorator


class TritonLanguagePlaceholder(types.ModuleType):

    def __init__(self):
        super().__init__("triton.language")
        self.constexpr = None
        self.dtype = None
        self.int64 = None
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
