# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
from typing import Any, Dict, List

import torch
from torch import fx as fx

from vllm.config import CompilationConfig
from vllm.logger import init_logger

from .fix_functionalization import FixFunctionalizationPass
from .fusion import FusionPass
from .inductor_pass import InductorPass
from .reshapes import RedundantReshapesPass
=======
from torch import fx as fx

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .activation_quant_fusion import ActivationQuantFusionPass
from .fix_functionalization import FixFunctionalizationPass
from .fusion import FusionPass
from .inductor_pass import CustomGraphPass, InductorPass, get_pass_context
from .noop_elimination import NoOpEliminationPass
from .sequence_parallelism import SequenceParallelismPass
from .vllm_inductor_pass import VllmInductorPass
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

logger = init_logger(__name__)


<<<<<<< HEAD
class PlaceHolder:
    pass


if torch.__version__ < "2.6":
    Parent = PlaceHolder  # type: ignore
else:
    Parent = torch._inductor.custom_graph_pass.CustomGraphPass  # type: ignore


class PostGradPassManager(Parent):
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It also supports pickling, which is used by the Inductor code cache.
    TODO(torch==2.6), use CustomGraphPass
    (torch._inductor.custom_graph_pass.CustomGraphPass)

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (RedundantReshapesPass, FusionPass)
=======
class PostGradPassManager(CustomGraphPass):
    """
    The pass manager for post-grad passes.
    It handles configuration, adding custom passes, and running passes.
    It supports uuid for the Inductor code cache. That includes torch<2.6
    support using pickling (in .inductor_pass.CustomGraphPass).

    The order of the post-grad post-passes is:
    1. passes (constructor parameter)
    2. default passes (NoopEliminationPass, FusionPass)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    3. config["post_grad_custom_post_pass"] (if it exists)
    4. fix_functionalization
    This way, all passes operate on a functionalized graph.
    """

    def __init__(self):
<<<<<<< HEAD
        self.passes: List[InductorPass] = []

    def __call__(self, graph: fx.Graph):
        for pass_ in self.passes:
            pass_(graph)
=======
        self.passes: list[VllmInductorPass] = []

    def __call__(self, graph: fx.Graph):
        shape = get_pass_context().runtime_shape
        for pass_ in self.passes:
            if pass_.is_applicable_for_shape(shape):
                pass_(graph)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

        # always run fix_functionalization last
        self.fix_functionalization(graph)

<<<<<<< HEAD
    def configure(self, pass_config: CompilationConfig.PassConfig):
        self.pass_config = pass_config
        if pass_config.enable_reshape:
            self.passes += [RedundantReshapesPass(pass_config)]

        if pass_config.enable_fusion:
            self.passes += [FusionPass.instance(pass_config)]

        self.fix_functionalization = FixFunctionalizationPass(pass_config)
=======
    def configure(self, config: VllmConfig):
        self.pass_config = config.compilation_config.pass_config
        if self.pass_config.enable_noop:
            self.passes += [NoOpEliminationPass(config)]

        if self.pass_config.enable_fusion:
            self.passes += [FusionPass.instance(config)]
            self.passes += [ActivationQuantFusionPass(config)]

        if self.pass_config.enable_sequence_parallelism:
            self.passes += [SequenceParallelismPass(config)]

        self.fix_functionalization = FixFunctionalizationPass(config)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def add(self, pass_: InductorPass):
        assert isinstance(pass_, InductorPass)
        self.passes.append(pass_)

    def uuid(self):
<<<<<<< HEAD
        return self.__getstate__()

    def __getstate__(self) -> Dict[str, List[Any]]:
        """
        Custom pickling for the pass manager, as some passes cannot be pickled.
        Pickling occurs because the pass manager is set as the value of
        `config["post_grad_custom_post_pass"]` in the Inductor config.
        The config is pickled to act as a key in the Inductor code cache.
        Any other passes in the config are pickled as well.

        TODO(torch==2.6), use the `uuid` method in CustomGraphPass instead.
=======
        """
        The PostGradPassManager is set as a custom pass in the Inductor and
        affects compilation caching. Its uuid depends on the UUIDs of all
        dependent passes and the pass config. See InductorPass for more info.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        """
        state = {"pass_config": self.pass_config.uuid(), "passes": []}
        for pass_ in self.passes:
            state["passes"].append(pass_.uuid())
        state["passes"].append(self.fix_functionalization.uuid())
<<<<<<< HEAD
        return state

    def __setstate__(self, state):
        """
        Do not allow unpickling of the pass manager.
        If this is needed in the future, it should properly pickle the passes.
        """
        raise ValueError("Cannot unpickle PostGradPassManager")
=======
        return InductorPass.hash_dict(state)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
