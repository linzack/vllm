# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Callable, Union

from torch import fx

from vllm.compilation.inductor_pass import InductorPass
<<<<<<< HEAD
=======
from vllm.config import get_current_vllm_config
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


class TestBackend:
    """
    This class provides a simple Inductor backend that can be used for testing.
    It takes a list of custom passes and runs them after Inductor's passes.
    It also saves the graph before and after the custom passes for inspection.
<<<<<<< HEAD
=======

    Inductor config can be modified directly by editing the inductor_config
    property. This can be helpful for adding passes like the
    'pre_grad_custom_pass' and the 'post_grad_custom_pre_pass'.
    Inductor config is default-initialized from VllmConfig.CompilationConfig.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """

    def __init__(self, *passes: Union[InductorPass, Callable[[fx.Graph],
                                                             None]]):
        self.custom_passes = list(passes)
<<<<<<< HEAD
        from torch._inductor import config
        self.current_config = config.shallow_copy_dict()
        self.current_config['force_disable_caches'] = True
        self.current_config['post_grad_custom_post_pass'] = self.post_pass

    def __call__(self, graph: fx.GraphModule, example_inputs):
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph,
                          example_inputs,
                          config_patches=self.current_config)
=======
        compile_config = get_current_vllm_config().compilation_config
        self.inductor_config = compile_config.inductor_compile_config
        self.inductor_config['force_disable_caches'] = True
        self.inductor_config['post_grad_custom_post_pass'] = self.post_pass

    def __call__(self, graph: fx.GraphModule, example_inputs):
        self.graph_pre_compile = deepcopy(graph)
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(graph,
                          example_inputs,
                          config_patches=self.inductor_config)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def post_pass(self, graph: fx.Graph):
        self.graph_pre_pass = deepcopy(graph)
        for pass_ in self.custom_passes:
            pass_(graph)

        self.graph_post_pass = deepcopy(graph)
        # assign by reference, will reflect the final state of the graph
        self.final_graph = graph
