(engine-args)=

# Engine Arguments

<<<<<<< HEAD
Below, you can find an explanation of every engine argument for vLLM:
=======
Engine arguments control the behavior of the vLLM engine.

- For [offline inference](#offline-inference), they are part of the arguments to `LLM` class.
- For [online serving](#openai-compatible-server), they are part of the arguments to `vllm serve`.

For references to all arguments available from `vllm serve` see the [serve args](#serve-args) documentation.

Below, you can find an explanation of every engine argument:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
<<<<<<< HEAD
=======
    :markdownhelp:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```

## Async Engine Arguments

<<<<<<< HEAD
Below are the additional arguments related to the asynchronous engine:
=======
Additional arguments are available to the asynchronous engine which is used for online serving:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

<!--- pyml disable-num-lines 7 no-space-in-emphasis -->
```{eval-rst}
.. argparse::
    :module: vllm.engine.arg_utils
    :func: _async_engine_args_parser
    :prog: vllm serve
    :nodefaultconst:
<<<<<<< HEAD
=======
    :markdownhelp:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```
