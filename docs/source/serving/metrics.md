# Production Metrics

vLLM exposes a number of metrics that can be used to monitor the health of the
system. These metrics are exposed via the `/metrics` endpoint on the vLLM
OpenAI compatible API server.

You can start the server using Python, or using [Docker](#deployment-docker):

```console
vllm serve unsloth/Llama-3.2-1B-Instruct
```

Then query the endpoint to get the latest metrics from the server:

```console
$ curl http://0.0.0.0:8000/metrics

# HELP vllm:iteration_tokens_total Histogram of number of tokens per engine_step.
# TYPE vllm:iteration_tokens_total histogram
vllm:iteration_tokens_total_sum{model_name="unsloth/Llama-3.2-1B-Instruct"} 0.0
vllm:iteration_tokens_total_bucket{le="1.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="8.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="16.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="32.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="64.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="128.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="256.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
vllm:iteration_tokens_total_bucket{le="512.0",model_name="unsloth/Llama-3.2-1B-Instruct"} 3.0
...
```

The following metrics are exposed:

:::{literalinclude} ../../../vllm/engine/metrics.py
:end-before: end-metrics-definitions
:language: python
:start-after: begin-metrics-definitions
:::

The following metrics are deprecated and due to be removed in a future version:

<<<<<<< HEAD
- *(No metrics are currently deprecated)*
=======
- `vllm:num_requests_swapped`, `vllm:cpu_cache_usage_perc`, and
  `vllm:cpu_prefix_cache_hit_rate` because KV cache offloading is not
  used in V1.
- `vllm:gpu_prefix_cache_hit_rate` is replaced by queries+hits
  counters in V1.
- `vllm:time_in_queue_requests` because it duplicates
  `vllm:request_queue_time_seconds`.
- `vllm:model_forward_time_milliseconds` and
  `vllm:model_execute_time_milliseconds` because
  prefill/decode/inference time metrics should be used instead.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

Note: when metrics are deprecated in version `X.Y`, they are hidden in version `X.Y+1`
but can be re-enabled using the `--show-hidden-metrics-for-version=X.Y` escape hatch,
and are then removed in version `X.Y+2`.
