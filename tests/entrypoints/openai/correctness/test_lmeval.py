# SPDX-License-Identifier: Apache-2.0
"""
This file test accuracy of the vLLM server via LMEval.
It uses local-completions, which interacts with vLLM
through the OAI API with N concurrent connections.
This simulates real work usage of the API and makes
sure that the zmq frontend mp RPC message passing and
AsyncLLMEngine are working correctly.
"""

import lm_eval
import pytest

from vllm.platforms import current_platform

from ....utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
<<<<<<< HEAD
EXPECTED_VALUE = 0.58
=======
EXPECTED_VALUE = 0.54
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
DEFAULT_ARGS = ["--max-model-len", "4096", "--disable-log-requests"]
MORE_ARGS_LIST = [
    [],  # Default
    ["--enable-chunked-prefill"],  # Chunked
    ["--num-scheduler-steps", "8"],  # MS
    ["--num-scheduler-steps", "8", "--multi-step-stream-outputs"]  # MS+Stream
]
MAX_WAIT_SECONDS = None

if current_platform.is_tpu():
    MORE_ARGS_LIST = [
        [],  # Default
        # ["--num-scheduler-steps", "8"], # Multi-step << currently fails
    ]
    MAX_WAIT_SECONDS = 600


def run_test(more_args):
    """Run the end to end accuracy test."""

    args = list(DEFAULT_ARGS)
    args.extend(more_args)
    print(f"Running with: {args}")

    with RemoteOpenAIServer(
            MODEL_NAME, args,
            max_wait_seconds=MAX_WAIT_SECONDS) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"

        model_args = (
            f"model={MODEL_NAME},"
            f"base_url={url},"
            f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False")

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=TASK,
        )

        measured_value = results["results"][TASK][FILTER]
        assert (measured_value - RTOL < EXPECTED_VALUE
                and measured_value + RTOL > EXPECTED_VALUE
                ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 currently only supported on CUDA and TPU")
<<<<<<< HEAD
def test_lm_eval_accuracy_v1_engine(monkeypatch):
=======
def test_lm_eval_accuracy_v1_engine(monkeypatch: pytest.MonkeyPatch):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Run with the V1 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")
        more_args = []

        # Limit compilation time for V1
        if current_platform.is_tpu():
            more_args = ["--max-num-seqs", "64"]

        run_test(more_args)


@pytest.mark.parametrize("more_args", MORE_ARGS_LIST)
<<<<<<< HEAD
def test_lm_eval_accuracy_v0_engine(monkeypatch, more_args):
=======
def test_lm_eval_accuracy_v0_engine(monkeypatch: pytest.MonkeyPatch,
                                    more_args):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Run with the V0 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")
        run_test(more_args)
