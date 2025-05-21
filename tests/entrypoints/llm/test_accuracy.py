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

<<<<<<< HEAD
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
=======
MODEL_NAMES = [
    "Qwen/Qwen2-1.5B-Instruct",
    "google/gemma-3-1b-it",
]
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
NUM_CONCURRENT = 500
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
RTOL = 0.03
<<<<<<< HEAD
EXPECTED_VALUE = 0.58


def run_test(more_args=None):
    """Run the end to end accuracy test."""

    model_args = f"pretrained={MODEL_NAME},max_model_len=4096"
=======
EXPECTED_VALUES = {
    "Qwen/Qwen2-1.5B-Instruct": 0.58,
    "google/gemma-3-1b-it": 0.25,
}


def run_test(model_name, more_args=None):
    """Run the end to end accuracy test."""

    model_args = f"pretrained={model_name},max_model_len=4096"
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    if more_args is not None:
        model_args = "{},{}".format(model_args, more_args)

    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks="gsm8k",
        batch_size="auto",
    )

    measured_value = results["results"][TASK][FILTER]
<<<<<<< HEAD
    assert (measured_value - RTOL < EXPECTED_VALUE
            and measured_value + RTOL > EXPECTED_VALUE
            ), f"Expected: {EXPECTED_VALUE} |  Measured: {measured_value}"
=======
    assert model_name in EXPECTED_VALUES, (
        f"Cannot find the expected value for the model {model_name=}")
    expected_value = EXPECTED_VALUES[model_name]
    assert (measured_value - RTOL < expected_value
            and measured_value + RTOL > expected_value
            ), f"Expected: {expected_value} |  Measured: {measured_value}"


# TODO: [AlexM] Fix it with new CI/CD tests
TPU_TP_TEST_STR = ""  #"tensor_parallel_size=4"
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea


@pytest.mark.skipif(not current_platform.is_cuda()
                    and not current_platform.is_tpu(),
                    reason="V1 is currently only supported on CUDA and TPU")
<<<<<<< HEAD
def test_lm_eval_accuracy_v1_engine(monkeypatch):
=======
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_lm_eval_accuracy_v1_engine(model, monkeypatch: pytest.MonkeyPatch):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Run with the V1 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        more_args = None
        if current_platform.is_tpu():
            # Limit compilation time for TPU V1
<<<<<<< HEAD
            more_args = "max_num_seqs=64"

        run_test(more_args)


def test_lm_eval_accuracy_v0_engine(monkeypatch):
=======
            more_args = "max_model_len=2048,max_num_seqs=64"

            # Add TP test (if provided)
            if TPU_TP_TEST_STR:
                more_args += ",{}".format(TPU_TP_TEST_STR)

        run_test(model, more_args)


def test_lm_eval_accuracy_v0_engine(monkeypatch: pytest.MonkeyPatch):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Run with the V0 Engine."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "0")
<<<<<<< HEAD
        run_test()
=======
        run_test("Qwen/Qwen2-1.5B-Instruct")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
