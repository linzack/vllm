# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


<<<<<<< HEAD
=======
def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(model="jason9693/Qwen2.5-1.5B-apeach",
                        task="classify",
                        enforce_eager=True)
    return parser.parse_args()


>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass task="classify" for classification models
    model = LLM(**vars(args))

    # Generate logits. The output is a list of ClassificationRequestOutputs.
    outputs = model.classify(prompts)

    # Print the outputs.
<<<<<<< HEAD
=======
    print("\nGenerated Outputs:\n" + "-" * 60)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    for prompt, output in zip(prompts, outputs):
        probs = output.outputs.probs
        probs_trimmed = ((str(probs[:16])[:-1] +
                          ", ...]") if len(probs) > 16 else probs)
<<<<<<< HEAD
        print(f"Prompt: {prompt!r} | "
              f"Class Probabilities: {probs_trimmed} (size={len(probs)})")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(model="jason9693/Qwen2.5-1.5B-apeach",
                        task="classify",
                        enforce_eager=True)
    args = parser.parse_args()
=======
        print(f"Prompt: {prompt!r} \n"
              f"Class Probabilities: {probs_trimmed} (size={len(probs)})")
        print("-" * 60)


if __name__ == "__main__":
    args = parse_args()
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    main(args)
