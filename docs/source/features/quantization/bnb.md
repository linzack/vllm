(bits-and-bytes)=

# BitsAndBytes

vLLM now supports [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) for more efficient model inference.
BitsAndBytes quantizes models to reduce memory usage and enhance performance without significantly sacrificing accuracy.
Compared to other quantization methods, BitsAndBytes eliminates the need for calibrating the quantized model with input data.

Below are the steps to utilize BitsAndBytes with vLLM.

```console
<<<<<<< HEAD
pip install bitsandbytes>=0.45.0
=======
pip install bitsandbytes>=0.45.3
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```

vLLM reads the model's config file and supports both in-flight quantization and pre-quantized checkpoint.

<<<<<<< HEAD
You can find bitsandbytes quantized models on <https://huggingface.co/models?other=bitsandbytes>.
=======
You can find bitsandbytes quantized models on <https://huggingface.co/models?search=bitsandbytes>.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
And usually, these repositories have a config.json file that includes a quantization_config section.

## Read quantized checkpoint

<<<<<<< HEAD
=======
For pre-quantized checkpoints, vLLM will try to infer the quantization method from the config file, so you don't need to explicitly specify the quantization argument.

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```python
from vllm import LLM
import torch
# unsloth/tinyllama-bnb-4bit is a pre-quantized checkpoint.
model_id = "unsloth/tinyllama-bnb-4bit"
<<<<<<< HEAD
llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
quantization="bitsandbytes", load_format="bitsandbytes")
=======
llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```

## Inflight quantization: load as 4bit quantization

<<<<<<< HEAD
=======
For inflight 4bit quantization with BitsAndBytes, you need to explicitly specify the quantization argument.

>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```python
from vllm import LLM
import torch
model_id = "huggyllama/llama-7b"
llm = LLM(model=model_id, dtype=torch.bfloat16, trust_remote_code=True, \
<<<<<<< HEAD
quantization="bitsandbytes", load_format="bitsandbytes")
=======
quantization="bitsandbytes")
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```

## OpenAI Compatible Server

<<<<<<< HEAD
Append the following to your 4bit model arguments:

```console
--quantization bitsandbytes --load-format bitsandbytes
=======
Append the following to your model arguments for 4bit inflight quantization:

```console
--quantization bitsandbytes
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
```
