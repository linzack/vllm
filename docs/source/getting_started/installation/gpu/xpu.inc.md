# Installation

<<<<<<< HEAD
vLLM initially supports basic model inferencing and serving on Intel GPU platform.
=======
vLLM initially supports basic model inference and serving on Intel GPU platform.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

:::{attention}
There are no pre-built wheels or images for this device, so you must build vLLM from source.
:::

## Requirements

- Supported Hardware: Intel Data Center GPU, Intel ARC GPU
<<<<<<< HEAD
- OneAPI requirements: oneAPI 2024.2
=======
- OneAPI requirements: oneAPI 2025.0
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built XPU wheels.

### Build wheel from source

<<<<<<< HEAD
- First, install required driver and intel OneAPI 2024.2 or later.
- Second, install Python packages for vLLM XPU backend building:

```console
source /opt/intel/oneapi/setvars.sh
pip install --upgrade pip
pip install -v -r requirements-xpu.txt
```

- Finally, build and install vLLM XPU backend:
=======
- First, install required driver and Intel OneAPI 2025.0 or later.
- Second, install Python packages for vLLM XPU backend building:

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install --upgrade pip
pip install -v -r requirements/xpu.txt
```

- Then, build and install vLLM XPU backend:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

```console
VLLM_TARGET_DEVICE=xpu python setup.py install
```

:::{note}
- FP16 is the default data type in the current XPU backend. The BF16 data
  type is supported on Intel Data Center GPU, not supported on Intel Arc GPU yet.
:::

## Set up using Docker

### Pre-built images

Currently, there are no pre-built XPU images.

### Build image from source

```console
<<<<<<< HEAD
$ docker build -f Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
=======
$ docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
$ docker run -it \
             --rm \
             --network=host \
             --device /dev/dri \
             -v /dev/dri/by-path:/dev/dri/by-path \
             vllm-xpu-env
```

## Supported features

<<<<<<< HEAD
XPU platform supports tensor-parallel inference/serving and also supports pipeline parallel as a beta feature for online serving. We requires Ray as the distributed runtime backend. For example, a reference execution likes following:
=======
XPU platform supports **tensor parallel** inference/serving and also supports **pipeline parallel** as a beta feature for online serving. We require Ray as the distributed runtime backend. For example, a reference execution like following:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

```console
python -m vllm.entrypoints.openai.api_server \
     --model=facebook/opt-13b \
     --dtype=bfloat16 \
<<<<<<< HEAD
     --device=xpu \
=======
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
     --max_model_len=1024 \
     --distributed-executor-backend=ray \
     --pipeline-parallel-size=2 \
     -tp=8
```

<<<<<<< HEAD
By default, a ray instance will be launched automatically if no existing one is detected in system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the <gh-file:examples/online_serving/run_cluster.sh> helper script.
=======
By default, a ray instance will be launched automatically if no existing one is detected in the system, with `num-gpus` equals to `parallel_config.world_size`. We recommend properly starting a ray cluster before execution, referring to the <gh-file:examples/online_serving/run_cluster.sh> helper script.
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
