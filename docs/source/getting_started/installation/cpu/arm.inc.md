# Installation

vLLM has been adapted to work on ARM64 CPUs with NEON support, leveraging the CPU backend initially developed for the x86 platform.

ARM CPU backend currently supports Float32, FP16 and BFloat16 datatypes.

:::{attention}
There are no pre-built wheels or images for this device, so you must build vLLM from source.
:::

## Requirements

- OS: Linux
- Compiler: `gcc/g++ >= 12.3.0` (optional, recommended)
- Instruction Set Architecture (ISA): NEON support is required

## Set up using Python

### Pre-built wheels

### Build wheel from source

<<<<<<< HEAD
:::{include} build.inc.md
=======
:::{include} cpu/build.inc.md
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
:::

Testing has been conducted on AWS Graviton3 instances for compatibility.

## Set up using Docker

### Pre-built images

### Build image from source

## Extra information
