# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import itertools
import pickle as pkl
import time
<<<<<<< HEAD
from typing import Callable, Iterable, List, Optional, Tuple
=======
from collections.abc import Iterable
from typing import Callable, Optional
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import make_rand_tensors
from weight_shapes import WEIGHT_SHAPES

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
<<<<<<< HEAD
    w8a8_block_fp8_matmul)
=======
    w8a8_block_fp8_matmul,
)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_BATCH_SIZES = [1, 16, 32, 64, 128, 256, 512]
DEFAULT_TP_SIZES = [1]


# bench
<<<<<<< HEAD
def bench_fn(label: str, sub_label: str, description: str, fn: Callable, *args,
             **kwargs) -> TMeasurement:
=======
def bench_fn(
    label: str, sub_label: str, description: str, fn: Callable, *args, **kwargs
) -> TMeasurement:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    min_run_time = 1

    globals = {
        "args": args,
        "kwargs": kwargs,
        "fn": fn,
    }
    return TBenchmark.Timer(
        stmt="fn(*args, **kwargs)",
        globals=globals,
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def bench_int8(
<<<<<<< HEAD
        dtype: torch.dtype,
        m: int,
        k: int,
        n: int,
        label: str,
        sub_label: str,
        bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
=======
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    bench_kernels: Optional[list[str]] = None,
) -> Iterable[TMeasurement]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Benchmark INT8-based kernels."""
    assert dtype == torch.int8
    a, b = make_rand_tensors(torch.int8, m, n, k)
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
<<<<<<< HEAD
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
    azp = torch.zeros((m, ), device="cuda", dtype=torch.int32)
    azp_adj = torch.zeros((n, ), device="cuda", dtype=torch.int32)

    bench_fns = {
        "pytorch_bf16_bf16_bf16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.bfloat16), b.to(dtype=torch.bfloat16)
                         ),
        "pytorch_fp16_fp16_fp16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.float16), b.to(dtype=torch.float16)),
        "cutlass_i8_i8_bf16_scaled_mm":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16),
        "cutlass_i8_i8_bf16_scaled_mm_bias":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16,
                                      bias),
        "cutlass_i8_i8_bf16_scaled_mm_azp":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj),
        "cutlass_i8_i8_bf16_scaled_mm_azp_bias":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj, None, bias),
        "cutlass_i8_i8_bf16_scaled_mm_azp_pt":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj, azp),
        "cutlass_i8_i8_bf16_scaled_mm_azp_pt_bias":
        lambda: ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, torch.
                                          bfloat16, azp_adj, azp, bias),
=======
    bias = torch.zeros((n,), device="cuda", dtype=torch.bfloat16)
    azp = torch.zeros((m,), device="cuda", dtype=torch.int32)
    azp_adj = torch.zeros((n,), device="cuda", dtype=torch.int32)

    bench_fns = {
        "pytorch_bf16_bf16_bf16_matmul-no-scales": lambda: torch.mm(
            a.to(dtype=torch.bfloat16), b.to(dtype=torch.bfloat16)
        ),
        "pytorch_fp16_fp16_fp16_matmul-no-scales": lambda: torch.mm(
            a.to(dtype=torch.float16), b.to(dtype=torch.float16)
        ),
        "cutlass_i8_i8_bf16_scaled_mm": lambda: ops.cutlass_scaled_mm(
            a, b, scale_a, scale_b, torch.bfloat16
        ),
        "cutlass_i8_i8_bf16_scaled_mm_bias": lambda: ops.cutlass_scaled_mm(
            a, b, scale_a, scale_b, torch.bfloat16, bias
        ),
        "cutlass_i8_i8_bf16_scaled_mm_azp": lambda: ops.cutlass_scaled_mm_azp(
            a, b, scale_a, scale_b, torch.bfloat16, azp_adj
        ),
        "cutlass_i8_i8_bf16_scaled_mm_azp_bias": lambda: ops.cutlass_scaled_mm_azp(
            a, b, scale_a, scale_b, torch.bfloat16, azp_adj, None, bias
        ),
        "cutlass_i8_i8_bf16_scaled_mm_azp_pt": lambda: ops.cutlass_scaled_mm_azp(
            a, b, scale_a, scale_b, torch.bfloat16, azp_adj, azp
        ),
        "cutlass_i8_i8_bf16_scaled_mm_azp_pt_bias": lambda: ops.cutlass_scaled_mm_azp(
            a, b, scale_a, scale_b, torch.bfloat16, azp_adj, azp, bias
        ),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    }

    timers = []
    for name, fn in bench_fns.items():
        # If bench_kernels is None, run all. Otherwise, run only exact matches.
        if bench_kernels is None or name in bench_kernels:
            print(f"Running {name}")
            timers.append(bench_fn(label, sub_label, name, fn))

    return timers


def bench_fp8(
<<<<<<< HEAD
        dtype: torch.dtype,
        m: int,
        k: int,
        n: int,
        label: str,
        sub_label: str,
        bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
=======
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    bench_kernels: Optional[list[str]] = None,
) -> Iterable[TMeasurement]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    """Benchmark FP8-based kernels."""
    assert dtype == torch.float8_e4m3fn
    a, b = make_rand_tensors(torch.float8_e4m3fn, m, n, k)
    a_cont = a.contiguous()
    scale_a = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(1.0, device="cuda", dtype=torch.float32)
<<<<<<< HEAD
    block_scale_a = torch.rand((m, k // 128),
                               device="cuda",
                               dtype=torch.float32)
    block_scale_b = torch.rand((k // 128, n // 128),
                               device="cuda",
                               dtype=torch.float32)
    block_scale_a_M_major = block_scale_a.t().contiguous().t()
    block_scale_b_K_major = block_scale_b.t().contiguous().t()
    bias = torch.zeros((n, ), device="cuda", dtype=torch.bfloat16)
=======

    def ceil_div(x: int, y: int) -> int:
        return (x + y - 1) // y

    block_scale_a = torch.rand(
        (m, ceil_div(k, 128)), device="cuda", dtype=torch.float32
    )
    block_scale_b = torch.rand(
        ceil_div(k, 128), ceil_div(n, 128), device="cuda", dtype=torch.float32
    )
    block_scale_a_M_major = block_scale_a.t().contiguous().t()
    block_scale_b_K_major = block_scale_b.t().contiguous().t()
    bias = torch.zeros((n,), device="cuda", dtype=torch.bfloat16)
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    print(m, k, n)

    bench_fns = {
<<<<<<< HEAD
        "pytorch_bf16_bf16_bf16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.bfloat16), b.to(dtype=torch.bfloat16)
                         ),
        "pytorch_fp16_fp16_fp16_matmul-no-scales":
        lambda: torch.mm(a.to(dtype=torch.float16), b.to(dtype=torch.float16)),
        "pytorch_fp8_fp8_fp16_scaled_mm":
        lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.float16),
        "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum":
        lambda: torch._scaled_mm(a,
                                 b,
                                 scale_a,
                                 scale_b,
                                 out_dtype=torch.float16,
                                 use_fast_accum=True),
        "pytorch_fp8_fp8_bf16_scaled_mm":
        lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.bfloat16),
        "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum":
        lambda: torch._scaled_mm(a,
                                 b,
                                 scale_a,
                                 scale_b,
                                 out_dtype=torch.bfloat16,
                                 use_fast_accum=True),
        "cutlass_fp8_fp8_bf16_scaled_mm":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16),
        "cutlass_fp8_fp8_fp16_scaled_mm":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.float16),
        "cutlass_fp8_fp8_bf16_scaled_mm_bias":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.bfloat16,
                                      bias),
        "cutlass_fp8_fp8_fp16_scaled_mm_bias":
        lambda: ops.cutlass_scaled_mm(a, b, scale_a, scale_b, torch.float16,
                                      bias.to(dtype=torch.float16)),
        "triton_fp8_fp8_fp16_scaled_mm_blockwise":
        lambda: w8a8_block_fp8_matmul(a_cont, b.t(), block_scale_a,
                                      block_scale_b.t(), (128, 128)),
        "cutlass_fp8_fp8_fp16_scaled_mm_blockwise":
        lambda: ops.cutlass_scaled_mm(a, b, block_scale_a_M_major,
                                      block_scale_b_K_major, torch.float16),
=======
        "pytorch_bf16_bf16_bf16_matmul-no-scales": lambda: torch.mm(
            a.to(dtype=torch.bfloat16), b.to(dtype=torch.bfloat16)
        ),
        "pytorch_fp16_fp16_fp16_matmul-no-scales": lambda: torch.mm(
            a.to(dtype=torch.float16), b.to(dtype=torch.float16)
        ),
        "pytorch_fp8_fp8_fp16_scaled_mm": lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.float16
        ),
        "pytorch_fp8_fp8_fp16_scaled_mm_fast_accum": lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.float16, use_fast_accum=True
        ),
        "pytorch_fp8_fp8_bf16_scaled_mm": lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.bfloat16
        ),
        "pytorch_fp8_fp8_bf16_scaled_mm_fast_accum": lambda: torch._scaled_mm(
            a, b, scale_a, scale_b, out_dtype=torch.bfloat16, use_fast_accum=True
        ),
        "cutlass_fp8_fp8_bf16_scaled_mm": lambda: ops.cutlass_scaled_mm(
            a, b, scale_a, scale_b, torch.bfloat16
        ),
        "cutlass_fp8_fp8_fp16_scaled_mm": lambda: ops.cutlass_scaled_mm(
            a, b, scale_a, scale_b, torch.float16
        ),
        "cutlass_fp8_fp8_bf16_scaled_mm_bias": lambda: ops.cutlass_scaled_mm(
            a, b, scale_a, scale_b, torch.bfloat16, bias
        ),
        "cutlass_fp8_fp8_fp16_scaled_mm_bias": lambda: ops.cutlass_scaled_mm(
            a, b, scale_a, scale_b, torch.float16, bias.to(dtype=torch.float16)
        ),
        "triton_fp8_fp8_fp16_scaled_mm_blockwise": lambda: w8a8_block_fp8_matmul(
            a_cont, b.t(), block_scale_a, block_scale_b.t(), (128, 128)
        ),
        "cutlass_fp8_fp8_fp16_scaled_mm_blockwise": lambda: ops.cutlass_scaled_mm(
            a, b, block_scale_a_M_major, block_scale_b_K_major, torch.float16
        ),
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    }

    timers = []
    for name, fn in bench_fns.items():
        # If bench_kernels is None, run all. Otherwise, run only exact matches.
        if bench_kernels is None or name in bench_kernels:
            print(f"Running {name}")
            timers.append(bench_fn(label, sub_label, name, fn))

    return timers


<<<<<<< HEAD
def bench(dtype: torch.dtype,
          m: int,
          k: int,
          n: int,
          label: str,
          sub_label: str,
          bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
=======
def bench(
    dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
    label: str,
    sub_label: str,
    bench_kernels: Optional[list[str]] = None,
) -> Iterable[TMeasurement]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    if dtype == torch.int8:
        return bench_int8(dtype, m, k, n, label, sub_label, bench_kernels)
    if dtype == torch.float8_e4m3fn:
        return bench_fp8(dtype, m, k, n, label, sub_label, bench_kernels)
    raise ValueError("unsupported type")


# runner
def print_timers(timers: Iterable[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


<<<<<<< HEAD
def run(dtype: torch.dtype,
        MKNs: Iterable[Tuple[int, int, int]],
        bench_kernels: Optional[List[str]] = None) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        timers = bench(dtype,
                       m,
                       k,
                       n,
                       f"scaled-{dtype}-gemm",
                       f"MKN=({m}x{k}x{n})",
                       bench_kernels=bench_kernels)
=======
def run(
    dtype: torch.dtype,
    MKNs: Iterable[tuple[int, int, int]],
    bench_kernels: Optional[list[str]] = None,
) -> Iterable[TMeasurement]:
    results = []
    for m, k, n in MKNs:
        timers = bench(
            dtype,
            m,
            k,
            n,
            f"scaled-{dtype}-gemm",
            f"MKN=({m}x{k}x{n})",
            bench_kernels=bench_kernels,
        )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        print_timers(timers)
        results.extend(timers)
    return results


<<<<<<< HEAD
def make_output(data: Iterable[TMeasurement],
                MKNs: Iterable[Tuple[int, int, int]],
                base_description: str,
                timestamp=None):
=======
def make_output(
    data: Iterable[TMeasurement],
    MKNs: Iterable[tuple[int, int, int]],
    base_description: str,
    timestamp=None,
):
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    print(f"== All Results {base_description} ====")
    print_timers(data)

    # pickle all the results
    timestamp = int(time.time()) if timestamp is None else timestamp
    with open(f"{base_description}-{timestamp}.pkl", "wb") as f:
        pkl.dump(data, f)


def run_square_bench(args):
<<<<<<< HEAD
    dim_sizes = list(
        range(args.dim_start, args.dim_end + 1, args.dim_increment))
=======
    dim_sizes = list(range(args.dim_start, args.dim_end + 1, args.dim_increment))
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    MKNs = list(zip(dim_sizes, dim_sizes, dim_sizes))
    data = run(args.dtype, MKNs, bench_kernels=args.kernels)
    make_output(data, MKNs, f"square_bench-{args.dtype}")


def run_range_bench(args):
    dim_sizes = list(range(args.dim_start, args.dim_end, args.dim_increment))
    n = len(dim_sizes)
    Ms = [args.m_constant] * n if args.m_constant is not None else dim_sizes
    Ks = [args.k_constant] * n if args.k_constant is not None else dim_sizes
    Ns = [args.n_constant] * n if args.n_constant is not None else dim_sizes
    MKNs = list(zip(Ms, Ks, Ns))
    data = run(args.dtype, MKNs, bench_kernels=args.kernels)
    make_output(data, MKNs, f"range_bench-{args.dtype}")


def run_model_bench(args):
    print("Benchmarking models:")
    for i, model in enumerate(args.models):
        print(f"[{i}]  {model}")

<<<<<<< HEAD
    def model_shapes(model_name: str, tp_size: int) -> List[Tuple[int, int]]:
=======
    def model_shapes(model_name: str, tp_size: int) -> list[tuple[int, int]]:
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
        KNs = []
        for KN, tp_split_dim in copy.deepcopy(WEIGHT_SHAPES[model_name]):
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            KNs.append(KN)
        return KNs

    model_bench_data = []
    models_tps = list(itertools.product(args.models, args.tp_sizes))
    for model, tp_size in models_tps:
        Ms = args.batch_sizes
        KNs = model_shapes(model, tp_size)
        MKNs = []
        for m in Ms:
            for k, n in KNs:
                MKNs.append((m, k, n))

        data = run(args.dtype, MKNs, bench_kernels=args.kernels)
        model_bench_data.append(data)

    # Print all results
    for data, model_tp in zip(model_bench_data, models_tps):
        model, tp_size = model_tp
        print(f"== Results {args.dtype} {model}-TP{tp_size} ====")
        print_timers(data)

    timestamp = int(time.time())

    all_data = []
    for d in model_bench_data:
        all_data.extend(d)
    # pickle all data
    with open(f"model_bench-{args.dtype}-{timestamp}.pkl", "wb") as f:
        pkl.dump(all_data, f)


<<<<<<< HEAD
if __name__ == '__main__':
=======
if __name__ == "__main__":
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea

    def to_torch_dtype(dt):
        if dt == "int8":
            return torch.int8
        if dt == "fp8":
            return torch.float8_e4m3fn
        raise ValueError("unsupported dtype")

    parser = FlexibleArgumentParser(
        description="""
Benchmark Cutlass GEMM.

    To run square GEMMs:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 square_bench --dim-start 128 --dim-end 512 --dim-increment 64
    
    To run constant N and K and sweep M:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 range_bench --dim-start 128 --dim-end 512 --dim-increment 64 --n-constant 16384 --k-constant 16384
    
    To run dimensions from a model:
        python3 ./benchmarks/cutlass_benchmarks/w8a8_benchmarks.py --dtype fp8 model_bench --models meta-llama/Llama-2-7b-hf --batch-sizes 16 --tp-sizes 1
    
    Output:
        - a .pkl file, that is a list of raw torch.benchmark.utils.Measurements for the pytorch and cutlass implementations for the various GEMMs.
            """,  # noqa: E501
<<<<<<< HEAD
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--dtype",
                        type=to_torch_dtype,
                        required=True,
                        help="Available options are ['int8', 'fp8']")
=======
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dtype",
        type=to_torch_dtype,
        required=True,
        help="Available options are ['int8', 'fp8']",
    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    parser.add_argument(
        "--kernels",
        nargs="+",
        type=str,
        default=None,
<<<<<<< HEAD
        help=
        "Exact names of the kernels to benchmark. If not set, runs all kernels."
=======
        help="Exact names of the kernels to benchmark. If not set, runs all kernels.",
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    )

    subparsers = parser.add_subparsers(dest="cmd")

    square_parser = subparsers.add_parser("square_bench")
    square_parser.add_argument("--dim-start", type=int, required=True)
    square_parser.add_argument("--dim-end", type=int, required=True)
    square_parser.add_argument("--dim-increment", type=int, required=True)
    square_parser.set_defaults(func=run_square_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--dim-start", type=int, required=True)
    range_parser.add_argument("--dim-end", type=int, required=True)
    range_parser.add_argument("--dim-increment", type=int, required=True)
    range_parser.add_argument("--m-constant", type=int, default=None)
    range_parser.add_argument("--n-constant", type=int, default=None)
    range_parser.add_argument("--k-constant", type=int, default=None)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
<<<<<<< HEAD
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=DEFAULT_MODELS,
                              choices=WEIGHT_SHAPES.keys())
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--batch-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_BATCH_SIZES)
=======
    model_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    model_parser.add_argument(
        "--tp-sizes", nargs="+", type=int, default=DEFAULT_TP_SIZES
    )
    model_parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES
    )
>>>>>>> eca18691d2fe29c4f6c1b466709eda9f123116ea
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
