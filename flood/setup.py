# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import subprocess
import warnings
from typing import Set

# import pkg_resources
from packaging.version import parse, Version
from setuptools import find_packages, setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, \
    CUDA_HOME

__version__ = "0.0.6"

current_dir = os.path.dirname(os.path.abspath(__file__))

SUPPORTED_ARCHS = {"8.0", "8.9", "9.0a"}
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.9;9.0a"

NVCC_FLAGS = ["-O3", "-std=c++17",
              '-U__CUDA_NO_HALF_OPERATORS__',
              '-U__CUDA_NO_HALF_CONVERSIONS__',
              '-U__CUDA_NO_HALF2_OPERATORS__',
              '-U__CUDA_NO_HALF2_CONVERSIONS__',
              '-DSM90_MM'
              ]


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_torch_arch_list() -> Set[str]:
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
    if env_arch_list is None:
        return set()

    torch_arch_list = set(env_arch_list.replace(" ", ";").split(";"))
    if not torch_arch_list:
        return set()

    # Filter out the invalid architectures and print a warning.
    valid_archs = SUPPORTED_ARCHS.union({s + "+PTX" for s in SUPPORTED_ARCHS})
    arch_list = torch_arch_list.intersection(valid_archs)
    # If none of the specified architectures are valid, raise an error.
    if not arch_list:
        raise RuntimeError(
            "None of the CUDA architectures in `TORCH_CUDA_ARCH_LIST` env "
            f"variable ({env_arch_list}) is supported. "
            f"Supported CUDA architectures are: {valid_archs}.")
    invalid_arch_list = torch_arch_list - valid_archs
    if invalid_arch_list:
        warnings.warn(
            f"Unsupported CUDA architectures ({invalid_arch_list}) are "
            "excluded from the `TORCH_CUDA_ARCH_LIST` env variable "
            f"({env_arch_list}). Supported CUDA architectures are: "
            f"{valid_archs}.")
    return arch_list


compute_capabilities = get_torch_arch_list()
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)

if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 4)
    NVCC_FLAGS += ["--threads", str(num_threads)]

for capability in compute_capabilities:
    num = capability[0] + capability[2]
    if capability.endswith("a"):
        num = capability[0] + capability[2] + capability[3]
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

print(NVCC_FLAGS)
ext_modules = []

# cutlass_sources = ["csrc/flood.cpp"]

custom_sources = ["csrc/flood.cpp",
                  "csrc/layernorm/rmsnorm.cu",
                  "csrc/activation/activation_kernels.cu",
                  "csrc/rope/rope.cu",
                  "csrc/cache/cache.cu",
                  "csrc/moe/topk_softmax_kernels.cu",
                  "csrc/moe/moe_align.cu",
                  "csrc/moe/moe_sum.cu",
                  # "csrc/w8a8/scaled_mm_c2x.cu",
                  # "csrc/w8a8/scaled_mm_c3x.cu",
                  # "csrc/w8a8/scaled_mm_entry.cu",
                  "csrc/quantize/fp8_quant.cu"
                  ]

# sources = cutlass_sources + custom_sources
sources = custom_sources
for item in sources:
    sources[sources.index(item)] = os.path.join(current_dir, item)

include_paths = []
include_paths.append(cpp_extension.include_paths(cuda=True))  # cuda path
include_paths.append(os.path.join(current_dir, 'csrc'))
include_paths.append(os.path.join(current_dir, 'csrc/layernorm'))
include_paths.append(os.path.join(current_dir, 'csrc/activation'))
include_paths.append(os.path.join(current_dir, 'csrc/rope'))
include_paths.append(os.path.join(current_dir, 'csrc/cache'))
include_paths.append(os.path.join(current_dir, 'csrc/moe'))
# include_paths.append(os.path.join(current_dir, 'csrc/cutlass/include'))
# include_paths.append(os.path.join(current_dir, 'csrc/w8a8'))
include_paths.append(os.path.join(current_dir, 'csrc/quantize'))

ext_modules.append(
    CUDAExtension(
        name="flood_cuda",
        sources=sources,
        include_dirs=include_paths,
        # libraries=['cublas', 'cudart', 'cudnn', 'curand', 'nvToolsExt'],
        extra_compile_args={
            "cxx": ['-g',
                    '-std=c++17',
                    '-DNDEBUG',
                    '-O3',
                    '-fopenmp',
                    '-lgomp',
                    '-Wno-deprecated-declarations',
                    '-Wno-deprecated',
                    "-Wno-unused-variable"],
            "nvcc": NVCC_FLAGS,
        },
        define_macros=[('VERSION_INFO', __version__),
                       # ('_DEBUG_MODE_', None),
                       ]
    )
)

# with pathlib.Path("requirements.txt").open() as f:
#     install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

setup(
    name="flood",
    version=__version__,
    license="MIT",
    license_files=("LICENSE",),
    description="Flood Inference Framework",
    URL="http://gitlab.alibaba-inc.com/infer/framework.git",
    packages=find_packages(
        exclude=("build", "csrc", "test", "example", "benchmark")),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
    install_requires=[],
    python_requires=">=3.8",
)
