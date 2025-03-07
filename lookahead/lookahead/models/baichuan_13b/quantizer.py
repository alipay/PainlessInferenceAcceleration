# Copyright (c) 2023, Baichuan Intelligent Technology. All rights reserved.

import torch
from typing import List
import bz2
import base64
import ctypes
from transformers.utils import logging
logger = logging.get_logger(__name__)

try:
    from cpm_kernels.kernels.base import LazyKernelCModule, KernelFunction, round_up

    class Kernel:
        def __init__(self, code: bytes, function_names: List[str]):
            self.code = code
            self._function_names = function_names
            self._cmodule = LazyKernelCModule(self.code)

            for name in self._function_names:
                setattr(self, name, KernelFunction(self._cmodule, name))
    quantization_code = "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"
    kernels = Kernel(
        bz2.decompress(base64.b64decode(quantization_code)),
        [
            "int4_to_fp16",
            "fp16_to_int4",
            "int8_to_fp16",
            "fp16_to_int8",
            "int4_to_bf16",
            "bf16_to_int4",
            "int8_to_bf16",
            "bf16_to_int8",
        ],
    )
except Exception as exception:
    kernels = None
    logger.warning("Failed to load kernels:" + str(exception))

def quant4(weight: torch.Tensor, scale: torch.Tensor):
    stream = torch.cuda.current_stream()
    num_row = weight.size(0)
    num_chan_fp16 = weight.size(1)
    # 4bit
    num_chan_int = num_chan_fp16 // 8
    qweight = torch.zeros((num_row, num_chan_int), dtype=torch.int32, device=weight.device)
    intweight = torch.empty(num_row, num_chan_fp16, dtype = torch.int32)
    intweight = torch.clip(torch.round(weight.to(scale.dtype) / scale[:, None]),-16, 15).to(dtype=torch.int32)

    for j in range(num_chan_int):
        qweight[:, j] = ((intweight[:, j*8+7] & 0x0f) << 28) \
            | ((intweight[:, j*8+6] & 0x0f) << 24) \
            | ((intweight[:, j*8+5] & 0x0f) << 20) \
            | ((intweight[:, j*8+4] & 0x0f) << 16) \
            | ((intweight[:, j*8+3] & 0x0f) << 12) \
            | ((intweight[:, j*8+2] & 0x0f) << 8) \
            | ((intweight[:, j*8+1] & 0x0f) << 4) \
            | ((intweight[:, j*8] & 0x0f))
    return qweight

def dequant4(qweight: torch.Tensor, scale: torch.Tensor, input: torch.Tensor):
    stream = torch.cuda.current_stream()
    num_row = qweight.size(0)
    num_chan_int = qweight.size(1)
    # 4bit
    num_chan_fp16 = num_chan_int * 8
    
    out = torch.empty((num_row, num_chan_fp16), dtype=input.dtype, device=qweight.device)

    blockDim = (128, 1, 1)
    gridDim = ((num_chan_int + blockDim[0] - 1) // blockDim[0], num_row, 1)
    if input.dtype == torch.bfloat16:
    	kernels.int4_to_bf16(
            gridDim,
       	    blockDim,
            0,
            stream,
            [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(qweight.data_ptr()), 
            ctypes.c_void_p(scale.data_ptr()), ctypes.c_int32(num_row), ctypes.c_int32(num_chan_int), ctypes.c_int32(num_chan_fp16)],
        )
    elif input.dtype == torch.float16:
        kernels.int4_to_fp16(
            gridDim,
            blockDim,
            0,
            stream,
            [ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(qweight.data_ptr()),
            ctypes.c_void_p(scale.data_ptr()), ctypes.c_int32(num_row), ctypes.c_int32(num_chan_int), ctypes.c_int32(num_chan_fp16)],
        )
    return out

class QLinear(torch.nn.Module):
    def __init__(self, bits: int, weight: torch.Tensor, bias=None):
        super().__init__()
        self.quant_bits = bits
        self.scale = weight.abs().max(dim=-1).values / ((2 ** (bits - 1)) - 1)
        self.scale = self.scale.to(torch.float32)
        if self.quant_bits == 4:
            self.weight = quant4(weight, self.scale)
        elif self.quant_bits == 8:
            self.weight = torch.round(weight.to(self.scale.dtype) / self.scale[:, None]).to(torch.int8)
        if self.quant_bits == 8:
            self.weight = self.weight.T
        self.bias = None

    def forward(self, input):
        if self.quant_bits == 4:
            assert(input.dtype == torch.bfloat16 or input.dtype == torch.float16)            

        if self.weight.device != input.device:
            self.weight = self.weight.to(input.device)
            self.scale = self.scale.to(input.device)
        
        if self.quant_bits == 4:
            self.scale = self.scale.to(input.dtype)
            rweight = dequant4(self.weight, self.scale, input).T
            output = torch.matmul(input, rweight)
        elif self.quant_bits == 8:
            rweight = self.weight.to(input.dtype) * self.scale.to(input.dtype)
            output = torch.matmul(input, rweight)
        if self.bias is not None:
            output = output + self.bias
        return output
