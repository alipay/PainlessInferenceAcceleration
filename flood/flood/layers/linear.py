# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from flood.ops import scaled_fp8_quant

try:
    from vllm import _custom_ops as ops
except:
    print('vllm.ops is not installed! W8A8INT8 is not supported!')


class AutoLinear():

    @classmethod
    def from_pretrained(cls,
                        in_features,
                        out_features,
                        weight=None,
                        bias=None,
                        device=None,
                        dtype=None,
                        input_scale=None,
                        weight_scale=None,
                        weight_dtype=None,
                        bias_dtype=None,
                        config=None,
                        name=None):

        quant_type = AutoLinear.quant_type(config, name)

        if quant_type is None:
            return NativeLinear(in_features,
                                out_features,
                                bias=bias,
                                weight=weight,
                                device=device,
                                dtype=dtype)
        elif quant_type == 'dynamic_fp8':
            return DynamicW8A8Fp8Linear(in_features,
                                        out_features,
                                        weight=weight,
                                        bias=bias,
                                        device=device,
                                        input_scale=input_scale,
                                        weight_scale=weight_scale,
                                        weight_dtype=weight_dtype,
                                        bias_dtype=bias_dtype,
                                        config=config)
        elif quant_type == 'static_fp8':
            return StaticW8A8Fp8Linear(in_features,
                                       out_features,
                                       weight=weight,
                                       bias=bias,
                                       device=device,
                                       input_scale=input_scale,
                                       weight_scale=weight_scale,
                                       weight_dtype=weight_dtype,
                                       bias_dtype=bias_dtype,
                                       config=config)
        elif quant_type == 'dynamic_int8':
            return DynamicW8A8Int8Linear(in_features,
                                         out_features,
                                         weight=weight,
                                         bias=bias,
                                         device=device,
                                         input_scale=input_scale,
                                         weight_scale=weight_scale,
                                         weight_dtype=weight_dtype,
                                         bias_dtype=bias_dtype,
                                         config=config)
        elif quant_type == 'static_int8':
            return StaticW8A8Int8Linear(in_features,
                                        out_features,
                                        weight=weight,
                                        bias=bias,
                                        device=device,
                                        input_scale=input_scale,
                                        weight_scale=weight_scale,
                                        weight_dtype=weight_dtype,
                                        bias_dtype=bias_dtype,
                                        config=config)
        else:
            raise ValueError(f'unknown quant_type:{quant_type}')

    @staticmethod
    def quant_type(config, layer_name):
        if not config or not hasattr(config,
                                     'quantization_config') or config.quantization_config is None:
            return None

        conf = config
        if hasattr(config, '_asdict'):
            conf = config._asdict()
        if hasattr(config, 'to_dict') and callable(config.to_dict):
            conf = config.to_dict()

        conf = conf['quantization_config']
        if hasattr(conf, 'to_dict') and callable(conf.to_dict):
            conf = conf.to_dict()
        if layer_name in conf['ignore']:
            return None
        conf = conf['config_groups']['group_0']
        if conf['input_activations']['dynamic'] and not conf['weights'][
            'dynamic'] and conf['weights']['type'] == 'float' and \
                conf['weights']['num_bits'] == 8:
            return 'dynamic_fp8'
        if not conf['input_activations']['dynamic'] and not conf['weights'][
            'dynamic'] and conf['weights']['type'] == 'float' and \
                conf['weights']['num_bits'] == 8:
            return 'static_fp8'
        if conf['input_activations']['dynamic'] and not conf['weights'][
            'dynamic'] and conf['weights']['type'] == 'int' and conf['weights'][
            'num_bits'] == 8 and conf['input_activations']['symmetric'] and \
                conf['weights']['symmetric']:
            return 'dynamic_int8'
        if not conf['input_activations']['dynamic'] and not conf['weights'][
            'dynamic'] and conf['weights']['type'] == 'int' and conf['weights'][
            'num_bits'] == 8 and conf['input_activations']['symmetric'] and \
                conf['weights']['symmetric']:
            return 'static_int8'
        else:
            return None


class NativeLinear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 weight=None,
                 bias=None,
                 dtype=None,
                 input_scale=None,
                 weight_scale=None,
                 device=None,
                 weight_dtype=None,
                 bias_dtype=None,
                 config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.device = device
        self.weight_dtype = weight_dtype or dtype
        self.bias_dtype = bias_dtype or dtype
        self.config = config

        assert weight is None or isinstance(weight, Parameter)
        if weight is None:
            data = torch.empty(out_features, in_features, dtype=weight_dtype)
            self.weight = Parameter(data, requires_grad=False)
        else:
            self.weight = weight

        assert bias is None or isinstance(bias, bool) or isinstance(bias,
                                                                    Parameter)
        if bias is None or isinstance(bias, Parameter):
            self.bias = bias
        elif bias is False:
            self.bias = None
        else:
            data = torch.empty(out_features, dtype=bias_dtype)
            self.bias = Parameter(data, requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight, bias=self.bias)

    @staticmethod
    def merge(linears):
        assert len(linears) > 1
        in_features = []
        out_features = []
        dtype = None
        device = None
        weights = []
        biases = []
        for linear in linears:
            in_features.append(linear.in_features)
            out_features.append(linear.out_features)
            weight = linear.weight.data
            dtype = weight.dtype
            device = weight.device
            weights.append(weight)
            biases.append(linear.bias.data if linear.bias is not None else None)

        assert min(in_features) == max(in_features)

        weight = Parameter(torch.cat(weights, dim=0), requires_grad=False)
        bias = None if any([b is None for b in biases]) else Parameter(
            torch.cat(biases, dim=0), requires_grad=False)
        in_features = in_features[0]
        out_features = sum(out_features)
        config = linears[0].config

        return NativeLinear(in_features,
                            out_features,
                            weight=weight,
                            bias=bias,
                            device=device,
                            dtype=dtype,
                            config=config)

    def patch(self):
        pass

    def __repr__(self):
        return (f'Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')

    def retype(self, dtype=torch.float8_e4m3fn, quant_type='dynamic'):
        if dtype == torch.float8_e4m3fn and quant_type == 'dynamic':
            weight_scale = torch.nn.Parameter(
                torch.max(self.weight.data.abs(), dim=1,
                          keepdim=True).values.float() / 448.0)
            weight = torch.nn.Parameter(
                (self.weight / weight_scale.to(self.weight.dtype)).to(
                    dtype).t(), requires_grad=False)
            self.weight = None
            delattr(self, 'weight')
            return DynamicW8A8Fp8Linear(self.in_features,
                                        self.out_features,
                                        weight=weight,
                                        bias=self.bias,
                                        weight_scale=weight_scale,
                                        device=self.device)
        else:
            raise ValueError(
                f"unknown dtype:{dtype} and quant_type:{quant_type}")


class DynamicW8A8Fp8Linear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 weight=None,
                 bias=None,
                 dtype=None,
                 input_scale=None,
                 weight_scale=None,
                 device=None,
                 weight_dtype=None,
                 bias_dtype=None,
                 config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.device = device
        self.weight_dtype = torch.float8_e4m3fn
        self.bias_dtype = bias_dtype or dtype
        self.config = config

        assert weight is None or isinstance(weight, Parameter)
        if weight is None:
            data = torch.empty(out_features, in_features,
                               dtype=self.weight_dtype)
            self.weight = Parameter(data, requires_grad=False)
        else:
            self.weight = weight

        assert bias is None or isinstance(bias, bool) or isinstance(bias,
                                                                    Parameter)
        if bias is None or isinstance(bias, Parameter):
            self.bias = bias
        elif bias is False:
            self.bias = None
        else:
            data = torch.empty(out_features, dtype=self.bias_dtype)
            self.bias = Parameter(data, requires_grad=False)

        assert weight_scale is None or isinstance(weight_scale, Parameter)
        if weight_scale is None:
            weight_scale = Parameter(
                torch.ones(out_features, 1, dtype=torch.float32),
                requires_grad=False)
        self.weight_scale = weight_scale

    def forward(self, x):
        return self.torch_forward(x)

    def torch_forward(self, x):
        m = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], self.out_features]

        qinput, x_scale = scaled_fp8_quant(
            m,
            scale=None,
            use_per_token_if_dynamic=True)
        # TODO:  use vllm.cutlass_scaled_mm on sm89 for beffer performance
        output = torch._scaled_mm(qinput,
                                  self.weight,
                                  scale_a=x_scale,
                                  scale_b=self.weight_scale.data.t(),
                                  out_dtype=x.dtype,
                                  use_fast_accum=True)
        if self.bias is not None:
            output = output + self.bias
        return output.view(*output_shape)


    @staticmethod
    def merge(linears):
        assert len(linears) > 1
        in_features = []
        out_features = []
        dtype = None
        device = None
        weights = []
        biases = []
        scales = []
        for linear in linears:
            in_features.append(linear.in_features)
            out_features.append(linear.out_features)
            weight = linear.weight.data
            dtype = weight.dtype
            device = weight.device
            weights.append(weight.view(torch.int8))
            biases.append(None if linear.bias is None else linear.bias.data)
            scales.append(linear.weight_scale.data)

        assert min(in_features) == max(in_features)

        weight = Parameter(torch.cat(weights, dim=0).view(dtype).t(),
                           requires_grad=False)
        bias = None if any([b is None for b in biases]) else Parameter(
            torch.cat(biases, dim=0), requires_grad=False)
        scale = Parameter(torch.cat(scales, dim=0))
        in_features = in_features[0]
        out_features = sum(out_features)
        config = linears[0].config

        return DynamicW8A8Fp8Linear(in_features,
                                    out_features,
                                    weight=weight,
                                    bias=bias,
                                    input_scale=None,
                                    weight_scale=scale,
                                    device=device,
                                    weight_dtype=None,
                                    bias_dtype=None,
                                    config=config)

    def patch(self):
        self.weight = Parameter(self.weight.t(), requires_grad=False)

    def __repr__(self):
        return (f'DynamicW8A8Fp8Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')


class StaticW8A8Fp8Linear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 weight=None,
                 bias=None,
                 dtype=None,
                 input_scale=None,
                 weight_scale=None,
                 device=None,
                 weight_dtype=None,
                 bias_dtype=None,
                 config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.device = device
        self.weight_dtype = torch.float8_e4m3fn
        self.bias_dtype = bias_dtype or dtype
        self.config = config

        assert weight is None or isinstance(weight, Parameter)
        if weight is None:
            data = torch.empty(out_features, in_features,
                               dtype=self.weight_dtype)
            self.weight = Parameter(data, requires_grad=False)
        else:
            self.weight = weight

        assert bias is None or isinstance(bias, bool) or isinstance(bias,
                                                                    Parameter)
        if bias is None or isinstance(bias, Parameter):
            self.bias = bias
        elif bias is False:
            self.bias = None
        else:
            data = torch.empty(out_features, dtype=self.bias_dtype)
            self.bias = Parameter(data, requires_grad=False)

        assert weight_scale is None or isinstance(weight_scale, Parameter)
        if weight_scale is None:
            weight_scale = Parameter(torch.empty(1, dtype=torch.float32),
                                     requires_grad=False)

        self.weight_scale = weight_scale

        assert input_scale is None or isinstance(input_scale, Parameter)
        if input_scale is None:
            input_scale = Parameter(torch.empty(1, dtype=torch.float32),
                                    requires_grad=False)
        self.input_scale = input_scale

    def forward(self, x):
        return self.torch_forward(x)

    def torch_forward(self, x):

        m = x.view(-1, x.shape[-1])
        output_shape = [*x.shape[:-1], self.out_features]

        qinput, x_scale = scaled_fp8_quant(
            m,
            scale=self.input_scale)
        output = torch._scaled_mm(qinput,
                                  self.weight,
                                  scale_a=x_scale,
                                  scale_b=self.weight_scale.data,
                                  out_dtype=x.dtype,
                                  use_fast_accum=True)
        if self.bias is not None:
            output = output + self.bias
        return output.view(*output_shape)

    @staticmethod
    def merge(linears):
        assert len(linears) > 1
        in_features = []
        out_features = []
        dtype = None
        device = None
        weights = []
        biases = []
        scales = []
        input_scales = []
        for linear in linears:
            in_features.append(linear.in_features)
            out_features.append(linear.out_features)
            weight = linear.weight.data
            dtype = weight.dtype
            device = weight.device
            weights.append(weight.view(torch.int8))
            biases.append(None if linear.bias is None else linear.bias.data)
            scales.append(linear.weight_scale.data)
            input_scales.append(linear.input_scale.data)

        assert min(in_features) == max(in_features)

        scale = max(scales)
        scale = Parameter(scale, requires_grad=False)
        weight = torch.cat(
            [(x.float() * (scales[i] / scale)) for i, x in enumerate(weights)],
            dim=0).to(dtype)
        weight = Parameter(weight.t(), requires_grad=False)
        bias = None if any([b is None for b in biases]) else Parameter(
            torch.cat(biases, dim=0), requires_grad=False)
        input_scale = Parameter(max(input_scales))
        in_features = in_features[0]
        out_features = sum(out_features)
        config = linears[0].config

        return StaticW8A8Fp8Linear(in_features,
                                   out_features,
                                   weight=weight,
                                   bias=bias,
                                   input_scale=input_scale,
                                   weight_scale=scale,
                                   device=device,
                                   weight_dtype=None,
                                   bias_dtype=None,
                                   config=config)

    @staticmethod
    def _convert_to_channelwise(
            weight_scale,
            logical_widths):
        # Create channelwise buffer
        weight_scale_channel = torch.empty((sum(logical_widths), 1),
                                           dtype=torch.float32,
                                           device=weight_scale[0].device)

        # Expand each scale to match the size of each logical matrix.
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_scale_channel[start:end, :] = weight_scale[idx]
            start = end

        return weight_scale_channel

    def patch(self):
        self.weight = Parameter(self.weight.t(), requires_grad=False)

    def __repr__(self):
        return (f'StaticW8A8Fp8Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')


class DynamicW8A8Int8Linear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 weight=None,
                 bias=None,
                 dtype=None,
                 input_scale=None,
                 weight_scale=None,
                 device=None,
                 weight_dtype=None,
                 bias_dtype=None,
                 config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.device = device
        self.weight_dtype = torch.int8
        self.bias_dtype = bias_dtype or dtype
        self.config = config

        assert weight is None or isinstance(weight, Parameter)
        if weight is None:
            data = torch.empty(out_features, in_features,
                               dtype=self.weight_dtype)
            self.weight = Parameter(data, requires_grad=False)
        else:
            self.weight = weight

        assert bias is None or isinstance(bias, bool) or isinstance(bias,
                                                                    Parameter)
        if bias is None or isinstance(bias, Parameter):
            self.bias = bias
        elif bias is False:
            self.bias = None
        else:
            data = torch.empty(out_features, dtype=self.bias_dtype)
            self.bias = Parameter(data, requires_grad=False)

        assert weight_scale is None or isinstance(weight_scale, Parameter)
        if weight_scale is None:
            weight_scale = Parameter(
                torch.empty(out_features, 1, dtype=torch.float32),
                requires_grad=False)
        self.weight_scale = weight_scale

    def forward(self, x):
        x_q, x_scale, _ = ops.scaled_int8_quant(x,
                                                None,
                                                None,
                                                symmetric=True)

        return ops.cutlass_scaled_mm(x_q,
                                     self.weight,
                                     scale_a=x_scale,
                                     scale_b=self.weight_scale,
                                     out_dtype=x.dtype,
                                     bias=self.bias)

    @staticmethod
    def merge(linears):
        assert len(linears) > 1
        in_features = []
        out_features = []
        dtype = None
        device = None
        weights = []
        biases = []
        scales = []
        for linear in linears:
            in_features.append(linear.in_features)
            out_features.append(linear.out_features)
            weight = linear.weight.data
            dtype = weight.dtype
            device = weight.device
            weights.append(weight)
            biases.append(None if linear.bias is None else linear.bias.data)
            scales.append(linear.weight_scale.data)

        assert min(in_features) == max(in_features)

        weight = Parameter(torch.cat(weights, dim=0).t(), requires_grad=False)
        bias = None if any([b is None for b in biases]) else Parameter(
            torch.cat(biases, dim=0), requires_grad=False)
        scale = Parameter(torch.cat(scales, dim=0))
        in_features = in_features[0]
        out_features = sum(out_features)
        config = linears[0].config

        return DynamicW8A8Int8Linear(in_features,
                                     out_features,
                                     weight=weight,
                                     bias=bias,
                                     input_scale=None,
                                     weight_scale=scale,
                                     device=device,
                                     weight_dtype=None,
                                     bias_dtype=None,
                                     config=config)

    def patch(self):
        self.weight = Parameter(self.weight.t(), requires_grad=False)

    def __repr__(self):
        return (f'DynamicW8A8Int8Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')


class StaticW8A8Int8Linear(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 weight=None,
                 bias=None,
                 dtype=None,
                 input_scale=None,
                 weight_scale=None,
                 device=None,
                 weight_dtype=None,
                 bias_dtype=None,
                 config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = weight
        self.bias = bias
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.device = device
        self.weight_dtype = torch.int8
        self.bias_dtype = bias_dtype or dtype
        self.config = config

        assert weight is None or isinstance(weight, Parameter)
        if weight is None:
            data = torch.empty(out_features, in_features,
                               dtype=self.weight_dtype)
            self.weight = Parameter(data, requires_grad=False)
        else:
            self.weight = weight

        assert bias is None or isinstance(bias, bool) or isinstance(bias,
                                                                    Parameter)
        if bias is None or isinstance(bias, Parameter):
            self.bias = bias
        elif bias is False:
            self.bias = None
        else:
            data = torch.empty(out_features, dtype=self.bias_dtype)
            self.bias = Parameter(data, requires_grad=False)

        assert weight_scale is None or isinstance(weight_scale, Parameter)
        if weight_scale is None:
            weight_scale = Parameter(torch.empty(1, dtype=torch.float32),
                                     requires_grad=False)

        self.weight_scale = weight_scale

        assert input_scale is None or isinstance(input_scale, Parameter)
        if input_scale is None:
            input_scale = Parameter(torch.empty(1, dtype=torch.float32),
                                    requires_grad=False)
        self.input_scale = input_scale

    def forward(self, x):
        x_q, x_scale, _ = ops.scaled_int8_quant(x,
                                                self.input_scale,
                                                None,
                                                symmetric=True)

        return ops.cutlass_scaled_mm(x_q,
                                     self.weight,
                                     scale_a=x_scale,
                                     scale_b=self.weight_scale,
                                     out_dtype=x.dtype,
                                     bias=self.bias)

    @staticmethod
    def merge(linears):
        assert len(linears) > 1
        in_features = []
        out_features = []
        dtype = None
        device = None
        weights = []
        biases = []
        scales = []
        input_scales = []
        for linear in linears:
            in_features.append(linear.in_features)
            out_features.append(linear.out_features)
            weight = linear.weight.data
            dtype = weight.dtype
            device = weight.device
            weights.append(weight)
            biases.append(None if linear.bias is None else linear.bias.data)
            scales.append(linear.weight_scale.data)
            input_scales.append(linear.input_scale.data)

        assert min(in_features) == max(in_features)

        scale = max(scales)
        scale = Parameter(scale, requires_grad=False)
        weight = torch.cat(
            [(x.float() * (scales[i] / scale)) for i, x in enumerate(weights)],
            dim=0).to(dtype)
        weight = Parameter(weight.t(), requires_grad=False)
        bias = None if any([b is None for b in biases]) else Parameter(
            torch.cat(biases, dim=0), requires_grad=False)
        input_scale = Parameter(max(input_scales))
        in_features = in_features[0]
        out_features = sum(out_features)
        config = linears[0].config

        return StaticW8A8Int8Linear(in_features,
                                    out_features,
                                    weight=weight,
                                    bias=bias,
                                    input_scale=input_scale,
                                    weight_scale=scale,
                                    device=device,
                                    weight_dtype=None,
                                    bias_dtype=None,
                                    config=config)

    def patch(self):
        self.weight = Parameter(self.weight.t(), requires_grad=False)

    def __repr__(self):
        return (f'StaticW8A8Int8Linear(in_features={self.in_features}, '
                f'out_features={self.out_features}, bias={self.bias is not None})')
