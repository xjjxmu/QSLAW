import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math
CLIPMIN = 1e-5
from scipy.stats import norm
def create_normal_map(offset=0.9677083, symmetric=False, num_bits = 4):
    variations = 2**num_bits
    # print("Doing normal float quantization")
    if symmetric == True:
        v = norm.ppf(torch.linspace(1 - offset, offset, variations + 1)).tolist()
        values = []
        for index in range(len(v) - 1):
            values.append(0.5 * v[index] + 0.5 * v[index + 1])
        v = values
    else:
        v1 = norm.ppf(torch.linspace(offset, 0.5, variations // 2 + 1)[:-1]).tolist()
        v2 = [0]
        v3 = (-norm.ppf(torch.linspace(offset, 0.5, variations // 2)[:-1])).tolist()
        v = v1 + v2 + v3


    values = torch.Tensor(v)
    values = values.sort().values
    values /= values.max()
    return values
    # assert values.

def quantize_tensor(X, L):

    X_expanded = X.unsqueeze(-1)

    # Reshape L to have the same number of dimensions as X_expanded
    L = L.to(X.device)
    L_reshaped = torch.tensor(L).reshape(1, -1)

    # Calculate the absolute difference between X_expanded and L_reshaped
    abs_diff = torch.abs(X_expanded - L_reshaped)

    # Find the index of the minimum absolute difference for each element
    min_index = torch.argmin(abs_diff, dim=-1)
    # print(min_index)
    return L[min_index]


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
        disable_zero_point=False,
        args=None,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        
        init_value = 4.             # inti value of learnable weight clipping
        self.quant_method = 'uniform'    # quantization method, nf for nf4
        if args.resume is not None:
            self.quant_method = 'uniform'
        if self.quant_method=='uniform' and lwc:
            if group_size:
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size
        self.group_num  = int(shape[0]*math.ceil(shape[1]/group_size))
        self.mode = 'calibration'
            

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        if self.disable_zero_point:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        
        if hasattr(self, 'dynamic_scale'):
            temp = scale * self.dynamic_scale
        else:
            temp = scale

        x_int = round_ste(x / temp)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant
    

    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)
  
        if self.quant_method == 'nf':
            if self.mode == 'calibration':
                self.data_format = create_normal_map(num_bits=self.n_bits).to(x.dtype)
            return self.quant_nf4_block(x)
        else:
            if ((self.dynamic_method == "per_token" or self.dynamic_method == "per_channel")) and self.mode == 'calibration':
                self.per_token_dynamic_calibration(x)
            if self.mode=='calibration':
                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
            else:
                x_dequant = self.fake_quant(x, self.scales, self.zeros) #? w1
            return x_dequant

    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def quant_nf4_block(self, weight, block_size=128):
        def quant_nf4(weight):
            max_abs = torch.abs(weight).max()
            weight_divabs = weight / max_abs
            self.data_format = self.data_format if self.data_format.dtype == weight.dtype else self.data_format.to(weight.dtype)
            weights_divabs = quantize_tensor(weight_divabs, self.data_format)
            return weights_divabs * max_abs
        # print(f"nf4 quantize by block with block size {block_size} using {num_bits}bits")
        weight_resize = weight.resize(weight.shape[0]*weight.shape[1]//block_size,block_size)
        if hasattr(self, 'dynamic_scale'):
            weight_resize = weight_resize * self.dynamic_scale + 1e-8 

        quant_block = torch.vmap(quant_nf4, out_dims=0)
        return quant_block(weight_resize).view(weight.shape[0],weight.shape[1])

    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point
    
    def register_scales_and_zeros_params(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)

        del self.scale
        del self.round_zero_point