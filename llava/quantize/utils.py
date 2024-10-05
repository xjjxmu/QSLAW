from collections import OrderedDict
from llava.quantize.int_linear import QuantLinear
import torch
from llava.quantize.int_matmul import QuantMatMul
from llava.quantize.transformation import *
from math import inf
import logging
from termcolor import colored
import sys
import os
import time
import subprocess
import re
import torch.nn as nn
@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)




def let_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)  

def get_omni_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def omni_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if module.weight_quantizer.quant_method == 'uniform':
                module.weight_quantizer.register_scales_and_zeros()


def add_scaling(model):
    names = []
    for name, m in model.named_modules():
        if isinstance(m, (QuantLinear)):
            names.append(name)
            m.weight_quantizer.register_parameter('dynamic_scale',
                                    nn.Parameter(torch.ones((m.weight_quantizer.group_num,1)).to(m.weight.device),requires_grad=True))

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     

def smooth_and_quant_temporary(model, args):
    if args.let:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                model.qkv_smooth_scale,model.qkv_smooth_shift)
        smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                model.fc1_smooth_scale,model.fc1_smooth_shift)
        smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                            model.out_smooth_scale, model.out_smooth_shift)
        smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                            model.qkt_smooth_scale)
        model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def smooth_and_quant_inplace(model, args):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False
    register_scales_and_zeros(model)

@torch.no_grad()
def smooth_and_quant_fake(model,scaling=False):
    quant_fn = 'nf'
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.use_temporary_parameter=False
            module.set_quant_state(weight_quant = True, act_quant = False)
            temp = module.weight.data
            module.weight = module.weight_quantizer(module.weight)
            module.weight.data = temp
            module.weight_quantizer.mode = 'training'
            if module.weight_quantizer.quant_method == 'uniform':
                quant_fn='uniform'
                del module.weight_quantizer.upbound_factor
                del module.weight_quantizer.lowbound_factor
    if quant_fn == 'uniform':
        register_scales_and_zeros(model)
    if scaling:
        add_scaling(model)
    

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)


def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}_{int(time.time())}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

def get_lowest_occupied_gpu(wait_memory=1000):

    now_lowest_memory = 1e9
    while now_lowest_memory > wait_memory:
        if not now_lowest_memory == 1e9:
            time.sleep(10)
        memory_info = get_gpu_memory()
        gpu_id, tot_mem, used_mem = sorted(
            memory_info, key=lambda x: x[2], reverse=False
        )[0]
        now_lowest_memory = used_mem

    return gpu_id

def nvidia_smi_memory_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    output = result.stdout.split("\n")[:-1]

    gpu_memory_info = []
    for line in output:
        gpu_id, total_memory, used_memory, free_memory = map(int, re.split(",\s", line))
        gpu_memory_info.append(
            {
                "id": gpu_id,
                "total_memory": total_memory,
                "used_memory": used_memory,
                "free_memory": free_memory,
            }
        )

    return gpu_memory_info

def get_gpu_memory():
    memory_info = []
    gpu_memory_info = nvidia_smi_memory_info()

    try:
        gpu_index = [int(k) for k in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        gpu_index = [x["id"] for x in gpu_memory_info]

    for gpu_id, i in enumerate( gpu_index):
        gpu = gpu_memory_info[i]
        total_memory = gpu["total_memory"]
        used_memory = gpu["used_memory"]
        memory_info.append((gpu_id, total_memory, used_memory))
    return memory_info



