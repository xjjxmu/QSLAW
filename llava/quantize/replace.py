import torch
import torch.nn as nn
from llava.quantize.int_llama_layer import QuantLlamaDecoderLayer
from llava.quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import os
import pdb
import gc
from llava.quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,smooth_and_quant_fake, add_scaling, \
                            clear_temp_variable,set_quant_state, NativeScalerWithGradNormCount \



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}

def copy_weights(source, target):
    for src_child, trg_child in zip(source.children(), target.children()):
        if isinstance(src_child, QuantLinear):
            trg_child.weight.data = src_child.weight.data.clone()
            if src_child.bias is not None:
                trg_child.bias.data = src_child.bias.data.clone()
        else:
            copy_weights(src_child, trg_child)

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def replace_quantization(
    lm,
    args,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm._device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" or "vicuna" in args.net.lower():
        is_llama = True
        layers = model.layers
        model.embed_tokens = model.embed_tokens.to(dev)
        model.norm = model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")


    layers[0].is_llama = is_llama
    # layers[0] = layers[0].module

    if "llama" in args.net.lower() or "mixtral" in args.net.lower() or "vicuna" in args.net.lower():
        model.embed_tokens = model.embed_tokens.cpu()
        model.norm = model.norm.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()


    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        raise ValueError("Please supply the initialization for quantization parameter.")
    
    if args.scaling_resume:
        scaling_parameters = torch.load(args.scaling_resume)

    for i in range(len(layers)):
        logger.info(f"=== Start replace origin layer {i} with quantization layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            # for mixtral, we only leverage lwc, which can be achieve by simply replace Linear with QuantLinear
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
                                
        qlayer.load_state_dict(omni_parameters[i], strict=False)
        qlayer = qlayer.to(dev)
        qlayer.half() 
        
        if args.scaling:
            # for visual instruction learning
            if args.scaling_resume is not None and args.tune_mm_mlp_adapter is False:
                if i > args.start_layer:
                    add_scaling(qlayer)
                    qlayer.load_state_dict(scaling_parameters[i], strict=False)
                
                # real quant llms for visual instruction learning 
                smooth_and_quant_inplace(qlayer, args)
                copy_weights(qlayer, layer)
                layers[i] = layer.to("cpu")
            else:    
                # for pretrain with modality-aware warmup
                smooth_and_quant_fake(qlayer, scaling = True if i > args.start_layer else False)
                if args.scaling_resume is not None:
                    if i > args.start_layer:
                        qlayer.load_state_dict(scaling_parameters[i], strict=False)
                layers[i] = qlayer.to("cpu") 
        else:
            # real smooth and quantization without scaling
            smooth_and_quant_inplace(qlayer, args)
            copy_weights(qlayer, layer)  
            layers[i] = layer.to("cpu")
        del layer
        torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

def omniquant_from_checkpoint(
    lm,
    args,
):
    print("Starting ...")
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm._device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    args.net = args.model_base.split('/')[-1]
    if "llama" or "vicuna" in args.net.lower():
        layers = model.layers
        model.embed_tokens = model.embed_tokens.to(dev)
        model.norm = model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer

    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}
    if args.scaling_resume:
        scaling_parameters = torch.load(args.scaling_resume)
    
    for i in range(len(layers)):
        print(f"=== Start loading layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower():  
            qlayer = copy.deepcopy(layer)
            for name, module in qlayer.named_modules():
                if isinstance(module,torch.nn.Linear) and not "gate" in name:       # do not quantize gate
                    quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                    add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)

        qlayer = qlayer.to(layer.device)
        qlayer.half() 
        # real smooth and quantization
        if args.scaling:
            if args.scaling_resume is not None:
                if i > args.start_layer:
                    add_scaling(qlayer)
                    qlayer.load_state_dict(scaling_parameters[i], strict=False)
                   
                smooth_and_quant_inplace(qlayer, args)
                copy_weights(qlayer, layer)
                layer.half() 
            else:
                raise ValueError("Scaling enable! please provide scaling resume")
        else:
            smooth_and_quant_inplace(qlayer, args)
            copy_weights(qlayer, layer)   
        del layer
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

