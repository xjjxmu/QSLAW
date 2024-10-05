import argparse
from collections import defaultdict
import torch
# 原始参数字典
parser = argparse.ArgumentParser(description='Process parameter dictionary file.')
parser.add_argument('input_file', type=str, help='Path to the input parameter dictionary file')
args = parser.parse_args()

param_dict = torch.load(args.input_file)
grouped_params = {}

for key, value in param_dict.items():
    layer_num = int(key.split('layers.')[1].split('.')[0])
    
    if layer_num not in grouped_params:
        grouped_params[layer_num]={}

    suffix = key.split(f'model.layers.{layer_num}.')[1]
    grouped_params[layer_num][suffix] = value

torch.save(grouped_params, args.input_file)
