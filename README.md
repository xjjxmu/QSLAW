# QSLAW: Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation

This repository contains the implementation of the ACM MM2024 paper: QSLAW: Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation [[Paper]](https://arxiv.org/abs/2408.03735). It will be orginized as follow: 

- [Setup](#setup)
- [Training](#train)
- [Evaluation](#eval)
- [Model Zoo](#model)

## <a id="setup"></a>Setup
1. Clone this repository and navigate to QSLAW folder
```
git clone https://github.com/xjjxmu/QSLAW.git
cd QSLAW
```

2. Install Package
```
conda create -n qslaw python=3.8 -y
conda activate qslaw

pip install --upgrade pip
pip install -e .
```

3. Data Preparation: You can follow this [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) in LLaVA.

## <a id="train"></a>Training

### Pretrain
We use the 558K subset of the LAION-CC-SBU dataset with BLIP captions to pretrain.
Training script with DeepSpeed ZeRO-2:`scripts/pretrain/pretrain.sh`

### Visual instruction learning
We follow LLaVA to use mixture instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json) where you need to download images according to LLaVA. After orginizing all of them, you can run training script: `scripts/sft/chat_for_13b_v1_5.sh`.


### Fine-tuning
You can also use QSLAW to finetune multimodal llms on downstream datasets after pretraining. To reproduce the performance of QSLAW on ScienceQA, you can run this script: `scripts/sqa_for_13b.sh`

## <a id="eval"></a>Evaluation
1. You can follow the [instruction](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) in LLaVA to prepare data and run scripts.
2. You can install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and we provide the script: `scripts/lmms_eval.sh`


## <a id="model"></a>Model Zoo
We release all potential components in [[Huggingface]](https://huggingface.co/USKM) you may use to reproduce the result on LLaVA-QSLAW-chat-vicuna-13b-v1.5: 
- Quantized language models: [Quantized language model](https://huggingface.co/USKM/vicuna-13b-v1.5-quant/tree/main). During visual instruction learning or finetuning,  LLMs would be frozen and you can directly load the checkpoint where quantization parameters and scaling optimized by qslaw have been absorbed into weights.
- The final model: [LLaVA-QSLAW ](https://huggingface.co/USKM/LLaVA-QSLAW-chat-vicuna-13b-v1.5/tree/main). We use lora to perform visual instruction learning based on frozen quantized language models. 
- Quantization initilization: [Quantization parameters](https://drive.google.com/drive/folders/1MIYw0ACZjFd5NyBMEauEP7p37aHLjjVK?dmr=1&ec=wgc-drive-hero-goto) . Omniquant is used to initialize quantization parameters (step size、zero-point) and qslaw optimize scaling to alleviate quantization error caused by multimodal inputs. 

# Citation
If you find this project useful, please cite our work:
```
@misc{xie2024advancingmultimodallargelanguage,
      title={Advancing Multimodal Large Language Models with Quantization-Aware Scale Learning for Efficient Adaptation}, 
      author={Jingjing Xie and Yuxin Zhang and Mingbao Lin and Liujuan Cao and Rongrong Ji},
      year={2024},
      eprint={2408.03735},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.03735}, 
}
```

# Acknowledgement

[LLaVA](https://github.com/haotian-liu/LLaVA) is the codebase we built upon, allowing us to easily perform visual instruction learning and we borrows some codes from [OmniQuant](https://github.com/OpenGVLab/OmniQuant), an practical post-training quantization method for large language models. [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is a awesome tool to evaluate multimodal llms with many benchmarks. Thanks for their great works.


