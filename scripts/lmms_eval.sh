accelerate launch --gpu_ids 0 \
                --num_processes 1 \
                -m lmms_eval \
                --model llava \
                --model_args "pretrained=./checkpoints/llava-v1.5-vicuna-13b-v1.5-13b-chat-scaling-lora,base=./checkpoints/vicuna-13b-v1.5-quant" \
                --tasks pope \
                --batch_size "auto" \
                --log_samples \
                --log_samples_suffix llava_v1.5_qslaw \
                --output_path ~/qslaw/result/ 