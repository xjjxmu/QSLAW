MODEL_VERSION=vicuna-13b-v1.3
gpu_vis=1,2,3,4
MASTER_PORT=2342
PROMPT_VERSION=v1

# We have perform modality-aware warmup during pretrain and the we use mix data to optimize scaling
deepspeed --include localhost:$gpu_vis llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/vicuna-13b-v1.3 \
    --bits 4 \
    --lwc \
    --epochs 0 \
    --mix \
    --resume ./checkpoints/llava-vicuna-13b-v1.3-pretrain_lcs558k_plain-ScienceQA_QCM_LEA-omni-lora-12e/omni_parameters.pth \
    --scaling \
    --start_layer 5 \
    --tune_mm_mlp_adapter \
    --net vicuna-13b-v1.3 \
    --version $PROMPT_VERSION \
    --data_path ~/data/llava_train_QCM-LEA.json \
    --image_folder ~/data/images/train \
    --vision_tower ./checkpoints/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain-scaling/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/scaling/vicuna-13b-v1.3/rank256 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \

# Finetune 
deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/vicuna-13b-v1.3 \
    --lora_enable True \
    --bits 4 \
    --lwc \
    --epochs 0 \
    --resume ./checkpoints/llava-vicuna-13b-v1.3-pretrain_lcs558k_plain-ScienceQA_QCM_LEA-omni-lora-12e/omni_parameters.pth \
    --scaling \
    --start_layer 5 \
    --scaling_resume ./checkpoints/scaling/vicuna-13b-v1.3/rank256/scaling.bin \
    --net vicuna-13b-v1.3 \
    --version $PROMPT_VERSION \
    --data_path ~/data/llava_train_QCM-LEA.json \
    --image_folder ~/data/images/train \
    --vision_tower ./checkpoints/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/scaling/vicuna-13b-v1.3/rank256/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-13b-sqa-finetune-scaling-lora_ranklow \
    --num_train_epochs 12 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \