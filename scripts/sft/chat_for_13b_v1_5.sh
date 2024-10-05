MODEL_VERSION=vicuna-13b-v1.5
gpu_vis=0,1,2,3
MASTER_PORT=2142
PROMPT_VERSION=v1


# We have perform modality-aware warmup during pretrain and the we use mix data to optimize scaling
deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --bits 4 \
    --lwc \
    --resume /path/to/quantization_initialization/omni_parameters.pth \
    --net $MODEL_VERSION \
    --mix \
    --scaling \
    --start_layer 5 \
    --tune_mm_mlp_adapter True \
    --version $PROMPT_VERSION \
    --data_path /path/to/dataset/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-$MODEL_VERSION-pretrain-scaling/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/scaling/$MODEL_VERSION/ \
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
    --report_to wandb \


python ./modify.py ./checkpoints/scaling/$MODEL_VERSION/scaling.bin


# Visual instruction learning
deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --net $MODEL_VERSION \
    --bits 4 \
    --lwc \
    --resume /path/to/quantization_initialization/omni_parameters.pth \
    --scaling \
    --start_layer 5 \
    --scaling_resume ./checkpoints/scaling/$MODEL_VERSION/scaling.bin \
    --data_path /path/to/dataset/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/scaling/$MODEL_VERSION/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-$MODEL_VERSION-13b-chat-qslaw \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb


