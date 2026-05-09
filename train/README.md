
%cd train
!python train_ray_lora.py \
    --clip_model       openai/clip-vit-base-patch32 \
    --llm_model        ibm-granite/granite-4.1-3b \
    --dataset          HuggingFaceM4/Docmatix \
    --dataset_config   images \
    --dataset_split    train \
    --image_col        images \
    --text_col         texts \
    --text_subfield    assistant \
    --train_samples    5000 \
    --streaming \
    --val_dataset      HuggingFaceM4/FineVision \
    --val_split        train \
    --val_samples      100 \
    --lora_targets     q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --dtype            bf16 \
    --gpus_per_worker  1 \
    --cpus_per_worker  6 \
    --num_workers 1 \
    --epochs 4 \
    --batch_size 4