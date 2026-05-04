# !ray start --head --port=6380 --dashboard-host=0.0.0.0
import os
# 1.25e6
totalRows = 1.27e6
percentageToKeep = 0.1

desiredRow = int(totalRows * percentageToKeep)

print(os.cpu_count())
# totalCPU = 8
# workers = max(1, 2)   
totalCPU = 8
workers = 3

local_coreset_size = desiredRow // workers
print(local_coreset_size)
page_size = 2000
sample_size = 256

# ray start --address='10.128.1.67:6380' --num-cpus=4  --num-gpus=1
# ray start --address='10.192.11.15:6380' --num-cpus=4 --num-gpus=1




!export HF_HUB_DOWNLOAD_TIMEOUT=120

<!-- !python -m coreset.train \
    --hf-dataset HuggingFaceM4/Docmatix \
    --hf-split train \
    --hf-image-col images \
    --total-samples {totalRows} \
    --page-size {page_size} \
    --sample-size {sample_size} \
    --final-coreset-size {desiredRow} \
    --local-coreset-size {local_coreset_size} \
    --embed-batch-size 128 \
    --workers {workers} \
    --model openai/clip-vit-base-patch32 \
    --output ./coreset_output \
    --cpus-per-worker 2 \
    --seed 42 -->












!ray start --head --port=6380 --dashboard-host=0.0.0.0
import os

totalRows = 1.27e6
percentageToKeep = 0.1

desiredRow = int(totalRows * percentageToKeep)

print(os.cpu_count())
totalCPU = 8
workers = 2

local_coreset_size = desiredRow // workers +1000
print(local_coreset_size)
page_size = 1000
sample_size = 256


 python train.py \
        --hf-dataset HuggingFaceM4/Docmatix \
        --hf-split train \
        --hf-image-col images \
        --hf-text-col texts \
        --total-samples {totalRows} \
        --page-size {page_size} \
        --sample-size {sample_size} \
        --final-coreset-size {desiredRow} \
        --local-coreset-size {local_coreset_size} \
        --embed-batch-size 128 \
        --workers {workers} \
        --model openai/clip-vit-base-patch32 \
        --output ./coreset_output \
        --gpus-per-worker 0.5 \
        --seed 42 \
        --push-to-hub your-org/docmatix-coreset \
        --translate-model Qwen/Qwen3-4B \
        --tp-size 2 \
        --max-num-seqs 256