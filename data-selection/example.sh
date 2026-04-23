import os
# 1.27e6
totalRows = int(10000)
percentageToKeep = 0.1

desiredRow = int(totalRows * percentageToKeep)

totalCPU = os.cpu_count()
workers = max(1, totalCPU - 2)   # avoid 0 or negative

local_coreset_size = desiredRow // workers
shuffle_buffer = 2000

!export HF_HUB_DOWNLOAD_TIMEOUT=120

!python -m coreset.train \
    --hf-dataset HuggingFaceM4/Docmatix \
    --hf-split train \
    --hf-image-col images \
    --total-samples {totalRows} \
    --shuffle-buffer {shuffle_buffer} \
    --final-coreset-size {desiredRow} \
    --local-coreset-size {local_coreset_size} \
    --batch-size 128 \
    --embed-batch-size 128 \
    --workers {workers} \
    --total-cpus {totalCPU} \
    --gpus 1 \
    --model openai/clip-vit-base-patch32 \
    --output ./coreset_output \
    --seed 42