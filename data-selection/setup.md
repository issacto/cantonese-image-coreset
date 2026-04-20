


coreset-select
cantonese-translate
image-training (fine tune + training)
verification


!pwd
!python -m coreset.train \
        --hf-dataset HuggingFaceM4/Docmatix \
        --hf-split train \
        --hf-image-col images \
        --total-samples 100000 \
        --shuffle-buffer 2000 \
        --final-coreset-size 1000 \
        --local-coreset-size 500 \
        --batch-size 256 \
        --embed-batch-size 16 \
        --workers 2 \
        --gpus 1 \
        --gpu-per-worker 0.5 \
        --model openai/clip-vit-base-patch32 \
        --output ./coreset_output \
        --seed 42