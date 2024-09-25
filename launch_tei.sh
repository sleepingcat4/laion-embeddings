## assumes 2 3090 GPUs
volume=/storage/hf_models
model=BAAI/bge-m3
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model &
docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8081:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model &
model=Alibaba-NLP/gte-Qwen2-1.5B-instruct
sleep 15 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8082:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 &
sleep 15 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8083:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 32768 &
model=dunzhang/stella_en_1.5B_v5
sleep 30 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=0  -p 8084:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 131072 &
sleep 30 ; docker run --gpus all -e CUDA_VISIBLE_DEVICES=1  -p 8085:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.5 --model-id $model --max-batch-tokens 131072 &

