dataset=laion/Wikipedia-X-Concat
faiss_index=laion/Wikipedia-M3
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "faiss_index":"laion/Wikipedia-M3"}' \
    -H 'Content-Type: application/json'

