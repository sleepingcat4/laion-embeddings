text="The quick brown fox jumps over the lazy dog"
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"The quick brown fox jumps over the lazy dog"}' \
    -H 'Content-Type: application/json'

