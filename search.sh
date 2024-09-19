text="orange juice"
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice"}' \
    -H 'Content-Type: application/json'

