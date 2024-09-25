To start an endpoint:
./run.sh

this runs a python command 

```
python3 -m fastapi run main.py
```

Then to pull the qdrant docker container and load the embeddings into it
./load.sh

this runs a curl command

```
curl 127.0.0.1:9999/load \
    -X POST \
    -d '{"dataset":"laion/Wikipedia-X-Concat", "knn_index":"laion/Wikipedia-M3", "dataset_split": "enwiki_concat", "knn_index_split": "enwiki_embed", "column": "Concat Abstract"}' \
    -H 'Content-Type: application/json'
```

NOTE: THAT THIS WILL TAKE HOURS TO DOWNLOAD / INGEST FOR LARGE DATASETS
NOTE: FAST API IS UNAVAILABLE WHILE THIS RUNS

Then to search the index
./search.sh 

this runs a curl command

```
curl 127.0.0.1:9999/search \
    -X POST \
    -d '{"text":"orange juice", "collection": "Wikipedia-X-Concat"}' \
    -H 'Content-Type: application/json'
```

To create an index
./create.sh

