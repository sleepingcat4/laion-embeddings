to start an endpoint:
./run.sh

Then to pull the qdrant docker container and load the embeddings into it
curl http://127.0.0.1:9000/load/my_dataset/my_faiss_index

then to search the index
curl http://127.0.0.1:9000/search/the_text_that_i_want_to_find
