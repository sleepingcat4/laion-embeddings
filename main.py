from typing import Union
import uvicorn
from fastapi import FastAPI
from search_embeddings import search_embeddings
from create_embeddings import create_embeddings

from pydantic import BaseModel

class LoadIndexRequest(BaseModel):
    dataset: str
    faiss_index: str

metadata = {
    "dataset": "laion/Wikipedia-X-Concat",
    "faiss_index": "laion/Wikipedia-M3",
    "model": "BAAI/bge-m3"
}
resources = {
    "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",1]],
    "libp2p_endpoints": []
}
vector_search = search_embeddings.search_embeddings(resources, metadata)
app = FastAPI(port=9999)

@app.post("/create")
def create_index_post(request: LoadIndexRequest):
    resources = request.resources
    metadata = request.metadata
    index_dataset = create_embeddings.create_embeddings(resources, metadata)
    index_dataset.main(metadata.dataset, metadata.coulmn, metadata.dst_path, metadata.models)
    return 

@app.get("/search/{text}")
def read_item(text: str, q: Union[str, None] = None):
    return vector_search.search(text)

@app.post("/search")
def read_item_post(request: LoadIndexRequest):
    return vector_search.search(request.text)

@app.get("/load/{dataset}/{faiss_index}")
def load_index(dataset: str, faiss_index: str):
    return vector_search.load_qdrant(dataset, faiss_index)

@app.post("/load")
def load_index_post(request: LoadIndexRequest):
    return vector_search.load_qdrant(request.dataset, request.faiss_index)

uvicorn.run(app, host="0.0.0.0", port=9999)