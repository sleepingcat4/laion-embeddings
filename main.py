from typing import Union
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from search_embeddings import search_embeddings
from create_embeddings import create_embeddings

from pydantic import BaseModel

class LoadIndexRequest(BaseModel):
    dataset: str
    faiss_index: str

class SearchRequest(BaseModel):
    text: str

class CreateIndexRequest(BaseModel):
    resources: dict
    metadata: dict  

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

# @app.post("/create")
# def create_index_post(request: CreateIndexRequest):
#     resources = request.resources
#     metadata = request.metadata
#     index_dataset = create_embeddings.create_embeddings(resources, metadata)
#     index_dataset.main(metadata.dataset, metadata.coulmn, metadata.dst_path, metadata.models)
#     return None


async def load_index_task(dataset: str, faiss_index: str):
    await vector_search.load_qdrant(dataset, faiss_index)
    return None

@app.post("/load")
def load_index_post(request: LoadIndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(load_index_task, request.dataset, request.faiss_index)
    return {"message": "Index loading started in the background"}

@app.post("/search")
def search_item_post(request: SearchRequest):
    return vector_search.search(request.text)

uvicorn.run(app, host="0.0.0.0", port=9999)