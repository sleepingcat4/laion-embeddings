from typing import Union
import uvicorn
from fastapi import FastAPI
from search_embeddings import search_embeddings
metadata = {
    "dataset": "laion/Wikipedia-X-Concat",
    "faiss_index": "laion/Wikipedia-M3",
    "model": "BAAI/bge-m3"
}
resources = {}
vector_search = search_embeddings.search_embeddings(resources, metadata)
app = FastAPI(port=9999)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/search/{text}")
def read_item(text: str, q: Union[str, None] = None):
    return vector_search.search(text)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9999)