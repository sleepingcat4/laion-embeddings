import asyncio
from aiohttp import ClientSession
from datasets import load_dataset
import os
import sys
import datasets
import json
from ipfs_embeddings_py import ipfs_embeddings_py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams
import numpy as np
import os
import sys
import subprocess
import json
import tiktoken

class create_embeddings_batch:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.index =  {}
        if len(list(metadata.keys())) > 0:
            for key in metadata.keys():
                setattr(self, key, metadata[key])
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        if "https_endpoints" in resources.keys():
            for endpoint in resources["https_endpoints"]:
                self.ipfs_embeddings_py.add_https_endpoint(endpoint[0], endpoint[1], endpoint[2])
        else:
            self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed", 8192)
            self.ipfs_embeddings_py.add_https_endpoint("Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://62.146.169.111:80/embed", 32768 )
        self.join_column = None

    async def producer(self, dataset_stream, queues):
        async for item in self.async_generator(dataset_stream):
            # Assuming `item` is a dictionary with required data
            for queue in queues:
                await queue.put(item)  # Non-blocking put

    async def async_generator(self, iterable):
        for item in iterable:
            yield item

    async def consumer(self, queue, batch_size, model_name):
        batch = []
        if model_name not in self.index.keys():
            self.index[model_name] = []
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch(batch, model_name)
                for i in range(len(results)):
                    this_cid = self.ipfs_embeddings_py.index_cid(batch[i]["text"])[0]
                    self.index[model_name].append({this_cid : results[i] })
                batch = []  # Clear batch after sending

    async def send_batch(self, batch, model_name):
        print(f"Sending batch of size {len(batch)} to model {model_name}")
        endpoint = list(self.ipfs_embeddings_py.https_endpoints[model_name].keys())[0]
        model_context_length = self.ipfs_embeddings_py.https_endpoints[model_name][endpoint]
        model_context_length = round(float(model_context_length * 0.75))
        new_batch = []
        for item in batch:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            this_item_tokens = len(self.tokenizer.encode(item["text"]))
            if this_item_tokens > model_context_length:
                encoded_item = self.tokenizer .encode(item["text"])
                truncated_encoded_item = encoded_item[:model_context_length]
                unencode_item = self.tokenizer.decode(truncated_encoded_item)
                new_batch.append(unencode_item)
            else:
                new_batch.append(item["text"])
        results = self.ipfs_embeddings_py.index_knn(new_batch, model_name)
        return results
    
    async def main(self, dataset, faiss_index, model1, model2):
        dataset_exists = False
        try:
            self.faiss_index = self.datasets.load_dataset(faiss_index)
        except:
            self.faiss_index = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        # Load a stream from HuggingFace datasets
        self.dataset = load_dataset(dataset, split='train', streaming=True)
        batch_size_1 = 32
        batch_size_2 = 32
        # batch_size_1 = await self.ipfs_embeddings_py.max_batch_size(model1)
        # batch_size_2 = await self.ipfs_embeddings_py.max_batch_size(model2)
        # Create queues for different models
        self.ipfs_embeddings_py.queue1 = asyncio.Queue(batch_size_1)
        self.ipfs_embeddings_py.queue2 = asyncio.Queue(batch_size_2)
        # Start producer and consumers
        producer_task = asyncio.create_task(self.producer(self.dataset, [self.ipfs_embeddings_py.queue1, self.ipfs_embeddings_py.queue2]))
        consumer_task1 = asyncio.create_task(self.consumer(self.ipfs_embeddings_py.queue1, batch_size_1, model1 ))
        consumer_task2 = asyncio.create_task(self.consumer(self.ipfs_embeddings_py.queue2, batch_size_2, model2))
        await asyncio.gather(producer_task, consumer_task1, consumer_task2)

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "faiss_index": "endomorphosis/Caselaw_Access_Project_M3_Embeddings",
        "model1": "BAAI/bge-m3",
        "model2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
    }
    resources = {
        "https_endpoints": [
            ["BAAI/bge-m3", "http://62.146.169.111:80/embed", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://127.0.0.1:8080/embed", 32768]
        ]
    }
    create_embeddings_batch = create_embeddings_batch(resources, metadata)
    asyncio.run(create_embeddings_batch.main(metadata["dataset"], metadata["faiss_index"], metadata["model1"], metadata["model2"]))

    