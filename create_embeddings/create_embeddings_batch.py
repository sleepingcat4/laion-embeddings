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

class create_embeddings_batch:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        if len(list(metadata.keys())) > 0:
            for key in metadata.keys():
                setattr(self, key, metadata[key])
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        if "https_endpoints" in resources.keys():
            for endpoint in resources["https_endpoints"]:
                self.ipfs_embeddings_py.add_https_endpoint(endpoint[0], endpoint[1], endpoint[2])
        else:
            self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed",1)
        self.join_column = None

    async def producer(dataset_stream, queues):
        async for item in dataset_stream:
            # Assuming `item` is a dictionary with required data
            for queue in queues:
                await queue.put(item)  # Non-blocking put

    async def consumer(queue, batch_size, model_name, session):
        batch = []
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                await send_batch(batch, model_name, session)
                batch = []  # Clear batch after sending

    async def send_batch(batch, model_name, session):
        url = f"https://api.example.com/embeddings/{model_name}"
        payload = {'data': batch}
        async with session.post(url, json=payload) as response:
            result = await response.json()
            print(f"Received result for {model_name}: {result}")
            # Process or store result

    async def main():
        # Load a stream from HuggingFace datasets
        dataset_stream = load_dataset('dataset_name', split='train', streaming=True)

        # Create queues for different models
        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()

        # Setup HTTP session
        async with ClientSession() as session:
            # Start producer and consumers
            producer_task = asyncio.create_task(producer(dataset_stream, [queue1, queue2]))
            consumer_task1 = asyncio.create_task(consumer(queue1, 10, 'model1', session))
            consumer_task2 = asyncio.create_task(consumer(queue2, 20, 'model2', session))

            await asyncio.gather(producer_task, consumer_task1, consumer_task2)

if __name__ == "__main__":
    asyncio.run(main())
