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
import transformers
from transformers import AutoTokenizer
import asyncio
import time
class create_embeddings_batch:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.index =  {}
        self.cid_list = []
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
        self.tokenizer = {}

    async def producer(self, dataset_stream, queues):
        async for item in self.async_generator(dataset_stream):
            # Assuming `item` is a dictionary with required data
            column_names = item.keys()
            this_cid = self.ipfs_embeddings_py.index_cid(item["text"])[0]
            if "cid" not in column_names:
                item["cid"] = self.ipfs_embeddings_py.index_cid(item["text"])[0]
            # Check if cid is in index
            if this_cid in self.cid_list:
                pass
            else:
                self.cid_list.append(this_cid)
                self.new_dataset = self.new_dataset.add_item(item)    
                for queue in queues:
                    await queue.put(item)  # Non-blocking put

    async def async_generator(self, iterable):
        for item in iterable:
            yield item

    async def consumer(self, queue, batch_size, model_name):
        batch = []
        if model_name not in self.index.keys():
            self.index[model_name] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch(batch, model_name)
                for i in range(len(results)):
                    this_cid = self.ipfs_embeddings_py.index_cid(batch[i]["text"])[0]
                    self.index[model_name] = self.index[model_name].add_item({"cid": this_cid, "embedding": results[i]})
                batch = []  # Clear batch after sending
                self.saved = False
            

    async def send_batch(self, batch, model_name):
        print(f"Sending batch of size {len(batch)} to model {model_name}")
        endpoint = list(self.ipfs_embeddings_py.https_endpoints[model_name].keys())[0]
        model_context_length = self.ipfs_embeddings_py.https_endpoints[model_name][endpoint]
        model_context_length = round(float(model_context_length * 0.5))
        new_batch = []
        for item in batch:
            if model_name not in self.tokenizer.keys():
                self.tokenizer[model_name] = AutoTokenizer.from_pretrained(model_name)
            this_item_tokens = len(self.tokenizer[model_name].encode(item["text"]))
            if this_item_tokens > model_context_length:
                encoded_item = self.tokenizer[model_name](item["text"], return_tensors="pt")["input_ids"].tolist()[0]
                truncated_encoded_item = encoded_item[:model_context_length]
                unencode_item = self.tokenizer[model_name].decode(truncated_encoded_item)
                new_batch.append(unencode_item)
            else:
                new_batch.append(item["text"])
        results = self.ipfs_embeddings_py.index_knn(new_batch, model_name)
        return results
    
    async def save_to_disk(self, dataset, dst_path, model1, model2):
        self.saved = False
        while True:
            await asyncio.sleep(300)
            if self.ipfs_embeddings_py.queue1.empty() and self.ipfs_embeddings_py.queue1.empty() and self.saved == False:   
                self.new_dataset.save_to_disk(f"{dst_path}/{dataset}.arrow")
                self.new_dataset.to_parquet(f"{dst_path}/{dataset}.parquet")
                self.index[model1].save_to_disk(f"{dst_path}/{model1.replace("/","---")}.arrow")
                self.index[model1].to_parquet(f"{dst_path}/{model1.replace("/","---")}.parquet")
                self.index[model2].save_to_disk(f"/{dst_path}/{model2.replace("/","---")}.arrow")
                self.index[model2].to_parquet(f"{dst_path}/{model2.replace("/","---")}.parquet")
                self.saved = True

    async def main(self, dataset, dst_path, model1, model2):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.dataset = load_dataset(dataset, split='train', streaming=True)
        columns = self.dataset.column_names
        columns.append("cid")
        if os.path.isfile(f"{dst_path}/{dataset}.arrow") == True:
            self.new_dataset = self.datasets.load_dataset(f"{dst_path}/{dataset}.arrow")
            self.cid_list = self.new_dataset["cid"]
        else:
            self.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })
        if os.path.isfile(f"{dst_path}/{model1.replace("/","---")}.arrow") == True:
            self.index[model1] = self.datasets.load_dataset(f"{dst_path}/{model1.replace("/","---")}.arrow")
        else:
            self.index[model1] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        if os.path.isfile(f"{dst_path}/{model2.replace("/","---")}.arrow") == True:
            self.index[model2] = self.datasets.load_dataset(f"{dst_path}/{model2.replace("/","---")}.arrow")
        else:
            self.index[model2] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        batch_size_1 = 32
        batch_size_2 = 32
        # batch_size_1 = await self.ipfs_embeddings_py.max_batch_size(model1)
        # batch_size_2 = await self.ipfs_embeddings_py.max_batch_size(model2)
        # Create queues for different models
        self.ipfs_embeddings_py.queue1 = asyncio.Queue(batch_size_1)
        self.ipfs_embeddings_py.queue2 = asyncio.Queue(batch_size_2)
        # Start producer and consumers
        producer_task = asyncio.create_task(self.producer(self.dataset, [self.ipfs_embeddings_py.queue1, self.ipfs_embeddings_py.queue2]))
        consumer_task1 = asyncio.create_task(self.consumer(self.ipfs_embeddings_py.queue1, batch_size_1, model1))
        consumer_task2 = asyncio.create_task(self.consumer(self.ipfs_embeddings_py.queue2, batch_size_2, model2))
        save_task = asyncio.create_task(self.save_to_disk(dataset, dst_path, model1, model2))
        await asyncio.gather(producer_task, consumer_task1, consumer_task2, save_task)

if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "model1": "BAAI/bge-m3",
        "model2": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "dst_path": "/storage/teraflopai"
    }
    resources = {
        "https_endpoints": [
            ["BAAI/bge-m3", "http://62.146.169.111:80/embed", 8192],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://127.0.0.1:8080/embed", 32768]
        ]
    }
    create_embeddings_batch = create_embeddings_batch(resources, metadata)
    asyncio.run(create_embeddings_batch.main(metadata["dataset"], metadata["dst_path"], metadata["model1"], metadata["model2"]))
    