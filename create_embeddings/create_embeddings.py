import asyncio
from aiohttp import ClientSession
from datasets import load_dataset
import os
import sys
import datasets
from ipfs_embeddings_py import ipfs_embeddings_py
import os
import sys
import subprocess
from transformers import AutoTokenizer

class create_embeddings:
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

    async def main(self, dataset, column, dst_path, models):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        self.ipfs_embeddings_py.queues = {}
        consumer_tasks = {}
        batch_sizes = {}
        self.dataset = load_dataset(dataset, split='train', streaming=True)
        columns = self.dataset.column_names
        columns.append("cid")
        if os.path.isfile(f"{dst_path}/{dataset}.arrow") == True:
            self.new_dataset = self.datasets.load_dataset(f"{dst_path}/{dataset}.arrow")
            self.cid_list = self.new_dataset["cid"]
        else:
            self.ipfs_embeddings_py.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })

        for model in models:
            batch_size = 32
            if os.path.isfile(f"{dst_path}/{model.replace("/","---")}.arrow") == True:
                self.index[model] = self.datasets.load_dataset(f"{dst_path}/{model.replace("/","---")}.arrow")
            else:
                self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
                self.ipfs_embeddings_py.queues[model] = asyncio.Queue(batch_size)
                consumer_tasks[model] = asyncio.create_task(self.ipfs_embeddings_py.consumer(self.ipfs_embeddings_py.queues[model], column, batch_size, model))

        producer_task = asyncio.create_task(self.ipfs_embeddings_py.producer(self.dataset, column, self.ipfs_embeddings_py.queues))        
        save_task = asyncio.create_task(self.ipfs_embeddings_py.save_to_disk(dataset, dst_path, models))
        await asyncio.gather(producer_task, save_task, *consumer_tasks.values()) 
        return None
    
if __name__ == "__main__":
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "column": "text",
        "models": [
            "BAAI/bge-m3",
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            # "dunzhang/stella_en_1.5B_v5",
        ],
        "dst_path": "/storage/teraflopai"
    }
    resources = {
        "https_endpoints": [
            ["BAAI/bge-m3", "http://62.146.169.111:80/embed", 8191],
            ["Alibaba-NLP/gte-Qwen2-1.5B-instruct", "http://127.0.0.1:8080/embed", 32786],
            ["dunzhang/stella_en_1.5B_v5", "http://127.0.0.1:8080/embed", 131072]
        ]
    }
    create_embeddings_batch = create_embeddings(resources, metadata)
    asyncio.run(create_embeddings_batch.main(metadata["dataset"], metadata["column"], metadata["dst_path"], metadata["models"]))    


