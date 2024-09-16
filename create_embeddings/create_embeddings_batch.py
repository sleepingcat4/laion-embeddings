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

    async def producer(self, dataset_stream, column, queues):
        async for item in self.async_generator(dataset_stream):
            # Assuming `item` is a dictionary with required data
            column_names = item.keys()
            this_cid = self.ipfs_embeddings_py.index_cid(item[column])[0]
            if "cid" not in column_names:
                item["cid"] = self.ipfs_embeddings_py.index_cid(item[column])[0]
            # Check if cid is in index
            if this_cid in self.cid_list:
                pass
            else:
                self.cid_list.append(this_cid)
                self.new_dataset = self.new_dataset.add_item(item)    
                for queue in queues.values():
                    await queue.put(item)  # Non-blocking put
        return None

    async def async_generator(self, iterable):
        for item in iterable:
            yield item

    async def consumer(self, queue, column, batch_size, model_name):
        batch = []
        if model_name not in self.index.keys():
            self.index[model_name] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        while True:
            item = await queue.get()  # Wait for item
            batch.append(item)
            if len(batch) >= batch_size:
                # Process batch
                results = await self.send_batch(batch, column, model_name)
                for i in range(len(results)):
                    self.index[model_name] = self.index[model_name].add_item({"cid": batch[i]["cid"], "embedding": results[i]})
                batch = []  # Clear batch after sending
                self.saved = False
        return None
                
    async def send_batch(self, batch, column, model_name):
        print(f"Sending batch of size {len(batch)} to model {model_name}")
        endpoint = list(self.ipfs_embeddings_py.https_endpoints[model_name].keys())[0]
        model_context_length = self.ipfs_embeddings_py.https_endpoints[model_name][endpoint]
        new_batch = []
        for item in batch:
            if model_name not in self.tokenizer.keys():
                self.tokenizer[model_name] = AutoTokenizer.from_pretrained(model_name)
            this_item_tokens = len(self.tokenizer[model_name].encode(item[column]))
            if this_item_tokens > model_context_length:
                encoded_item = self.tokenizer[model_name](item[column], return_tensors="pt")["input_ids"].tolist()[0]
                truncated_encoded_item = encoded_item[:model_context_length]
                unencode_item = self.tokenizer[model_name].decode(truncated_encoded_item)
                new_batch.append(unencode_item)
            else:
                new_batch.append(item[column])
        results = self.ipfs_embeddings_py.index_knn(new_batch, model_name)
        return results
    
    async def save_to_disk(self, dataset, dst_path, models):
        self.saved = False
        while True:
            await asyncio.sleep(30)
            empty = True
            for queue in self.ipfs_embeddings_py.queues.values():
                if not queue.empty():
                    empty = False

            if empty == True and self.saved == False:   
                self.new_dataset.save_to_disk(f"{dst_path}/{dataset.replace("/","---")}.arrow")
                self.new_dataset.to_parquet(f"{dst_path}/{dataset.replace("/","---")}.parquet")
                for model in models:
                    self.index[model].save_to_disk(f"{dst_path}/{model.replace("/","---")}.arrow")
                    self.index[model].to_parquet(f"{dst_path}/{model.replace("/","---")}.parquet")
                self.saved = True
        return None
                
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
            self.new_dataset = datasets.Dataset.from_dict({key: [] for key in columns })

        for model in models:
            batch_size = 32
            if os.path.isfile(f"{dst_path}/{model.replace("/","---")}.arrow") == True:
                self.index[model] = self.datasets.load_dataset(f"{dst_path}/{model.replace("/","---")}.arrow")
            else:
                self.index[model] = datasets.Dataset.from_dict({"cid": [], "embedding": []})
                self.ipfs_embeddings_py.queues[model] = asyncio.Queue(batch_size)
                consumer_tasks[model] = asyncio.create_task(self.consumer(self.ipfs_embeddings_py.queues[model], column, batch_size, model))

        producer_task = asyncio.create_task(self.producer(self.dataset, column, self.ipfs_embeddings_py.queues))        
        # save_task = asyncio.create_task(self.save_to_disk(dataset, dst_path, models))
        save_task = asyncio.create_task(self.save_to_disk(dataset, dst_path, models))
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
    create_embeddings_batch = create_embeddings_batch(resources, metadata)
    asyncio.run(create_embeddings_batch.main(metadata["dataset"], metadata["column"], metadata["dst_path"], metadata["models"]))    


