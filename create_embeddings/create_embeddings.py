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

class create_embeddings:
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

    def index_dataset (self, dataset, faiss_dst, model):
        #check if the dataset exists in the faiss index
        dataset_exists = False
        try:
            self.faiss_index = self.datasets.load_dataset(faiss_dst)
        except:
            self.faiss_index = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        self.dataset = self.datasets.load_dataset(dataset)['train']
        self.dataset_name = dataset
        self.faiss_index_name = faiss_dst
        dataset_columns = self.dataset.column_names
        dataset_columns.append("cid")
        ## create new column in the dataset called "cid"
        self.dataset.add_column("cid", [str(i) for i in range(len(self.dataset))])
        self.new_dataset = datasets.Dataset.from_dict({key: [] for key in dataset_columns })
        new_rows = []
        for row in self.dataset:
            this_row = row
            cid = self.ipfs_embeddings_py.index_cid(this_row["text"])
            this_row["cid"] = cid
            find_cid = self.faiss_index.filter(lambda x: x["cid"] == cid)
            if find_cid.num_rows > 0:
                pass
            else:
                embedding = self.ipfs_embeddings_py.index_knn(this_row["text"], model)[0]
                new_row = {}
                new_row["cid"] = cid
                new_row["embedding"] = embedding
                new_rows.append(new_row)
        new_dataset = datasets.Dataset.from_dict({key: [row[key] for row in new_rows] for key in new_rows[0].keys()})
        self.faiss_index = new_dataset
        self.dataset.save_to_disk(f"/storage/teraflopai/{self.dataset_name}.arrow")
        self.faiss_index.save_to_disk(f"/storage/teraflopai/{self.faiss_index_name}.arrow")
        self.dataset.to_parquet(f"/storage/teraflopai/{self.dataset_name}.parquet")
        self.faiss_index.to_parquet(f"/storage/teraflopai/{self.faiss_index_name}.parquet")
        self.dataset.push_to_hub(self.dataset_name, use_temp_dir=True, message="Update dataset")
        self.faiss_index.push_to_hub(self.faiss_index_name, use_temp_dir=True, message="Update faiss index")
        return None
    
    async def batch_index_dataset (self, dataset, faiss_dst, model):
        dataset_exists = False
        try:
            self.faiss_index = self.datasets.load_dataset(faiss_dst)
        except:
            self.faiss_index = datasets.Dataset.from_dict({"cid": [], "embedding": []})
        self.dataset = self.datasets.load_dataset(dataset)['train']
        self.dataset_name = dataset
        self.faiss_index_name = faiss_dst
        dataset_columns = self.dataset.column_names
        dataset_columns.append("cid")
        ## create new column in the dataset called "cid"
        self.dataset.add_column("cid", [str(i) for i in range(len(self.dataset))])
        self.new_dataset = datasets.Dataset.from_dict({key: [] for key in dataset_columns })
        new_rows = []
        for this_row in self.dataset:
            self.ipfs_embeddings_py.queue_index_cid(this_row["text"])
            self.ipfs_embeddings_py.queue_index_knn(this_row["text"], model)
        return None
    
if __name__ == '__main__':
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "faiss_index": "endomorphosis/Caselaw_Access_Project_M3_Embeddings",
        "model": "BAAI/bge-m3"
    }
    resources = {
        "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",1]]
    }
    create_embeddings = create_embeddings(resources, metadata)
    # results = create_embeddings.index_dataset(metadata["dataset"], metadata["faiss_index"], metadata["model"])
    results = create_embeddings.index_dataset(metadata["dataset"], metadata["faiss_index"], metadata["model"])

    print(results)
