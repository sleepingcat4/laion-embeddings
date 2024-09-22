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
import asyncio
import hashlib

class search_embeddings:
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
        self.qdrant_found = False
        qdrant_port_cmd = "nc -zv localhost 6333"
        qdrant_port_cmd_results = os.system(qdrant_port_cmd)
        if qdrant_port_cmd_results != 0:
            self.start_qdrant()
            qdrant_port_cmd_results = os.system(qdrant_port_cmd)
            if qdrant_port_cmd_results == 0:
                self.qdrant_found = True
            else:
                print("Qdrant failed to start, fallback to faiss")
        else:
            self.qdrant_found = True

    def start_qdrant(self):
        docker_pull_cmd = "sudo docker pull qdrant/qdrant:latest"
        os.system(docker_pull_cmd)
        docker_ps = "sudo docker ps | grep qdrant/qdrant:latest"
        try:
            docker_ps_results = subprocess.check_output(docker_ps, shell=True).decode("utf-8")
        except subprocess.CalledProcessError as e:
            docker_ps_results = e
            docker_stopped_ps = "sudo docker ps -a | grep qdrant/qdrant:latest"
            try:
                docker_stopped_ps_results  = subprocess.check_output(docker_stopped_ps, shell=True).decode("utf-8")
                start_qdrant_cmd = "sudo docker start $(sudo docker ps -a -q --filter ancestor=qdrant/qdrant:latest --format={{.ID}})"
                os.system(start_qdrant_cmd)
            except subprocess.CalledProcessError as e:
                docker_stopped_ps_results = e
                start_qdrant_cmd = "sudo docker run -d -p 6333:6333 -v /storage/qdrant:/qdrant/data qdrant/qdrant:latest"
                os.system(start_qdrant_cmd)
        return 1
    
    def stop_qdrant(self):
        kill_qdrant_cmd = "sudo docker stop $(sudo docker ps -a -q --filter ancestor=qdrant/qdrant:latest --format={{.ID}})"
        os.system(kill_qdrant_cmd)
        return None
    
    async def join_datasets(self, dataset, knn_index, join_column):
        dataset_iter = iter(dataset)
        knn_index_iter = iter(knn_index)
        while True:
            try:
                dataset_item = next(dataset_iter)
                knn_index_item = next(knn_index_iter)
                results = {}
                for key in dataset_item.keys():
                    results[key] = dataset_item[key]
                same = True
                for column in join_column:
                    if dataset_item[column] != knn_index_item[column]:
                        same = False
                        break
                if same == True:
                    for key in knn_index_item.keys():
                        results[key] = knn_index_item[key]
                else:
                    if "knn_index_hash" in self.keys() and "datasets_hash" in self.keys() and len(self.knn_index_hash) > 0 and len(self.datasets_hash) > 0:
                        this_hash_key = {}
                        for columin in join_column:
                            this_hash_key[column] = dataset_item[column]
                        this_hash_value = hashlib.md5(json.dumps(this_hash_key).encode()).hexdigest()
                        if this_hash_value in self.knn_index_hash:
                            knn_index_item = self.knn_index_hash.index(this_hash_value)
                            for key in knn_index_item[knn_index_item].keys():
                                results[key] = knn_index_item[key]
                    else:
                        async for item in self.dataset:
                            this_hash_key = {}
                            for column in join_column:
                                this_hash_key[column] = item[column]
                            new_hash_value = hashlib.md5(json.dumps(this_hash_key).encode()).hexdigest()
                            self.datasets_hash.append(new_hash_value)
                        async for item in self.knn_index:
                            this_hash_key = {}
                            for column in join_column:
                                this_hash_key[column] = item[column]
                            new_hash_value = hashlib.md5(json.dumps(this_hash_key).encode()).hexdigest()
                            self.knn_index_hash.append(new_hash_value)
                        if this_hash_value in self.knn_index_hash and this_hash_value in self.datasets_hash:
                            knn_index_item = self.knn_index_hash.index(this_hash_value)
                            for key in knn_index_item[knn_index_item].keys():
                                results[key] = knn_index_item[key]
                yield results
            except StopAsyncIteration:
                break
            yield results
    
    async def load_qdrant_iter(self, dataset, knn_index, dataset_split= None, knn_index_split=None):
        self.knn_index_hash = []
        self.datasets_hash = []
        if dataset_split is not None:
            self.dataset = self.datasets.load_dataset(dataset, split=dataset_split, streaming=True)
        else:
            self.dataset = self.datasets.load_dataset(dataset, streaming=True)
        if knn_index_split is not None:
            self.knn_index = self.datasets.load_dataset(knn_index, split=knn_index_split, streaming=True)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            single_row = next(iter(self.knn_index.take(1)))
            self.embedding_size = len(single_row["embeddings"][0])
            self.knn_index = self.datasets.load_dataset(knn_index, split=knn_index_split, streaming=True)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            # self.knn_index_length = sum(1 for _ in self.knn_index)
        else:
            self.knn_index = self.datasets.load_dataset(knn_index, streaming=True)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            single_row = next(iter(self.knn_index.take(1)))
            self.embedding_size = len(single_row["embeddings"][0])
            self.knn_index = self.datasets.load_dataset(knn_index, streaming=True)
            if "Embeddings" in self.knn_index.column_names:
                self.knn_index = self.knn_index.rename_column("Embeddings", "embeddings")
            # self.knn_index_length = sum(1 for _ in self.knn_index)

        self.dataset_name = dataset
        self.knn_index_name = knn_index
        knn_columns = self.knn_index.column_names
        dataset_columns = self.dataset.column_names
        common_columns = set(dataset_columns).intersection(set(knn_columns))
        self.join_column = common_columns
        self.joined_dataset = self.join_datasets(self.dataset, self.knn_index, self.join_column)
        return None

    async def ingest_qdrant_iter(self, column_name):
        embedding_size = 0
        self.knn_index_length = 99999
        collection_name = self.dataset_name.split("/")[1]
        client = QdrantClient(url="http://localhost:6333")
        # Define the collection name
        collection_name = self.dataset_name.split("/")[1]
        if (client.collection_exists(collection_name)):
            print(collection_name + " Collection already exists")
        else:
            print("Creating collection" + collection_name)        
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.embedding_size, distance=Distance.COSINE),
            )

        # Chunk size for generating points
        chunk_size = 100
        # Prepare the points to be inserted in chunks
        processed_rows = 0
        points = []
        async for item in self.joined_dataset:
            processed_rows += 1
            points.append(models.PointStruct(
                id=processed_rows,
                vector=item["embeddings"][0],
                payload={"text": item[column_name]}
            ))
            if len(points) == chunk_size:
                print(f"Processing chunk {processed_rows-chunk_size} to {processed_rows}")
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                points = []        
        
        print("Data successfully ingested into Qdrant")
        print("All data successfully ingested into Qdrant from huggingface dataset")
        return True
        

    async def load_qdrant(self, dataset, knn_index):
        self.knn_index = self.datasets.load_dataset(knn_index, streaming=True)
        self.dataset = self.datasets.load_dataset(dataset, streaming=True)
        self.dataset_name = dataset
        self.knn_index_name = knn_index
        knn_columns = self.knn_index.column_names[list(self.knn_index.column_names.keys())[0]]
        dataset_columns = self.dataset.column_names[list(self.dataset.column_names.keys())[0]]
        # Check if the dataset has the same columns as the knn_index
        found = False
        common_columns = None
        for column in dataset_columns:
            if column in knn_columns:
                found = True
                common_columns = column
                self.join_column = common_columns
                break
        
        columns = self.dataset.column_names["enwiki_concat"]
        columns_to_keep = [common_columns, "Concat Abstract"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        self.dataset = self.dataset.remove_columns(columns_to_remove)
        temp_dataset2 = self.knn_index['enwiki_embed'].to_pandas()
        temp_dataset1 = self.dataset['enwiki_concat'].to_pandas()
        self.joined_dataset = temp_dataset1.join(temp_dataset2.set_index(common_columns), on=common_columns)
        client = QdrantClient(url="http://localhost:6333")
        # Define the collection name
        collection_name = self.dataset_name.split("/")[1]
        embedding_size = len(self.knn_index[list(self.knn_index.keys())[0]].select([0])['Embeddings'][0][0])

        if (client.collection_exists(collection_name)):
            print("Collection already exists")
            return False
        else:
            print("Creating collection")        
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )

        # Chunk size for generating points
        chunk_size = 100
        knn_index_length = self.joined_dataset.shape[0]# Get the number of rows in the dataset
        # Prepare the points to be inserted in chunks

        for start in range(0, knn_index_length, chunk_size):
            end = min(start + chunk_size, knn_index_length)
            chunk_df = self.joined_dataset.iloc[start:end]
            points = []
            print(f"Processing chunk {start}:{end}")
            for index, row in chunk_df.iterrows():
                text = row["Concat Abstract"]
                embedding = row["Embeddings"][0]
                points.append(models.PointStruct(
                    id=index,
                    vector=embedding.tolist() if embedding is not None else None,  # Convert embedding to list if not None
                    payload={"text": text}
                ))

            client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        print("Data successfully ingested into Qdrant")
        print("All data successfully ingested into Qdrant from huggingface dataset")
        return True

    def rm_cache(self):
        homedir = os.path.expanduser("~")
        cache_dir = homedir + "/.cache/huggingface/datasets/"
        cache_dir = os.path.expanduser(cache_dir)
        os.system("rm -rf " + cache_dir)
        return None

    async def generate_embeddings(self, query, model=None):
        if model is not None:
            model = self.metadata["model"]
        if isinstance(query, str):
            query = [query]
        elif not isinstance(query, list):
            raise ValueError("Query must be a string or a list of strings")
        self.ipfs_embeddings_py.index_knn(query, "")
        selected_endpoint = self.ipfs_embeddings_py.choose_endpoint(self.model)
        embeddings = await self.ipfs_embeddings_py.index_knn(selected_endpoint, self.model)
        return embeddings
    
    def search_embeddings(self, embeddings):
        scores, samples = self.knn_index.get_nearest_examples(
           "embeddings", embeddings, k=5
        )
        return scores, samples 
    
    def search_qdrant(self, query_vector, dataset_name, n=5):
        query_vector = np.array(query_vector[0][0])
        client = QdrantClient(url="http://localhost:6333")
        search_result = client.search(
            collection_name=dataset_name,
            query_vector=query_vector,
            limit=n
        )
        results = []
        for point in search_result:
            results.append({point.id: {
                    "text": point.payload["text"],
                    "score": point.score
                }})       
        return results
    
    async def search(self, query, n=5):
        if self.qdrant_found == True:
            query_embeddings = await self.generate_embeddings(query)
            vector_search = self.search_qdrant(query_embeddings, self.dataset.split("/")[1], n)
        else:
            print("Qdrant failed to start")
            ## Fallback to faiss
            return None
        return vector_search

    async def test_low_memory(self):
        start = self.start_qdrant()
        load_qdrant = await self.load_qdrant_iter("laion/Wikipedia-X-Concat", "laion/Wikipedia-M3", "enwiki_concat", "enwiki_embed")
        ingest_qdrant = await self.ingest_qdrant_iter("Concat Abstract")
        results = await search_embeddings.search("Machine Learning")
        return None
    
    async def test_high_memory(self):
        start = self.start_qdrant()
        load_qdrant = await self.load_qdrant("laion/Wikipedia-X-Concat", "laion/Wikipedia-M3")
        ingest_qdrant = await self.ingest_qdrant("Concat Abstract")
        results = await search_embeddings.search("Machine Learning")
        return None
    
if __name__ == '__main__':
    metadata = {
        "dataset": "laion/Wikipedia-X-Concat",
        "faiss_index": "laion/Wikipedia-M3",
        "model": "BAAI/bge-m3"
    }
    resources = {
        "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",8192]]
    }
    search_embeddings = search_embeddings(resources, metadata)
    asyncio.run(search_embeddings.test_low_memory())
    print()