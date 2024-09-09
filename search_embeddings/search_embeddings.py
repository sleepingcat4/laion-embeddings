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

class search_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        if len(list(metadata.keys())) > 0:
            for key in metadata.keys():
                setattr(self, key, metadata[key])
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
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
        start_qdrant_cmd = "sudo docker run -d -p 6333:6333 -v /storage/qdrant:/qdrant/data qdrant/qdrant:latest"
        os.system(start_qdrant_cmd)
        return None
    
    def stop_qdrant(self):
        kill_qdrant_cmd = "sudo docker stop $(sudo docker ps -a -q --filter ancestor=qdrant/qdrant:latest --format={{.ID}})"
        os.system(kill_qdrant_cmd)
        return None
    
    def load_qdrant(self, dataset, knn_index):
        self.knn_index = self.datasets.load_dataset(knn_index)
        self.dataset = self.datasets.load_dataset(dataset)
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

    def generate_embeddings(self, query):
        if isinstance(query, str):
            query = [query]
        elif not isinstance(query, list):
            raise ValueError("Query must be a string or a list of strings")
        
        self.ipfs_embeddings_py.queue_index_knn(query)
        selected_endpoint = self.ipfs_embeddings_py.choose_endpoint(self.model)
        embeddings = self.ipfs_embeddings_py.https_index_knn(selected_endpoint, self.model)
        return embeddings
    
    def search_embeddings(self, embeddings):
        scores, samples = self.knn_index.get_nearest_examples(
           "embeddings", embeddings, k=5
        )
        return scores, samples 
    
    def search_qdrant(self, query_vector, dataset_name):
        query_vector = np.array(query_vector[0][0])
        client = QdrantClient(url="http://localhost:6333")
        search_result = client.search(
            collection_name=dataset_name,
            query_vector=query_vector,
            limit=5  # Return 5 closest points
        )
        results = []
        for point in search_result:
            results.append({point.id: {
                    "text": point.payload["text"],
                    "score": point.score
                }})       
        return results
    
    def search(self, query):
        if self.qdrant_found == True:
            query_embeddings = self.generate_embeddings(query)
            vector_search = search_embeddings.search_qdrant(query_embeddings, self.dataset.split("/")[1])
        else:
            print("Qdrant failed to start")
            ## Fallback to faiss
            return None
        return vector_search
    

if __name__ == '__main__':
    metadata = {
        "dataset": "laion/Wikipedia-X-Concat",
        "faiss_index": "laion/Wikipedia-M3",
        "model": "BAAI/bge-m3"
    }
    resources = {}
    search_embeddings = search_embeddings(resources, metadata)
    results = search_embeddings.search("Machine Learning")
    print(results)
