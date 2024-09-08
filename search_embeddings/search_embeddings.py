import os
import sys
import datasets
import json
import ipfs_embeddings_py
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import sys

class search_embeddings:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata
        self.datasets = datasets
        self.ipfs_embeddings_py = ipfs_embeddings_py.ipfs_embeddings_py(resources, metadata)
        self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed",1)
        self.join_column = None

    def init(self, dataset, knn_index):
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
        
        columns = self.dataset.column_names["data"]
        columns_to_keep = [common_columns, "Abstract"]
        columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
        self.dataset = self.dataset.remove_columns(columns_to_remove)
        self.dataset = datasets.concatenate_datasets([
            self.dataset["data1"],
            self.dataset["data2"],
            self.dataset["data3"],
            self.dataset["data4"],
            self.dataset["data5"],
            self.dataset["data6"],
            self.dataset["data7"],
            self.dataset["data8"],
            self.dataset["data9"],
            self.dataset["data10"],
            self.dataset["data11"],
            self.dataset["data12"],
            self.dataset["data13"],
            self.dataset["data14"],
            self.dataset["data15"],
            self.dataset["data16"],
            self.dataset["data17"],
            self.dataset["data18"],
            self.dataset["data19"],
            self.dataset["data20"]])
        
        temp_dataset2 = self.knn_index['enwiki_embed'].to_pandas()
        temp_dataset1 = self.dataset.to_pandas()
        self.joined_dataset = temp_dataset1.join(temp_dataset2.set_index(common_columns), on=common_columns)
        return found

    def start_qdrant(self):
        start_qdrant_cmd = "docker run -d --name qdrant -p 6333:6333 -v /tmp/qdrant:/qdrant/data qdrant/qdrant:latest"
        os.system(start_qdrant_cmd)
        return None
    
    def stop_qdrant(self):
        kill_qdrant_cmd = "docker stop qdrant"
        os.system(kill_qdrant_cmd)
        return None
    
    def load_qdrant(self):
        # Load the Parquet file

        # Initialize Qdrant client
        client = QdrantClient(host="localhost", port=6333)# Replace with your Qdrant server URL

        # Define the collection name
        collection_name = self.dataset_name
        embedding_size = len(self.knn_index[list(self.knn_index.keys())[0]].select([0])['Embeddings'][0][0])
        # Define the collection schema (adjust this to match your data's structure)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embedding_size, distance=models.Distance.COSINE),
        )

        # Chunk size for generating points
        chunk_size = 1
        knn_index_length = len(list(self.knn_index))
        # Prepare the points to be inserted in chunks

        for start in range(0, knn_index_length, chunk_size):
            end = min(start + chunk_size, knn_index_length)
            chunk_df = self.joined_dataset[start:end]
            points = []
            for index, row in chunk_df.iterrows():
                text = row["Abstract"]
                embedding = row["Embeddings"]
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

    def generate_embeddings(self, query, model):
        if isinstance(query, str):
            query = [query]
        elif not isinstance(query, list):
            raise ValueError("Query must be a string or a list of strings")
        
        self.ipfs_embeddings_py.queue_index_knn(query)
        selected_endpoint = self.ipfs_embeddings_py.choose_endpoint(model)
        embeddings = self.ipfs_embeddings_py.https_index_knn(selected_endpoint, model)
        return embeddings
    
    def search_embeddings(self, embeddings):
        scores, samples = self.knn_index.get_nearest_examples(
           "embeddings", embeddings, k=5
        )
        return None
    
    def search_qdrant(self, embeddings):

        return None
    

if __name__ == '__main__':
    resources = {}
    metadata = {}
    dataset = "laion/Wikipedia-X"
    faiss_index = "laion/Wikipedia-M3"
    search_query = "hello world"
    search_embeddings = search_embeddings(resources, metadata)
    # search_embeddings.install_qdrant()
    search_embeddings.init(dataset, faiss_index)
    # search_embeddings.stop_qdrant()
    # search_embeddings.start_qdrant()
    search_embeddings.load_qdrant()
    embedding_results = search_embeddings.generate_embeddings(search_query, "BAAI/bge-m3")
    embeddings_search = search_embeddings.search_embeddings(embedding_results)
