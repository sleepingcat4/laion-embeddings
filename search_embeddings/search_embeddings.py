import os
import sys
import datasets
import json
import ipfs_embeddings_py

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

        knn_columns = self.knn_index.column_names
        dataset_columns = self.dataset.column_names
        # Check if the dataset has the same columns as the knn_index
        if not all(elem in dataset_columns for elem in knn_columns):
            raise ValueError("The dataset must have the same columns as the knn_index")
        else:
            common_columns = list(set(knn_columns) & set(dataset_columns))
            if len(common_columns) == 0:
                raise ValueError("The dataset must have the same columns as the knn_index")
            else:
                self.join_column = common_columns[0]
        return None


    def start_qdrant(self):
        return None
    
    def stop_qdrant(self):
        return None
    
    def load_qdrant(self):
        return None
    
    def qdrant_search(self, query, model):
        if isinstance(query, str):
            query = [query]
        elif not isinstance(query, list):
            raise ValueError("Query must be a string or a list of strings")
        
        self.ipfs_embeddings_py.queue_index_knn(query)
        selected_endpoint = self.ipfs_embeddings_py.choose_endpoint(model)
        embeddings = self.ipfs_embeddings_py.https_index_knn(selected_endpoint, model)
        return embeddings

    

if __name__ == '__main__':
    resources = {}
    metadata = {}
    dataset = "laion/Wikipedia-X"
    faiss_index = "laion/Wikipedia-M3"
    search_query = "hello world"
    search_embeddings = search_embeddings(resources, metadata)
    search_embeddings.init(dataset, faiss_index)
    search_embeddings.stop_qdrant()
    search_embeddings.start_qdrant()
    search_embeddings.load_qdrant()
    search_embeddings.qdrant_search(search_query, "BAAI/bge-m3")
