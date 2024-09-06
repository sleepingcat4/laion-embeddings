from ipfs_embeddings_py.ipfs_embeddings import ipfs_embeddings_py
from datasets import load_dataset

class test_ipfs_embeddings:
    def __init__(self):
        resources = "resources"
        metadata = "metadata"
        self.dataset = {}
        self.ipfs_embeddings = ipfs_embeddings_py(resources, metadata)
        return None
    
    def test(self):
        load_these_datasets = ["laion/Wikipedia-X", "laion/Wikipedia-X-Full", "laion/Wikipedia-X-Concat", "laion/Wikipedia-X-M3"]
        self.dataset = load_dataset(load_these_datasets[0])
        print(len(self.dataset))
        self.ipfs_embeddings
        return None
