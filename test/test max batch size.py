from ipfs_embeddings_py import ipfs_embeddings_py

class test_embeddings:
    def __init__(self, resources, metadata):
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://
        return None
    
    def test(self, request, model):
    
        return self.ipfs_embeddings_py.index_knn(request)
    
    def __test__(self):
