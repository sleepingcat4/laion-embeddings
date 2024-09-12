from ipfs_embeddings_py import ipfs_embeddings_py

class test_embeddings:
    def __init__(self, resources, metadata):
        self.ipfs_embeddings_py = ipfs_embeddings_py(resources, metadata)
        self.ipfs_embeddings_py.add_https_endpoint("BAAI/bge-m3", "http://62.146.169.111:80/embed",1)
        return None
    
    def test(self, model):
        embed_fail = False
        exponent = 1
        batch = []
        batch_size = 2**exponent
        while embed_fail == False:
            while len(batch) < batch_size:
                batch.append("Hello World")
            try:
                embeddings = self.ipfs_embeddings_py.index_knn(batch, model)
                if not isinstance(embeddings[0], list):
                    raise Exception("Embeddings not returned as list")
                embed_fail = False
                exponent += 1
                batch_size = 2**exponent
            except:
                embed_fail = True
                pass
        return 2**(exponent-1)
    
if __name__ == '__main__':
    metadata = {
        "dataset": "TeraflopAI/Caselaw_Access_Project",
        "faiss_index": "endomorphosis/Caselaw_Access_Project_M3_Embeddings",
        "model": "BAAI/bge-m3"
    }
    resources = {
        "https_endpoints": [["BAAI/bge-m3", "http://62.146.169.111:80/embed",1]]
    }
    test = test_embeddings(resources, metadata)
    print(test.test(metadata["model"]))