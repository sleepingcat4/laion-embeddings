
class ipfs_embeddings_py:
    def __init__(self, resources, metedata):
        self.tei_https_endpoints = {}
        self.libp2p_endpoints = {}
        self.queue = iter([])
        self.index = 0
        self.index = {}
        self.ipfsIndex = {}
        self.endpointStatus = {}
        return None
    
    def load_index(self, index):
        self.index = index
        return None 
    
    def add_tei_https_endpoint(self, model, endpoint, batch_size):
        if model not in self.tei_https_endpoints:
            self.tei_https_endpoints[model] = {}
        if endpoint not in self.tei_https_endpoints[model]:  
            self.tei_https_endpoints[model][endpoint] = batch_size
        return None
    
    def add_libp2p_endpoint(self, model, endpoint, batch_size):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = {}
        if endpoint not in self.libp2p_endpoints[model]:  
            self.libp2p_endpoints[model][endpoint] = batch_size
        return None
    
    def rm_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_https_endpoints and endpoint in self.tei_https_endpoints[model]:
            del self.tei_https_endpoints[model][endpoint]
        return None
    
    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            del self.libp2p_endpoints[model][endpoint]
        return None
    
    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_https_endpoints and endpoint in self.tei_https_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False

    def test(self):
        self.add_tei_https_endpoint("BAAI/bge-m3", "62.146.169.111:80/embed",1)
        self.add_tei_https_endpoint("BAAI/bge-m3", "62.146.169.111:8080/embed",1)
        self.add_tei_https_endpoint("BAAI/bge-m3", "62.146.168.111:8081/embed",1)
        print("test")

    def status 

if __name__ == '__main__':
    ipfs_embeddings = ipfs_embeddings_py()
    ipfs_embeddings.test()
    print("test")