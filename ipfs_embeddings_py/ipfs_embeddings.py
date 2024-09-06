
class ipfs_embeddings_py:
    def __init__(self, ipfs_path):
        self.ipfs_path = ipfs_path
        self.tei_https_endpoints = {}
        self.libp2p_endpoints = {}
        self.queue = iter([])
        self.index = 0
        self.index = {}
        self.ipfsIndex = {}
        return None
    
    def load_index(self, index):
        self.index = index
        return None 
    
    def add_tei_https_endpoint(self, model, endpoint, batch_size):
        if model not in self.tei_https_endpoints:
            self.tei_https_endpoints[model] = []
        if endpoint not in self.tei_https_endpoints[model]:  
            self.tei_https_endpoints[model].push({endpoint: batch_size})
        return None
    
    def add_libp2p_endpoint(self, model, endpoint, batch_size):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = []
        if endpoint not in self.libp2p_endpoints[model]:  
            self.libp2p_endpoints[model].push({endpoint: batch_size})
        return None
    
    def rm_tei_https_endpoint(self, model, endpoint):
        if model in self.tei_https_endpoints and endpoint in self.tei_https_endpoints[model]:
            self.tei_https_endpoints[model].remove(endpoint in self.libp2p_endpoints[model].keys().filter(endpoint))
        return None
    
    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            self.libp2p_endpoints[model].remove(endpoint in self.libp2p_endpoints[model].keys().filter(endpoint))
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