from .ipfs_multiformats import *
from .ipfs_only_hash import *
import requests
import subprocess
import os
import json
import re
class ipfs_embeddings_py:
    def __init__(self, resources, metedata):
        self.multiformats = ipfs_multiformats_py(resources, metedata)
        self.ipfs_only_hash = ipfs_only_hash_py(resources, metedata)
        self.https_endpoints = {}
        self.libp2p_endpoints = {}
        self.cid_queue = iter([])
        self.knn_queue = iter([])
        self.cid_index = {}
        self.knn_index = {}
        self.endpoint_status = {}
        self.add_https_endpoint = self.add_https_endpoint
        self.rm_https_endpoint = self.rm_https_endpoint
        self.queue_index_cid = self.queue_index_cid
        self.queue_index_knn = self.queue_index_knn
        self.choose_endpoint = self.choose_endpoint
        self.pop_index_knn = self.pop_index_knn
        self.pop_index_cid = self.pop_index_cid
        self.max_batch_size = self.max_batch_size
        return None
    
    def load_index(self, index):
        self.index = index
        return None 
    
    def add_https_endpoint(self, model, endpoint, context_length):
        if model not in self.https_endpoints:
            self.https_endpoints[model] = {}
            self.endpoint_status[endpoint] = 1
        if endpoint not in self.https_endpoints[model]:  
            self.https_endpoints[model][endpoint] = context_length
            self.endpoint_status[endpoint] = 1
        return None
    
    def add_libp2p_endpoint(self, model, endpoint, context_length):
        if model not in self.libp2p_endpoints:
            self.libp2p_endpoints[model] = {}
            self.endpoint_status[endpoint] = 1
        if endpoint not in self.libp2p_endpoints[model]:  
            self.libp2p_endpoints[model][endpoint] = context_length
            self.endpoint_status[endpoint] = 1
        return None
    
    def rm_https_endpoint(self, model, endpoint):
        if model in self.https_endpoints and endpoint in self.https_endpoints[model]:
            del self.https_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None
    
    def rm_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            del self.libp2p_endpoints[model][endpoint]
            del self.endpoint_status[endpoint]
        return None
    
    def test_tei_https_endpoint(self, model, endpoint):
        if model in self.https_endpoints and endpoint in self.https_endpoints[model]:
            return True
        return False

    def test_libp2p_endpoint(self, model, endpoint):
        if model in self.libp2p_endpoints and endpoint in self.libp2p_endpoints[model]:
            return True
        return False

    def get_https_endpoint(self, model):
        if model in self.https_endpoints:
            return self.https_endpoints[model]
        return None
    
    def get_libp2p_endpoint(self, model):
        if model in self.libp2p_endpoints:
            return self.libp2p_endpoints[model]
        return None

    def request_https_endpoint(self, model, batch_size):
        if model in self.https_endpoints:
            for endpoint in self.https_endpoints[model]:
                if self.endpoint_status[endpoint] == 1:
                    return endpoint
        return None

    def index_cid(self, samples):
        results = []
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is iter:
            for this_sample in samples:
                this_sample_cid = self.multiformats.get_cid(this_sample)
                self.cid_index[this_sample_cid] = this_sample
            pass
        if type(samples) is list:
            for this_sample in samples:
                this_sample_cid = self.multiformats.get_cid(this_sample)
                self.cid_index[this_sample_cid] = this_sample
                results.append(this_sample_cid)
        return results
    
    def max_batch_size(self, model, endpoint=None):
        print("max_batch_size")
        embed_fail = False
        exponent = 1
        batch = []
        batch_size = 2**exponent
        while embed_fail == False:
            while len(batch) < batch_size:
                batch.append("Hello World")
            try:
                embeddings = self.index_knn(batch, model, endpoint)
                if not isinstance(embeddings[0], list):
                    raise Exception("Embeddings not returned as list")
                embed_fail = False
                exponent += 1
                batch_size = 2**exponent
            except:
                embed_fail = True
                pass
        return 2**(exponent-1)
    
    def index_knn(self, samples, model, chosen_endpoint=None):
        knn_stack = []
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if type(samples) is str:
            samples = [samples]
        if type(samples) is iter:
            if chosen_endpoint is None:
                chosen_endpoint = self.choose_endpoint(model)
            this_query = {"inputs": samples}
            query_response = self.make_post_request(chosen_endpoint, this_query)
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        if type(samples) is list:
            if chosen_endpoint is None:
                chosen_endpoint = self.choose_endpoint(model)
            this_query = {"inputs": samples}
            query_response = self.make_post_request(chosen_endpoint, this_query)
            if isinstance(query_response, dict) and "error" in query_response.keys():
                raise Exception("error: " + query_response["error"])
            else:
                knn_stack = query_response
            pass
        return knn_stack
    
    def queue_index_cid(self, samples):
        if type(samples) is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
            self.knn_queue = iter(list(samples))
            return True
        if isinstance(samples, list):
            self.knn_queue = list(self.knn_queue)
            for this_sample in samples:
                self.knn_queue.append(this_sample)
            self.knn_queue = iter(self.knn_queue)
            return True
        else:
            raise ValueError("samples must be a list")
    
    def queue_index_knn(self, samples):
        print("queue_index_knn")
        print(samples)
        print(type(samples))
        if samples is None:
            raise ValueError("samples must be a list")
        if isinstance(samples, str):
            samples = [samples]
            self.knn_queue = iter(list(samples))
            return True
        if isinstance(samples, list):
            self.knn_queue = list(self.knn_queue)
            for this_sample in samples:
                self.knn_queue.append(this_sample)
            self.knn_queue = iter(self.knn_queue)
            return True
        else:
            raise ValueError("samples must be a list")


    def make_post_request(self, endpoint, data):
        headers = {'Content-Type': 'application/json'}
        # sanitize data
        for input in range(len(data["inputs"])):
            this_input = data["inputs"][input]
            if isinstance(this_input, str):
                # remove anything that would cause errors in json parsing
                # this_input = re.sub(r'[^\x00-\x7F]+', ' ', this_input)  # remove non-ASCII characters
                this_input = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', this_input)  # remove control characters
                this_input = re.sub(r'[\\"]', ' ', this_input)  # remove backslashes and double quotes
                ## convert double quotes to escape characters
                this_input = re.sub(r'["]', '\"', this_input)
                this_input = re.sub(r'[\b\f\n\r\t]', ' ', this_input)  # remove escape characters
                # this_input = re.sub(r'[^a-zA-Z0-9\s]', ' ', this_input)  # remove all special characters
                this_input = re.sub(r'\s+', ' ', this_input)  # remove extra spaces
                data["inputs"][input] = this_input
        print(data)
        response = requests.post(endpoint, headers=headers, json=data)
        return response.json()

    def choose_endpoint(self, model):
        print("choose_endpoint")
        print(model)
        https_endpoints = self.get_https_endpoint(model)
        libp2p_endpoints = self.get_libp2p_endpoint(model)
        filtered_libp2p_endpoints = {k: v for k, v in self.endpoint_status.items() if v == 1 and libp2p_endpoints is not None and k in list(libp2p_endpoints.keys())}
        filtered_https_endpoints = {k: v for k, v in self.endpoint_status.items() if v == 1 and https_endpoints is not None and k in list(https_endpoints.keys())}
        if ( not filtered_https_endpoints and not filtered_libp2p_endpoints):
            return None
        else:
            this_endpoint = None
            if len(list(filtered_https_endpoints.keys())) > 0:
                this_endpoint = list(filtered_https_endpoints.keys())[0]
            elif len(list(filtered_libp2p_endpoints.keys())) > 0:
                this_endpoint = list(filtered_libp2p_endpoints.keys())[0]
            return this_endpoint
        
    def https_index_cid(self, samples, endpoint):
        endpoint_chunk_size = self.https_endpoints[endpoint]
        all_chunk = []
        this_chunk = []
        for i in range(samples):
            self
            ## request endpoint
            pass
        return None
    
    def https_index_knn(self, selected_endpoint, model):
        batch_size = 0
        if "http" in selected_endpoint:
            batch_size = self.https_endpoints[model][selected_endpoint]
        elif "libp2p" in selected_endpoint:
            batch_size = self.libp2p_endpoints[model][selected_endpoint]
        knn_stack = []
        queue_knn = self.pop_index_knn(batch_size)
        json_queue_knn = json.dumps(queue_knn)
        for i in queue_knn:
            this_sample = {"inputs": i}
            query_response = self.make_post_request(selected_endpoint, this_sample)
            knn_stack.append(query_response)
        return knn_stack
    
    def select_endpoint(self, model):
        if model in self.https_endpoints:
            for endpoint in self.https_endpoints[model]:
                if self.endpoint_status[endpoint] == 1:
                    self.endpoint_status[endpoint] = 0
                    return endpoint
        return None

    def pop_index_cid(self, number):
        results = []
        if number > len(self.cid_queue):
            raise ValueError("number is greater than the queue size")
        if number <= 0:
            raise ValueError("number must be greater than 0")
        for i in range(number):
            results.push(self.cid_queue.pop())
            i += 1
        return results
    
    def pop_index_knn(self, number):
        results = []
        knn_queue_list = list(self.knn_queue)
        if number > len(knn_queue_list):
            raise ValueError("number is greater than the queue size")
        if number <= 0:
            raise ValueError("number must be greater than 0")
        for i in range(number):
            results.append(knn_queue_list.pop())
            i += 1
        self.knn_queue = iter(knn_queue_list)
        return results


    def test(self):
        self.https_endpoints("BAAI/bge-m3", "62.146.169.111:80/embed",1)
        self.https_endpoints("BAAI/bge-m3", "62.146.169.111:8080/embed",1)
        self.https_endpoints("BAAI/bge-m3", "62.146.168.111:8081/embed",1)
        test_knn_index = {}
        test_cid_index = {}
        test_data = {
            "test1", "test2", "test3"
        }

        for data in test_data:
            test_cid_index = self.index_cid(data)
            test_knn_index = self.index_knn(data)

        print("test")

    def status(self):
        return self.endpoint_status
    
    def setStatus(self,endpoint , status):
        self.endpoint_status[endpoint] = status
        return None