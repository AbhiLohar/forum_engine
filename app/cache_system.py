import numpy as np
from collections import OrderedDict

class SemanticCache:
    def __init__(self, size=50, threshold=0.65): 
        self.cache = OrderedDict()
        self.size = size
        self.threshold = threshold

    def get(self, query_vec):
        
        query_vec = np.array(query_vec).flatten()
        
        for vec_tuple, result in reversed(self.cache.items()):
            vec = np.array(vec_tuple)
            
            
            norm_q = np.linalg.norm(query_vec)
            norm_v = np.linalg.norm(vec)
            
            if norm_q == 0 or norm_v == 0: continue
                
            sim = np.dot(query_vec, vec) / (norm_q * norm_v)
            
            if sim >= self.threshold:
                self.cache.move_to_end(vec_tuple)
                return result, sim
        return None, 0

    def set(self, vec, result):
        
        vec_tuple = tuple(np.array(vec).flatten())
        self.cache[vec_tuple] = result
        
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)