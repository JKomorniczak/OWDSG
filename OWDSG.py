import numpy as np
from sklearn.datasets import make_classification


class OWDSG:
    def __init__(self, n_drifts, n_novel, even_gt=True, 
                 hide_label=False, 
                 n_features=10, n_informative = 10, 
                 percentage_novel=0.1, 
                 n_classes=2, weights=None,
                 n_chunks=200, chunk_size=200, 
                 n_clusters_per_class=1, class_sep=1.,
                 random_state = None, allow_projection=True):
        
        self.n_drifts = n_drifts
        self.n_novel = n_novel
        self.even_gt = even_gt
        self.hide_label = hide_label
        self.percentage_novel = percentage_novel
        self.weights = np.round(np.ones(n_classes)/n_classes, 3) if weights is None else weights
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.allow_projection = allow_projection 

        self.n_classes = n_classes
        self.n_features = n_features
        self.n_informative = n_informative
        
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.random_state = random_state
        np.random.seed(self.random_state)
                
        if len(self.weights)!= self.n_classes:
            raise ValueError('Length of weights must correspond to the number od KC')

        # Define concept drift moments and novel class moments
        self.drift_gt = []
        self.novel_gt = []
        if n_drifts>0:
            interval_drift = self.n_chunks//self.n_drifts
            if self.even_gt:
                self.drift_gt = np.array([interval_drift//2 + i*interval_drift for i in range(self.n_drifts)])
            else:
                self.drift_gt = np.sort(np.random.choice(np.arange(interval_drift//4, self.n_chunks), self.n_drifts, replace=False))

        if n_novel>0:
            interval_novel = self.n_chunks//self.n_novel
            if self.even_gt:
                self.novel_gt = np.array([interval_novel//2 + i*interval_novel for i in range(self.n_novel)])
            else:
                self.novel_gt = np.sort(np.random.choice(np.arange(interval_novel//4, self.n_chunks), self.n_novel, replace=False))
                 
                                    
        # Use make classification generator
        _n_classes = self.n_classes*(self.n_drifts+1) + self.n_novel
        
        # Prepare for feature projections (if required and allowed)
        minimum_req_features = 2**self.n_informative
        expected_min_features = _n_classes*self.n_clusters_per_class
        
        if minimum_req_features<expected_min_features and self.allow_projection:
            print('Performing feature projections from %i to %i dimensions. The value for n_informative is ignored.' % (expected_min_features, self.n_features))
            n_informative_mc = expected_min_features
            n_features_mc = expected_min_features
        else:
            n_informative_mc = self.n_informative
            n_features_mc = self.n_features
        
        _X, _y = make_classification(n_samples=_n_classes*self.chunk_size*self.n_chunks,
                                    n_features=n_features_mc,
                                    n_repeated=0,
                                    n_redundant=0,
                                    n_clusters_per_class=self.n_clusters_per_class,
                                    class_sep=self.class_sep,
                                    n_informative=n_informative_mc,
                                    n_classes=_n_classes,
                                    flip_y=0.0,
                                    random_state=self.random_state)
        
        # Feature projections 
        if minimum_req_features<expected_min_features and self.allow_projection:
            projection_matrix = np.random.normal(size=(n_features_mc, self.n_features))
            _X = _X@projection_matrix
            _X /= n_features_mc

        # Arange into stream
        unused_mask = np.ones(_y.shape[0]).astype(bool)
        
        self.X = []
        self.y = []
        
        known_class_ids = np.arange(self.n_classes)
        unknown_class_ids = []

        for i in range(self.n_chunks):
            
            # Establish indexes of known and unknown classes for this chunk
            if i in self.drift_gt:
                known_class_ids = known_class_ids + self.n_classes
            if i in self.novel_gt:
                unknown_class_ids.append(self.n_classes*(self.n_drifts+1) + len(unknown_class_ids))
                            
            chunk_X = []
            chunk_y = []
            
            # Sample knowns
            for cl_k in known_class_ids:
                possible_mask = np.array([_y==cl_k]).flatten()
                possible_mask = possible_mask*unused_mask
                
                possible_ids = np.argwhere(possible_mask==1).flatten() 
                n_r_samples = int(self.weights[cl_k%self.n_classes]*self.chunk_size)  
                if cl_k ==  known_class_ids[-1]:
                    # last -- ensure correct chunk size
                    n_r_samples = self.chunk_size - len(chunk_X)
                chosen_ids = np.random.choice(possible_ids, 
                                              n_r_samples, 
                                              replace=False)
                
                chunk_X.extend(_X[chosen_ids])
                chunk_y.extend(_y[chosen_ids]%self.n_classes)
                
                unused_mask[chosen_ids] = False
            
            # Replace with unknowns
            for cl_u in unknown_class_ids:
                possible_mask = np.array([_y==cl_u]).flatten()
                possible_mask = possible_mask*unused_mask
                
                possible_ids = np.argwhere(possible_mask==1).flatten()
                n_r_samples_uc = int((self.percentage_novel)*self.chunk_size)
                chosen_ids = np.random.choice(possible_ids,
                                              n_r_samples_uc, 
                                              replace=False)
                
                random_to_remove = np.random.choice(len(chunk_X), len(chosen_ids), replace=False)
                chunk_X = np.array(chunk_X)
                chunk_y = np.array(chunk_y)
                chunk_X[random_to_remove] = _X[chosen_ids]
                chunk_y[random_to_remove] = _y[chosen_ids] - self.n_classes*(self.n_drifts)
                
                unused_mask[chosen_ids] = False
            
            perm = np.random.permutation(self.chunk_size)
            chunk_X = np.array(chunk_X)[perm]
            chunk_y = np.array(chunk_y)[perm]
            
            self.X.extend(chunk_X)
            self.y.extend(chunk_y)
                            
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        
    def get_chunk(self, chunk_id):
        start = chunk_id*self.chunk_size
        end = (chunk_id+1)*self.chunk_size
        _X = self.X[start:end]
        _y = self.y[start:end]
        if self.hide_label:
            _y[_y>self.n_classes] = self.n_classes
        
        return [_X, _y]
                
    def get_drift_gt(self):
        return self.drift_gt
    
    def get_novel_gt(self):
        return self.novel_gt
    
    