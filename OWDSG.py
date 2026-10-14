import numpy as np
from sklearn.datasets import make_classification
from scipy.spatial.distance import pdist

class OWDSG:
    def __init__(self, n_drifts, n_novel, even_gt=True, 
                 hide_label=False, 
                 n_features=10, n_informative = 10,
                 percentage_novel=0.1, 
                 n_classes=2, weights=None,
                 n_chunks=200, chunk_size=200, 
                 n_clusters_per_class=1, class_sep=1.,
                 random_state = None, allow_projection=True,
                 X=None, y=None, class_order=None,
                 drift_magnitude=1.0, smooth_drift=False, n_drifting_classes=None):
        
        """ Non-stationary data stream generator combining the events of concept drift and novel class ocurrence.

        :param n_drifts: Number of concept drifts in the stream.
            Two drift mechanisms:
            - sudden drift -- a different source class is substituted under the
              same label; n_classes*(n_drifts+1) + n_novel classes are needed;
            - smooth drift -- the class samples are gradually shifted 
              along a random direction; n_classes + n_novel classes are needed.
        :param smooth_drift: Which drift mechanism to use. False (default)
            applies sudden drift. True applies smooth drift. 
            Both work for generated and for provided data.
        :param drift_magnitude: Approximate length of the shift vector in smooth drift,
            expressed in mean distance between class centroids. Ignored for 
            sudden drift.
        :param n_drifting_classes: Number of known classes shifted at each smooth 
            drift; the classes are chosen at random for every drift. All known 
            classes by default. Ignored for sudden drift.
        :param n_novel: Number of new classes in the stream.
        :param even_gt: The boolean value describing if the concept drifts and 
            the moments on new class novelties are equally distributed across 
            the stream.
        :param hide_label: The boolean value describing if labels of all 
            unknown classes are unified to a common value.
        :param n_features: Dimensionality of the data. Ignored when X is given.
        :param n_informative: Number of informative features.
        :param percentage_novel: The proportion of a chunk that will 
            be occupied by novel class samples.
        :param n_classes: Number of known classes.
        :param weights: Proportions of known classes.
        :param n_chunks: Length of the data stream in batches.
        :param chunk_size:  Number of problem samples in each batch.
        :param n_clusters_per_class: Number of clusters in each class.
        :param class_sep: The side lengths of a hypercube for data generation.
        :param random_state: Random seed for data generation .
        :param allow_projection: The boolean value determining if the random 
            feature projections can be performed. If the value is False, 
            the generation process will result in an error when prerequisites 
            are not satisfied. Otherwise, the method will perform projections 
            to a requested dimensionality.
        :param X: Optional real dataset to sample from instead of generating 
            with make_classification. When given, the stream is built by drawing 
            from provised samples. Known classes are present in every chunk, 
            novel classes appearing progressively.
        :param y: Labels for X. Required when X is provided.
        :param class_order: Which original labels take which role (known/novel). 
            The first n_classes entries (n_classes*(n_drifts+1) for sudden drift, 
            in drift order) become the known classes, the next n_novel become 
            the novel ones. Defaults to sorted unique labels.


        :type n_drifts: integer
        :type n_novel: integer
        :type even_gt: boolean, optional (default=True)
        :type hide_label: boolean, optional (default=False)
        :type n_features: integer, optional (default=10)
        :type n_informative: integer, optional (default=10)
        :type percentage_novel: float, optional (default=0.1)
        :type n_classes: integer, optional (default=2)
        :type weights: tuple, optional (balanced by default)
        :type n_chunks: integer, optional (default=200)
        :type chunk_size: integer, optional (default=200)
        :type n_clusters_per_class: integer, optional (default=1)
        :type class_sep: float, optional (default=1)
        :type random_state: integer, optional (default=None)
        :type allow_projection: boolean, optional (default=True)
        :type X: array-like, optional (default=None)
        :type y: array-like, optional (default=None)
        :type class_order: array-like, optional (default=None)
        :type drift_magnitude: float, optional (default=1.0)
        :type smooth_drift: boolean, optional (default=False)
        :type n_drifting_classes: integer, optional (default=n_classes)
        """
        
        self.drift_magnitude = drift_magnitude
        self.smooth_drift = smooth_drift
        self.n_drifting_classes = n_classes if n_drifting_classes is None else n_drifting_classes
        if not 0 < self.n_drifting_classes <= n_classes:
            raise ValueError('n_drifting_classes must be between 1 and n_classes')

        if (X is None) != (y is None):
            raise ValueError('Both X and y must be provided when sampling from real data.')
        self._provided_data = X is not None

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
        self.drift_gt = self._event_points(self.n_drifts)
        self.novel_gt = self._event_points(self.n_novel)

        # Sudden drift -- substitute a different source class
        # Smooth drift -- translate (move) concept
        self._n_concepts = 1 if self.smooth_drift else self.n_drifts + 1
        _n_classes = self.n_classes*self._n_concepts + self.n_novel
        
        # Prepare for feature projections (if required and allowed)
        expected_min_features = _n_classes*self.n_clusters_per_class
        project = (not self._provided_data
                   and 2**self.n_informative < expected_min_features
                   and self.allow_projection)
        n_features_mc, n_informative_mc = self.n_features, self.n_informative
        if project:
            print('Performing feature projections from %i to %i dimensions. The value for n_informative is ignored.' % (expected_min_features, self.n_features))
            n_features_mc = n_informative_mc = expected_min_features
        
        if self._provided_data:
            _X, _y = self._prepare_provided(X, y, _n_classes, class_order)
        else:
            _X, _y = self._generate(_n_classes, n_features_mc, n_informative_mc)

        # Feature projections (generated only)
        if project:
            projection_matrix = np.random.normal(size=(n_features_mc, self.n_features))
            _X = _X@projection_matrix
            _X /= n_features_mc

        self._build_stream(_X, _y)

    def _event_points(self, n_events):
        """Chunk indices of n_events events, evenly spaced or random (even_gt)."""
        if n_events == 0:
            return []
        interval = self.n_chunks//n_events
        if self.even_gt:
            return interval//2 + interval*np.arange(n_events)
        return np.sort(np.random.choice(np.arange(interval//4, self.n_chunks), n_events, replace=False))

    def _generate(self, _n_classes, n_features_mc, n_informative_mc):
        return make_classification(n_samples=_n_classes*self.chunk_size*self.n_chunks,
                                    n_features=n_features_mc,
                                    n_repeated=0,
                                    n_redundant=0,
                                    n_clusters_per_class=self.n_clusters_per_class,
                                    class_sep=self.class_sep,
                                    n_informative=n_informative_mc,
                                    n_classes=_n_classes,
                                    flip_y=0.0,
                                    random_state=self.random_state)

    def _prepare_provided(self, X, y, _n_classes, class_order):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        order = np.unique(y) if class_order is None else np.asarray(class_order)
        if len(order) < _n_classes:
            raise ValueError("The dataset has %i classes, but the stream requires %i (%i known x %i concept(s) + %i novel)" % (len(order), _n_classes, self.n_classes, self._n_concepts, self.n_novel))

        self.n_features = X.shape[1]

        keep = order[:_n_classes]
        mask = np.isin(y, keep)
        
        roles = np.argmax(y[mask][:, None] == keep[None, :], axis=1)
        return X[mask], roles

    def _draw(self, role, n_samples):
        """Draw n_samples unused indices of a class and mark them as used."""
        pool = self._unused[role]
        if len(pool) >= n_samples:
            chosen = np.random.choice(pool, n_samples, replace=False)
        else:
            if role not in self._exhausted:
                self._exhausted.add(role)
                print("Class %i exhausted and will be sampled with replacement" % role)
            # if the pool of unused samples is empty -- use all samples of the class
            chosen = np.random.choice(pool if len(pool) else self._all_ids[role],
                                      n_samples, replace=True)
        self._unused[role] = pool[np.isin(pool, chosen)==False]
        return chosen

    def _drift_shifts(self, _X):
        """Per-chunk shift of the drifting known classes for smooth drift."""
        centroids = np.vstack([_X[ids].mean(axis=0) for ids in self._all_ids.values()])
        
        # mean distance over all pairs of different classes times magnitude
        scale = pdist(centroids).mean() * self.drift_magnitude

        shift = np.zeros((self.n_chunks, self.n_classes, self.n_features))
        drift_points = list(self.drift_gt) + [self.n_chunks]
        
        for d, next_drift in zip(drift_points[:-1], drift_points[1:]):
            shift_end = d + (next_drift - d) // 2
            progress = np.minimum(1.0, np.arange(self.n_chunks - d) / (shift_end - d))[:, None]
            
            for c in np.random.choice(self.n_classes, self.n_drifting_classes, replace=False):
                u = np.random.normal(size=self.n_features) / np.sqrt(self.n_features)
                shift[d:, c] += u * scale * progress
                
        return shift

    def _build_stream(self, _X, _y):
        """Arrange samples into a stream: known classes in every chunk, novel classes added progressively."""
        self._all_ids = {int(c): np.where(_y == c)[0] for c in np.unique(_y)}
        
        self._unused = dict(self._all_ids)  
        self._exhausted = set()            

        novel_base = self.n_classes*self._n_concepts
        shifts = self._drift_shifts(_X) if self.smooth_drift else None

        n_known = (np.asarray(self.weights)*self.chunk_size).astype(int)
        n_known[-1] = self.chunk_size - n_known[:-1].sum()
        n_novel_samples = int(self.percentage_novel*self.chunk_size)

        chunks_X = []
        chunks_y = []
        concept = 0
        novel_roles = []

        for i in range(self.n_chunks):
            
            # Establish roles of known and unknown classes for this chunk
            if i in self.drift_gt:
                concept = min(concept + 1, self._n_concepts - 1)
            if i in self.novel_gt:
                novel_roles.append(novel_base + len(novel_roles))

            # Sample knowns
            parts_X = []
            parts_y = []
            for label in range(self.n_classes):
                role = concept*self.n_classes + label
                chosen_ids = self._draw(role, n_known[label])
                samples = _X[chosen_ids]
                if self.smooth_drift:
                    samples = samples + shifts[i, label]
                parts_X.append(samples)
                parts_y.append(np.full(len(chosen_ids), label))
            chunk_X = np.concatenate(parts_X)
            chunk_y = np.concatenate(parts_y)
            
            # Replace with unknowns
            for role in novel_roles:
                chosen_ids = self._draw(role, n_novel_samples)
                random_to_remove = np.random.choice(len(chunk_X), len(chosen_ids), replace=False)
                chunk_X[random_to_remove] = _X[chosen_ids]
                chunk_y[random_to_remove] = role - novel_base + self.n_classes
            
            perm = np.random.permutation(self.chunk_size)
            chunks_X.append(chunk_X[perm])
            chunks_y.append(chunk_y[perm])
                            
        self.X = np.concatenate(chunks_X)
        self.y = np.concatenate(chunks_y)
        
    def get_chunk(self, chunk_id):
        start = chunk_id*self.chunk_size
        end = (chunk_id+1)*self.chunk_size
        _X = self.X[start:end]
        _y = self.y[start:end].copy()
        if self.hide_label:
            _y[_y>=self.n_classes] = self.n_classes
        
        return [_X, _y]
                
    def get_drift_gt(self):
        return self.drift_gt
    
    def get_novel_gt(self):
        return self.novel_gt
    
