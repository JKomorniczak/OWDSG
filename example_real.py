from OWDSG import OWDSG
import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

# Load a real dataset (8x8 digit images, 10 classes) and scale pixels to [0,1]
X_real, y_real = load_digits(return_X_y=True)
X_real = X_real / 16.
X_real += np.random.normal(size=X_real.shape)

# Generate data stream from the real dataset
n_chunks = 100
n_c = 5
n_n = 3
stream = OWDSG(n_drifts = 2,
            n_novel = n_n,
            n_classes = n_c,
            n_chunks = n_chunks,
            chunk_size = 100,
            percentage_novel = 0.1,
            random_state = 443,
            hide_label = False,
            X = X_real,
            y = y_real,
            smooth_drift = True,
            drift_magnitude = 0.7)


# Get first data chunk
X, y = stream.get_chunk(0)
print('chunk shape:', X.shape)
print('unique labels (first chunk):', np.unique(y, return_counts=True))

# Get last data chunk
X, y = stream.get_chunk(n_chunks-1)
print('unique labels(last chunk):', np.unique(y, return_counts=True))

# Get ground-truth of concept drift
print('Concept drift in chunks:', stream.get_drift_gt())

# Get ground-truth of novelty
print('Novelty in chunks:', stream.get_novel_gt())

# Exemplary experiment -- UC recognition with MLP and support thresholding
clf = MLPClassifier(hidden_layer_sizes=(10))
scores = []
inner_scores = []
threshold = 0.95

for chunk in range(n_chunks):
    _X, _y = stream.get_chunk(chunk)

    # Filter KC for model training
    _X_known = _X[_y<n_c]
    _y_known = _y[_y<n_c]

    if chunk==0:
        [clf.partial_fit(_X_known, _y_known, np.arange(n_c).astype(int)) for i in range(5)]
    else:
        ### Test
        proba = clf.predict_proba(_X)

        # Establish unknowns based on threshold
        known_mask = (_y<n_c).astype(int)
        preds_outer = (np.max(proba, axis=1)>threshold).astype(int)

        # Calculate outer score
        outer_score = balanced_accuracy_score(known_mask, preds_outer)
        scores.append(outer_score)

        # Calculate inner score -- classification quality on KC samples
        preds_inner = np.argmax(proba, axis=1)[_y<n_c]
        inner_scores.append(balanced_accuracy_score(_y_known, preds_inner))

        ### Train
        [clf.partial_fit(_X_known, _y_known) for i in range(10)]


# Plot results
fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.plot(np.arange(1,n_chunks), scores, color='r', label='outer score')
ax.plot(np.arange(1,n_chunks), inner_scores, color='b', label='inner score')
ax.set_xticks(np.concatenate([stream.get_drift_gt(), stream.get_novel_gt()]))

ax.grid(ls=':')
ax.set_xlim(0,n_chunks)
ax.set_xlabel('chunk')
ax.set_ylabel('score')
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/example_real.png')
