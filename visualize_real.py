from OWDSG import OWDSG
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# Load a real dataset (8x8 digit images, 10 classes), scale pixels to [0,1]
X_real, y_real = load_digits(return_X_y=True)
X_real = X_real / 16.

# Generate data stream from the real dataset
n_chunks = 100
n_c = 2
n_n = 2
n_drifts = 1
p_novel = 0.2
stream = OWDSG(n_drifts = n_drifts,
            n_novel = n_n,
            n_classes = n_c,
            n_chunks = n_chunks,
            chunk_size = 200,
            percentage_novel = p_novel,
            hide_label = False,
            X = X_real,
            y = y_real,
            smooth_drift = True,
            drift_magnitude = 1.8,
            n_drifting_classes=1)

# Fit PCA on the whole stream
pca = PCA(n_components=2, random_state=4567).fit(stream.X)
X_all = pca.transform(stream.X)

# Visualize
n_vis = 7
chunks_vis = np.linspace(1,n_chunks-1,n_vis).astype(int)

fig, ax = plt.subplots(1,n_vis,figsize=((2*n_vis),2*1.4), sharex=True, sharey=True)
title = 'D%i | N%i (P%i)' % (n_drifts,n_n,p_novel*100)

cols = plt.cm.coolwarm([0.3, 0.7, 0.0, 1.0])

for i_id, i_chunk in enumerate(chunks_vis):
    X, y = stream.get_chunk(i_chunk)
    X = pca.transform(X)
    for c in np.unique(y):
        ax[i_id].scatter(X[y==c,0], X[y==c,1], color=cols[c], s=3,
                         marker='o' if c<n_c else 'x')
    ax[i_id].grid(ls=':')
    ax[i_id].set_title('chunk %i' % i_chunk)
    
    if i_id==0:
        ax[i_id].set_ylabel('%s \n PC 2' % title)

    ax[i_id].set_xlabel('PC 1')
    ax[i_id].spines['top'].set_visible(False)
    ax[i_id].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/scatter_real.png')
