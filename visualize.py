from OWDSG import OWDSG
import matplotlib.pyplot as plt
import numpy as np

n_features = 2
n_chunks = 100
chunk_size = 500
n_drifts = 1
n_novel = 2
p_novel = 0.2
weights = [0.2,0.8]

# Generate 
stream = OWDSG(n_drifts=n_drifts, 
                    n_novel=n_novel, 
                    n_features=n_features,
                    n_informative=n_features,
                    n_chunks=n_chunks, 
                    chunk_size=chunk_size, 
                    percentage_novel=p_novel,
                    n_clusters_per_class=1,
                    class_sep=2,
                    random_state=4567, 
                    weights = weights)


# Visualize
n_vis = 7
chunks_vis = np.linspace(1,n_chunks-1,n_vis).astype(int)

fig, ax = plt.subplots(1,n_vis,figsize=((2*n_vis),2*1.2), sharex=True, sharey=True)
title = 'D%i | N%i (P%i) | W%i' % (n_drifts,n_novel,p_novel*100,weights[0]*100)

cols = plt.cm.coolwarm(np.linspace(0.2,0.5,(n_novel+2)))
cols -= 0.8
cols = np.clip(cols,0,1)

cols = np.array([[0., 0., 0., 1.] for i in range(n_novel+2)])
cols[0] = plt.cm.coolwarm([0.0])
cols[1] = plt.cm.coolwarm([1.0])

cols = plt.cm.coolwarm([0.3, 0.7, 0.0, 1.0])

for i_id, i_chunk in enumerate(chunks_vis):
    X, y = stream.get_chunk(i_chunk)
    ax[i_id].scatter(X[:,0], X[:,1], c=cols[y],s=1)
    ax[i_id].grid(ls=':')
    ax[i_id].set_title('chunk %i' % i_chunk)
    
    if i_id==0:
        ax[i_id].set_ylabel('%s \n feature 2' % title)

    ax[i_id].set_xlabel('feature 1')
    ax[i_id].spines['top'].set_visible(False)
    ax[i_id].spines['right'].set_visible(False)


plt.tight_layout()
plt.savefig('figures/scatter.png')
