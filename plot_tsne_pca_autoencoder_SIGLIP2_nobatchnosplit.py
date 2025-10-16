# plot tsne and PCA for selected embeddings
# color based on memorability

import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv

name = 'SigLIP2'
# model with best correlation
seed = 0
epoch = 4

with open('autoencoder_representations_' + name + '_nobatchnosplit1e4_' + str(seed) + '_' + str(epoch) + '.json', 'r') as f:
    autoencoder_reps = json.load(f)

embeds = [] # autoencoder's latent reps (from the bottleneck layer, 100-dim)
mems = []

with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        img_id = row[1].split('.')[0]
        embeds.append(autoencoder_reps[img_id])
        mem = float(row[-1])
        mems.append(mem)

embeds = np.asarray(embeds)
embeds = StandardScaler().fit_transform(embeds)

# plot tsne
tsne = TSNE(n_components=2, random_state=42, perplexity=100, max_iter=1000)
tsne_results = tsne.fit_transform(embeds)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=mems, cmap='Spectral_r', alpha=1.0)
plt.colorbar(scatter, label='Memorability')
plt.title('t-SNE - SigLIP2 Autoencoder Latents')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('plot_TSNE_memcat_siglip2nosplitnobatches1e4_AUTOENCODER_LATENTS.pdf', dpi=300, bbox_inches='tight')
plt.close()


# # also pca
# pca = PCA(n_components=2)
# pca_results = pca.fit_transform(embeds)
# plt.figure(figsize=(8, 6))
# scatter_pca = plt.scatter(pca_results[:, 0], pca_results[:, 1], c=mems, cmap='Spectral_r', alpha=1.0)
# plt.colorbar(scatter_pca, label='Memorability')
# plt.title('PCA - CLIP Autoencoder Latents')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.tight_layout()
# plt.savefig('test.pdf', dpi=300, bbox_inches='tight')