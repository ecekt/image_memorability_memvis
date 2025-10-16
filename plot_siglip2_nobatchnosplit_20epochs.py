import json
import matplotlib.pyplot as plt
import numpy as np

color_dict = {'CLIP': '#1f77b4', 'SigLIP2': '#2ca02c'}

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.figure(figsize=(6, 4))

with open('autoencoder_loss_correlation_stats_SigLIP2_nobatchNOSPLIT_20epochs_seed0.json', 'r') as f:
    epoch_corrs = json.load(f)

cor_scores = []
eps = []
for ep in epoch_corrs:
    cor_scores.append(epoch_corrs[ep])
    eps.append(int(ep) + 1)

plt.plot(eps, cor_scores, color=color_dict['SigLIP2'], label='SigLIP2', alpha=0.6, marker='.')
plt.xlabel('Epoch')
plt.ylabel('Spearman\'s Correlation Coefficient')
plt.xticks(np.arange(1, len(eps) + 1))
plt.title('SiGLIP2 Single Exposure Autoencoder Loss Correlation')
plt.tight_layout()
plt.savefig('siglip2_nobatchnosplit_20epochs_corrs.pdf', dpi=300, bbox_inches='tight')
#plt.show()