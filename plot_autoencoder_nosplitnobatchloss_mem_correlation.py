import json
import matplotlib.pyplot as plt
import numpy as np

reps_name = ['CLIP', 'SigLIP2']

color_dict = {'CLIP': '#1f77b4', 'SigLIP2': '#2ca02c'}

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.figure(figsize=(6, 4))

x = np.arange(5)
width = 0.2

model_count = 0

for i, name in enumerate(reps_name):
    print(name)
    if model_count == 0:
        offsets = x + (i + 3.5) * width
        model_count = 1
    else:
        offsets = x + (i + 4.5) * width

    if name == 'CLIP':
        path = 'autoencoder_loss_correlation_stats_CLIP.json'
        path_nosplit = 'autoencoder_loss_correlation_stats_CLIP_nobatchnosplit1e4.json'
    elif name == 'SigLIP2':
        path = 'autoencoder_loss_correlation_stats_SigLIP2.json'
        path_nosplit = 'autoencoder_loss_correlation_stats_SigLIP2_nobatchnosplit1e4.json'

    with open(path, 'r') as f:
        epoch_coefs = json.load(f)

    with open(path_nosplit, 'r') as f:
        epoch_coefs_nosplit = json.load(f)

    # plot the results
    for j in x:
        ep_mean = np.mean(epoch_coefs[str(j)])
        ep_std = np.std(epoch_coefs[str(j)])
        plt.bar(offsets[j], ep_mean, yerr=ep_std, label=name if j == 0 else "",
                color=color_dict[name], alpha=0.4, width=width)
        plt.errorbar(offsets[j], ep_mean, yerr=ep_std, color='black', capsize=3, elinewidth=0.5)

    # plot the nosplit results
    for j in x:
        ep_mean_nosplit = np.mean(epoch_coefs_nosplit[str(j)])
        ep_std_nosplit = np.std(epoch_coefs_nosplit[str(j)])
        plt.bar(offsets[j] + width, ep_mean_nosplit, yerr=ep_std_nosplit,
                label=name + ' (Single)' if j == 0 else "",
                color=color_dict[name], alpha=1, width=width)
        plt.errorbar(offsets[j] + width, ep_mean_nosplit, yerr=ep_std_nosplit, color='black', capsize=3, elinewidth=0.5)


plt.xlabel('Epoch')
plt.ylabel('Spearman\'s Correlation Coefficient')
plt.xticks(np.arange(1, len(epoch_coefs) + 1))
plt.ylim(0.2, 0.6)
plt.title('Human Memorability vs. Sparse Autoencoder Loss')
plt.legend(fontsize=8, loc='lower right')
plt.tight_layout()
#plt.show()
plt.savefig('sparse_autoencoder_NOSPLITNOBATCH1e4_coefs_all_models.pdf', dpi=300, bbox_inches='tight')
plt.close()
