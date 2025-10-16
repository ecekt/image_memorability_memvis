from PIL import Image
import requests
from transformers import AutoImageProcessor, AutoModel
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json
import csv
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
from scipy.stats import entropy
from cycler import cycler

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

mems = []
with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        mem = float(row[-1])
        mems.append(mem)

# load the internal signals
model_types = ['clip', 'dinov2', 'siglip2']
mode_strs = {'clip': 'CLIP', 'dinov2': 'DINOv2', 'siglip2': 'SigLIP2'}

# patch regularity
with open('clip_patch_regularity.json', 'r') as f:
    patch_regularity_clip = json.load(f)

with open('dinov2_patch_regularity.json', 'r') as f:
    patch_regularity_dinov2 = json.load(f)

with open('siglip2_patch_regularity.json', 'r') as f:
    patch_regularity_siglip2 = json.load(f)

to_plot_patch_clip = []
to_plot_patch_dinov2 = []
to_plot_patch_siglip2 = []

sigs_patch_clip = []
sigs_patch_dinov2 = []
sigs_patch_siglip2 = []
coefs_patch_clip = []
coefs_patch_dinov2 = []
coefs_patch_siglip2 = []

print('correlation between human memorability and patch uniformity')
for i in [str(j) for j in range(len(patch_regularity_clip))]:
    print(f'Layer {i}:')
    #print('CLIP')
    corr = spearmanr(mems, patch_regularity_clip[i])
    to_plot_patch_clip.append(corr.statistic)
    #print(corr)
    sigs_patch_clip.append(corr.pvalue)
    coefs_patch_clip.append(corr.statistic)

    #print('DINOv2')
    corr = spearmanr(mems, patch_regularity_dinov2[i])
    to_plot_patch_dinov2.append(corr.statistic)
    #print(corr)
    sigs_patch_dinov2.append(corr.pvalue)
    coefs_patch_dinov2.append(corr.statistic)

    #print('SigLIP2')
    corr = spearmanr(mems, patch_regularity_siglip2[i])
    to_plot_patch_siglip2.append(corr.statistic)
    #print(corr)
    sigs_patch_siglip2.append(corr.pvalue)
    coefs_patch_siglip2.append(corr.statistic)

# print sigs and coefs as a table
print('\nCLIP Patch uniformity:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_patch_clip):
    print(f'{i}\t{coef:.4f}\t{sigs_patch_clip[i]:.4f}')

print('\nDINOv2 Patch uniformity:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_patch_dinov2):
    print(f'{i}\t{coef:.4f}\t{sigs_patch_dinov2[i]:.4f}')

print('\nSigLIP2 Patch uniformity:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_patch_siglip2):
    print(f'{i}\t{coef:.4f}\t{sigs_patch_siglip2[i]:.4f}')

# plot
plt.figure(figsize=(5, 3))
# plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.plot(range(0, len(to_plot_patch_clip)), to_plot_patch_clip, label='CLIP', marker='.')
plt.plot(range(0, len(to_plot_patch_dinov2)), to_plot_patch_dinov2, label='DINOv2', marker='.')
plt.plot(range(0, len(to_plot_patch_siglip2)), to_plot_patch_siglip2, label='SigLIP2', marker='.')
plt.xlabel('Layer')
plt.ylabel('Spearman\'s  Correlation Coefficient')
plt.ylim(-0.4, 0.4)
plt.title('Human Memorability vs. Patch Uniformity')
plt.xticks(range(0, len(to_plot_patch_clip)))
plt.legend()
plt.tight_layout()
plt.savefig('memorability_correlations_patch_uniformity.pdf', dpi=300, bbox_inches='tight')
plt.close()

