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

for model_type in model_types:
    model_str = mode_strs[model_type]
    print(f'Processing {model_str} model...')

    with open(model_type + '_activations_mean.json', 'r') as f:
        activations_mean = json.load(f)
    with open(model_type + '_activations_max.json', 'r') as f:
        activations_max = json.load(f)
    with open(model_type + '_activations_max_abs.json', 'r') as f:
        activations_max_abs = json.load(f)

    with open(model_type + '_activations_delta_total.json', 'r') as f:
        activations_delta_total = json.load(f)

    # plot correlations to human mem scores
    plt.figure(figsize=(5, 3))
    # plt.grid(color='gray', linestyle='--', linewidth=0.5)
    to_plot_mean = []
    to_plot_max = []
    to_plot_max_abs = []

    sigs_mean = []
    sigs_max = []
    sigs_max_abs = []
    coefs_mean = []
    coefs_max = []
    coefs_max_abs = []

    print('HIDDEN ACTIVATIONS')
    for i in [str(j) for j in range(len(activations_mean))]:
        # skip 0 as it is constant
        print('Computing correlations for layer', i)
        if i == '0':
            continue

        #print('correlation between human memorability and mean activation')
        corr = spearmanr(mems, activations_mean[i])
        to_plot_mean.append(corr.statistic)
        #print(corr)
        sigs_mean.append(corr.pvalue)
        coefs_mean.append(corr.statistic)

        #print('correlation between human memorability and max activation')
        corr = spearmanr(mems, activations_max[i])
        to_plot_max.append(corr.statistic)
        #print(corr)
        sigs_max.append(corr.pvalue)
        coefs_max.append(corr.statistic)

       # print('correlation between human memorability and max abs activation')
        corr = spearmanr(mems, activations_max_abs[i])
        to_plot_max_abs.append(corr.statistic)
        #print(corr)
        sigs_max_abs.append(corr.pvalue)
        coefs_max_abs.append(corr.statistic)

    # print sigs and coefs as a table
    print(f'\n{model_str} Mean Activations:')
    print('Layer\tSpearman Coef\tP-value')
    for i, coef in enumerate(coefs_mean):
        print(f'{i+1}\t{coef:.4f}\t{sigs_mean[i]:.4f}')

    print(f'\n{model_str} Max Activations:')
    print('Layer\tSpearman Coef\tP-value')
    for i, coef in enumerate(coefs_max):
        print(f'{i+1}\t{coef:.4f}\t{sigs_max[i]:.4f}')

    print(f'\n{model_str} Max Abs Activations:')
    print('Layer\tSpearman Coef\tP-value')
    for i, coef in enumerate(coefs_max_abs):
        print(f'{i+1}\t{coef:.4f}\t{sigs_max_abs[i]:.4f}')

    print('correlation between human memorability and attention delta total')
    corr_total = spearmanr(mems, activations_delta_total)
    #print(corr_total)

    # print sigs and coefs as a table
    print(f'\n{model_str} Total Delta Activations:')
    print('Spearman Coef\tP-value')
    print(f'{corr_total.statistic:.4f}\t{corr_total.pvalue:.4f}')

    # plot
    chosen_colors = [plt.cm.tab10.colors [i] for i in [3, 4, 9]]
    custom_cycler = cycler(color=chosen_colors)
    plt.rcParams['axes.prop_cycle'] = custom_cycler #plt.cycler(plt.cm.Set1.colors)
    plt.plot(range(1, len(to_plot_mean) + 1), to_plot_mean, label='Mean Activation', marker='.')
    plt.plot(range(1, len(to_plot_max) + 1), to_plot_max, label='Max Activation', marker='.')
    plt.plot(range(1, len(to_plot_max_abs) + 1), to_plot_max_abs, label='Max Abs Activation', marker='.')

    plt.xlabel('Layer')
    plt.title('Human Memorability vs. ' + model_str + ' Activations')
    plt.xticks(range(1, len(to_plot_mean) + 1))
    plt.ylim(-0.4, 0.4)
    if model_str == 'CLIP':
        # show ylabel and legend once
        plt.ylabel('Spearman\'s  Correlation Coefficient')
        plt.legend()
    plt.tight_layout()
    plt.savefig(model_str + '_memorability_correlations.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# reset colors
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)

with open('clip_activations_delta.json', 'r') as f:
    activations_delta_clip = json.load(f)

with open('dinov2_activations_delta.json', 'r') as f:
    activations_delta_dinov2 = json.load(f)

with open('siglip2_activations_delta.json', 'r') as f:
    activations_delta_siglip2 = json.load(f)

to_plot_clip = []
to_plot_dinov2 = []
to_plot_siglip2 = []

sigs_delta_clip = []
sigs_delta_dinov2 = []
sigs_delta_siglip2 = []
coefs_delta_clip = []
coefs_delta_dinov2 = []
coefs_delta_siglip2 = []

print('correlation between human memorability and attention delta')
for i in [str(j) for j in range(len(activations_delta_clip))]:
    print(f'Layer {i}:')
    #print('CLIP')
    corr = spearmanr(mems, activations_delta_clip[i])
    to_plot_clip.append(corr.statistic)
    #print(corr)
    sigs_delta_clip.append(corr.pvalue)
    coefs_delta_clip.append(corr.statistic)

    #print('DINOv2')
    corr = spearmanr(mems, activations_delta_dinov2[i])
    to_plot_dinov2.append(corr.statistic)
    #print(corr)
    sigs_delta_dinov2.append(corr.pvalue)
    coefs_delta_dinov2.append(corr.statistic)

    #print('SigLIP2')
    corr = spearmanr(mems, activations_delta_siglip2[i])
    to_plot_siglip2.append(corr.statistic)
    #print(corr)
    sigs_delta_siglip2.append(corr.pvalue)
    coefs_delta_siglip2.append(corr.statistic)

# print sigs and coefs as a table
print('\nCLIP Activations Delta:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_delta_clip):
    print(f'{i+1}\t{coef:.4f}\t{sigs_delta_clip[i]:.4f}')

print('\nDINOv2 Activations Delta:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_delta_dinov2):
    print(f'{i+1}\t{coef:.4f}\t{sigs_delta_dinov2[i]:.4f}')

print('\nSigLIP2 Activations Delta:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_delta_siglip2):
    print(f'{i+1}\t{coef:.4f}\t{sigs_delta_siglip2[i]:.4f}')

# plot
plt.figure(figsize=(5, 3))
# plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.plot(range(1, len(to_plot_clip) + 1), to_plot_clip, label='CLIP', marker='.')
plt.plot(range(1, len(to_plot_dinov2) + 1), to_plot_dinov2, label='DINOv2', marker='.')
plt.plot(range(1, len(to_plot_siglip2) + 1), to_plot_siglip2, label='SigLIP2', marker='.')
plt.xlabel('Layer')
plt.ylabel('Spearman\'s  Correlation Coefficient')
plt.ylim(-0.4, 0.4)
plt.title('Human Memorability vs. Activation Delta')
ticks = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']
plt.xticks(range(1, len(to_plot_clip) + 1), ticks)
plt.legend()
plt.tight_layout()
plt.savefig('memorability_correlations_activation_delta.pdf', dpi=300, bbox_inches='tight')
plt.close()


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

print('correlation between human memorability and patch regularity')
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
print('\nCLIP Patch Regularity:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_patch_clip):
    print(f'{i}\t{coef:.4f}\t{sigs_patch_clip[i]:.4f}')

print('\nDINOv2 Patch Regularity:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_patch_dinov2):
    print(f'{i}\t{coef:.4f}\t{sigs_patch_dinov2[i]:.4f}')

print('\nSigLIP2 Patch Regularity:')
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
plt.title('Human Memorability vs. Patch Regularity')
plt.xticks(range(0, len(to_plot_patch_clip)))
plt.legend()
plt.tight_layout()
plt.savefig('memorability_correlations_patch_regularity.pdf', dpi=300, bbox_inches='tight')
plt.close()


# attention entropy
with open('clip_attentions_entropy.json', 'r') as f:
    attentions_entropy_clip = json.load(f)

with open('dinov2_attentions_entropy.json', 'r') as f:
    attentions_entropy_dinov2 = json.load(f)

with open('siglip2_attentions_entropy.json', 'r') as f:
    attentions_entropy_siglip2 = json.load(f)

to_plot_ent_clip = []
to_plot_ent_dinov2 = []
to_plot_ent_siglip2 = []

sigs_ent_clip = []
sigs_ent_dinov2 = []
sigs_ent_siglip2 = []
coefs_ent_clip = []
coefs_ent_dinov2 = []
coefs_ent_siglip2 = []

print('correlation between human memorability and attention entropy')
for i in [str(j) for j in range(len(attentions_entropy_clip))]:
    print(f'Layer {i}:')
    #print('CLIP')
    corr = spearmanr(mems, attentions_entropy_clip[i])
    to_plot_ent_clip.append(corr.statistic)
    #print(corr)
    sigs_ent_clip.append(corr.pvalue)
    coefs_ent_clip.append(corr.statistic)

    #print('DINOv2')
    corr = spearmanr(mems, attentions_entropy_dinov2[i])
    to_plot_ent_dinov2.append(corr.statistic)
    #print(corr)
    sigs_ent_dinov2.append(corr.pvalue)
    coefs_ent_dinov2.append(corr.statistic)

    #print('SigLIP2')
    corr = spearmanr(mems, attentions_entropy_siglip2[i])
    to_plot_ent_siglip2.append(corr.statistic)
    #print(corr)
    sigs_ent_siglip2.append(corr.pvalue)
    coefs_ent_siglip2.append(corr.statistic)

# print sigs and coefs as a table
print('\nCLIP Attention Entropy:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_ent_clip):
    print(f'{i}\t{coef:.4f}\t{sigs_ent_clip[i]:.4f}')

print('\nDINOv2 Attention Entropy:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_ent_dinov2):
    print(f'{i}\t{coef:.4f}\t{sigs_ent_dinov2[i]:.4f}')

print('\nSigLIP2 Attention Entropy:')
print('Layer\tSpearman Coef\tP-value')
for i, coef in enumerate(coefs_ent_siglip2):
    print(f'{i}\t{coef:.4f}\t{sigs_ent_siglip2[i]:.4f}')

# plot
plt.figure(figsize=(5, 3))
# plt.grid(color='gray', linestyle='--', linewidth=0.7)
plt.plot(range(0, len(to_plot_ent_clip)), to_plot_ent_clip, label='CLIP', marker='.')
plt.plot(range(0, len(to_plot_ent_dinov2)), to_plot_ent_dinov2, label='DINOv2', marker='.')
plt.plot(range(0, len(to_plot_ent_siglip2)), to_plot_ent_siglip2, label='SigLIP2', marker='.')
plt.xlabel('Layer')
plt.ylabel('Spearman\'s  Correlation Coefficient')
plt.ylim(-0.4, 0.4)
plt.title('Human Memorability vs. Attention Entropy')
plt.xticks(range(0, len(to_plot_ent_clip)))
plt.legend()
plt.tight_layout()
plt.savefig('memorability_correlations_attention_entropy.pdf', dpi=300, bbox_inches='tight')
plt.close()


# clip first-last delta activations
with open('clip_activations_delta_first-last.json', 'r') as f:
    clip_first_last_delta = json.load(f)

print('correlation between human memorability and first-last delta activations for CLIP')
corr = spearmanr(mems, clip_first_last_delta)
print(corr)
