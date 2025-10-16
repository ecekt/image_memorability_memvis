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

# Load the dinov2 model and processor
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model = AutoModel.from_pretrained('facebook/dinov2-base')

# extract image embeds for memcat images (all)

mems = []

activations_mean = defaultdict(list)
activations_sum = defaultdict(list)
activations_max = defaultdict(list)
activations_max_abs = defaultdict(list)

attentions_entropy = defaultdict(list)

activations_delta = defaultdict(list)
activations_delta_total = []

patch_regularity = defaultdict(list)

count = 0
with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # if row[6] == 'coco':
        count += 1
        print(count)
        img_id = row[1]
        cat = row[2]
        obj = row[3]
        mem = float(row[-1])
        mems.append(mem)

        path = 'data/memcat/MemCat_images/MemCat/' + cat + '/' + obj + '/' + img_id

        assert os.path.exists(path), f"File {path} does not exist."

        stripped_img_id = row[1].split('.')[0]

        image = Image.open(f'{path}').convert('RGB')

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
            # image_rep = model_full.get_image_features(**inputs)
            # vit-base 32
            # (visual_projection): Linear(in_features=768, out_features=512, bias=False)
            # (text_projection): Linear(in_features=512, out_features=512, bias=False)

            hidden_states = outputs.hidden_states
            attentions = outputs.attentions
            layers_cls = [hidden_states[i][:, 0, :] for i in range(len(hidden_states))]
            layers_patches = [hidden_states[i][:, 1:, :] for i in range(len(hidden_states))]

            hs = layers_cls
            att = attentions

            delta_total = 0.0
            for i, rep in enumerate(hs):
                # print(i, rep.shape)
                activations_mean[i].append(rep.mean().item())
                activations_sum[i].append(rep.sum().item())
                activations_max[i].append(rep.max().item())
                activations_max_abs[i].append(rep.abs().max().item())

                if i != len(att):
                    attn_mean = att[i][0].mean(dim=0).numpy()
                    attn_mean_cls = attn_mean[0, 1:]  # CLS to all patch tokens, remove CLS to CLS
                    # normalize
                    attn_mean_cls /= attn_mean_cls.sum()
                    # entropy of attention distribution
                    attn_entropy = entropy(attn_mean_cls, base=2)
                    attentions_entropy[i].append(attn_entropy.item())

                if i < len(hs) - 1:
                    # delta activation is the difference between current and next layer
                    next_rep = hs[i + 1]
                    # delta_activation = (next_rep - rep).mean().item()
                    # cosine distance
                    delta_activation = 1 - torch.nn.functional.cosine_similarity(rep, next_rep, dim=1).item()
                    activations_delta[i].append(delta_activation)
                    delta_total += delta_activation

            for p, reps in enumerate(layers_patches):
                # calculate patch regularity leave one out cosine sim
                reg = []
                for j in range(reps.shape[1]):
                    # leave one out
                    left_out = reps[:, j, :].unsqueeze(1)
                    others = torch.cat([reps[:, :j, :], reps[:, j + 1:, :]], dim=1)
                    cossim = torch.nn.functional.cosine_similarity(left_out, others, dim=2)
                    reg.append(np.mean(cossim.numpy()))

                patch_regularity[p].append(np.mean(reg).item())

            # store total delta activation for the image
            activations_delta_total.append(delta_total)

# save the dicts
with open('dinov2_activations_mean.json', 'w') as f:
    json.dump(activations_mean, f)

with open('dinov2_activations_sum.json', 'w') as f:
    json.dump(activations_sum, f)

with open('dinov2_activations_max.json', 'w') as f:
    json.dump(activations_max, f)

with open('dinov2_activations_max_abs.json', 'w') as f:
    json.dump(activations_max_abs, f)

with open('dinov2_attentions_entropy.json', 'w') as f:
    json.dump(attentions_entropy, f)

with open('dinov2_activations_delta.json', 'w') as f:
    json.dump(activations_delta, f)

with open('dinov2_activations_delta_total.json', 'w') as f:
    json.dump(activations_delta_total, f)

with open('dinov2_patch_regularity.json', 'w') as f:
    json.dump(patch_regularity, f)
