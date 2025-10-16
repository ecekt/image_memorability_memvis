from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel, CLIPVisionModel
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

# Load the CLIP model and processor
# model_full = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True, output_attentions=True)
model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", output_hidden_states=True)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# extract image embeds for memcat images (all)

mems = []

activations_delta = []

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
            rep_first = hs[0]
            rep_last = hs[-1]

            delta_activation = 1 - torch.nn.functional.cosine_similarity(rep_first, rep_last, dim=1).item()
            activations_delta.append(delta_activation)

with open('clip_activations_delta_first-last.json', 'w') as f:
    json.dump(activations_delta, f)
