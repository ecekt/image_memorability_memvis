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

# selected layer
layer = 10
selected_reps = []
selected_reps_dict = defaultdict(list)

mems = []

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
            # cls only
            image_rep = hidden_states[layer].squeeze(0)[0].tolist()
            selected_reps.append(image_rep)
            selected_reps_dict[stripped_img_id].append(image_rep)

# save
with open('memcat_DINOV2_reps_layer_LAYER10.json', 'w') as f:
    json.dump(selected_reps, f)

with open('memcat_DINOV2_memscores_layer_LAYER10.json', 'w') as f:
    json.dump(mems, f)

# save dict
with open('memcat_DINOV2_reps_dict_layer_LAYER10.json', 'w') as f:
    json.dump(selected_reps_dict, f)