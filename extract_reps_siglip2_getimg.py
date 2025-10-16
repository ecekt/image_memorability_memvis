from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, Siglip2VisionModel, SiglipVisionModel
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

# Load the siglip2 model and processor
model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
# extract image embeds for memcat images (all)

# selected layer
layer = -1
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
            image_rep = model.get_image_features(**inputs)
            # cls only
            image_rep = image_rep.squeeze(0).tolist()
            selected_reps_dict[stripped_img_id].append(image_rep)

# save dict
with open('memcat_SIGLIP2_reps_dict_layer_GETIMGS.json', 'w') as f:
    json.dump(selected_reps_dict, f)