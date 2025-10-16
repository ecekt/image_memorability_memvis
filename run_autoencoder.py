import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoImageProcessor, ViTMAEForPreTraining
import requests
from PIL import Image
import json
import random
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import numpy as np
import evaluate
import csv
import os

processor_vm_base = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
model_vm_base = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

processor_vm_large = AutoImageProcessor.from_pretrained('facebook/vit-mae-large')
model_vm_large = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-large')

memcat_coco = defaultdict(float)
mems = []

dict_vm_losses_base = defaultdict(float)
dict_vm_losses_large = defaultdict(float)
vm_losses_base = []
vm_losses_large = []

count = 0
with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        count += 1
        if count % 100 == 0:
            print(count)

        img_id = row[1]
        cat = row[2]
        obj = row[3]
        mem = float(row[-1])
        mems.append(mem)

        path = 'data/memcat/MemCat_images/MemCat/' + cat + '/' + obj + '/' + img_id

        assert os.path.exists(path), f"File {path} does not exist."

        image = Image.open(f'{path}').convert('RGB')

        with torch.no_grad():
            inputs_vm = processor_vm_base(images=image, return_tensors="pt")
            outputs = model_vm_base(**inputs_vm)
            loss = outputs.loss.item()
            vm_losses_base.append(loss)
            dict_vm_losses_base[img_id] = loss

            inputs_vm = processor_vm_large(images=image, return_tensors="pt")
            outputs = model_vm_large(**inputs_vm)
            loss_large = outputs.loss.item()
            vm_losses_large.append(loss_large)
            dict_vm_losses_large[img_id] = loss_large

print(count)

with open('dict_memcat_ALL_vitmae_LARGE_losses.json', 'w') as f:
    json.dump(dict_vm_losses_large, f)

with open('dict_memcat_ALL_vitmae_BASE_losses.json', 'w') as f:
    json.dump(dict_vm_losses_base, f)

