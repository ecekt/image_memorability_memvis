# train simple autoencoder
import torch
import json
import numpy as np
import csv
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt

with open('memcat_CLIP_reps_dict_layer_LAST.json', 'r') as f:
    image_representations = json.load(f)

# simple autoencoder model
class AutoEncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AutoEncoderModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inps):
        # inps = self.dropout(inps)
        inps_encoded = self.fc1(inps)
        inps_decoded = F.relu(inps_encoded)  # todo batchtopk
        inps_decoded = self.fc2(inps_decoded)

        return inps_decoded, inps_encoded

name = 'CLIP'
# model with best correlation
seed = 2025
epoch = 4

with open('memcat_CLIP_reps_dict_layer_LAST.json', 'r') as f:
    image_representations = json.load(f)

model_path = 'SPARSE_autoencoder_model_FULLmemcat_nobatchnosplit1e4_' + name + '_' + str(seed) + '_' + str(epoch) + '.pt'
# load model
model = AutoEncoderModel(input_size=768, hidden_size=100)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

criterion = nn.MSELoss(reduction='sum')

image_data = []
for img in image_representations:
    image_data.append(torch.tensor(image_representations[img], dtype=torch.float32))

mean = torch.stack(image_data).mean(dim=0)
std = torch.stack(image_data).std(dim=0)
# print(mean, std)
# image_data = [(x - mean) / (std + 1e-8) for x in image_data]

autoencoder_reps = defaultdict(list)
min_loss = math.inf
min_img = ''
max_loss = -math.inf
max_img = ''

losses_all = []
imgs_all = []

for img in image_representations:
    rep = torch.tensor(image_representations[img], dtype=torch.float32)
    rep = (rep - mean) / (std + 1e-8)  # normalize

    outputs_dec, outputs_enc = model(rep)
    loss = (criterion(outputs_dec, rep)).item()  # only reconstruction loss

    losses_all.append(loss)
    imgs_all.append(img)

    if loss < min_loss:
        min_loss = loss
        min_img = img
    if loss > max_loss:
        max_loss = loss
        max_img = img

    # latents in autoencoder
    autoencoder_reps[img] = outputs_enc.squeeze(0).detach().tolist()

    if img == '000000543264':
        print('Most memorable image reconstruction loss:', loss)
    elif img == 'sun_acgqrysmhvkbzwyo':
        print('Least memorable image reconstruction loss:', loss)

# siglip2 nobatch nosplit epoch 4 seed 0
# Most memorable image reconstruction loss: 294.9536437988281
# Least memorable image reconstruction loss: 219.02182006835938
# this below is clip
# Most memorable image reconstruction loss: 287.64935302734375
# Least memorable image reconstruction loss: 139.25709533691406

print('model min loss:', min_loss, 'image:', min_img)
print('model max loss:', max_loss, 'image:', max_img)
# # siglip2
# model min loss: 99.11082458496094 image: n02396427_104
# model max loss: 736.5144653320312 image: n02834778_6712

# # save autoencoder representations
# with open('autoencoder_representations_' + name + '_nobatchnosplit1e4_' + str(seed) + '_' + str(epoch) + '.json', 'w') as f:
#     json.dump(autoencoder_reps, f)

# top 5 img mem
top_5_imgs = sorted(zip(losses_all, imgs_all))[:5]
print('Bottom 5 images with lowest reconstruction loss:')
for loss, img in top_5_imgs:
    print(f'Image: {img}, Loss: {loss}')

# bottom 5 img mem
bottom_5_imgs = sorted(zip(losses_all, imgs_all))[-5:]
print('Top 5 images with highest reconstruction loss:')
for loss, img in bottom_5_imgs:
    print(f'Image: {img}, Loss: {loss}')
