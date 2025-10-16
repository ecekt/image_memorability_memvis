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
autoencoder_reps_sparsity = defaultdict(list)

for img in image_representations:
    rep = torch.tensor(image_representations[img], dtype=torch.float32)
    rep = (rep - mean) / (std + 1e-8)  # normalize

    outputs_dec, outputs_enc = model(rep)
    loss = (criterion(outputs_dec, rep)).item()  # only reconstruction loss

    sparsity = outputs_enc.abs().sum().item()  # sparsity of the encoded representation
    sparsity = sparsity / outputs_enc.numel()  # normalize by number of elements
    autoencoder_reps_sparsity[img] = sparsity
    #
    # # latents in autoencoder
    # autoencoder_reps[img] = outputs_enc.squeeze(0).detach().tolist()

    if img == '000000543264':
        print('Most memorable image sum:', sparsity)
    elif img == 'sun_acgqrysmhvkbzwyo':
        print('Least memorable image sum:', sparsity)

# Most memorable image reconstruction loss: 287.64935302734375
# Least memorable image reconstruction loss: 139.25709533691406

# save autoencoder representations
with open('autoencoder_reps_sparsity_' + name + '_nobatchnosplit1e4_' + str(seed) + '_' + str(epoch) + '.json', 'w') as f:
    json.dump(autoencoder_reps_sparsity, f)

# clip no batch no split 1e4
# Most memorable image sum: 136.173828125
# Least memorable image sum: 150.51632690429688

# clip
# Most memorable image sum: 129.18870544433594
# Least memorable image sum: 116.203369140625

# siglip2
# Most memorable image sum: 136.173828125
# Least memorable image sum: 150.51632690429688