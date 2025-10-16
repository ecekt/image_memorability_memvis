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

reps_name = ['SigLIP2'] #  ['CLIP', [

for name in reps_name:
    # set seed
    seeds = [0] # , 42, 2025, 2, 17]
    epochs_sigs = defaultdict(list)

    for seed in seeds:
        print(f'Autoencoder training using {name} representations with seed {seed}...')

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


        # simple autoencoder model
        class AutoEncoderModel(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(AutoEncoderModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, input_size)
                self.dropout = nn.Dropout(0.1)

            def forward(self, inps):
                #inps = self.dropout(inps)
                inps_encoded = self.fc1(inps)
                inps_decoded = F.relu(inps_encoded) # todo batchtopk
                inps_decoded = self.fc2(inps_decoded)

                return inps_decoded, inps_encoded


        # create training data
        class ImageRepsDataset(torch.utils.data.Dataset):
            def __init__(self, image_data):
                self.data = image_data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                input_rep = self.data[idx]
                output_rep = self.data[idx]
                return input_rep, output_rep


        # image_representations = torch.load('memcat_coco_CLIP_image_representations.pt', weights_only=False)
        # image_representations = torch.load('memcat_all_siglip2_image_representations.pt', weights_only=False)
        # image_representations = torch.load('memcat_coco_DINOV2_image_representations.pt', weights_only=False)

        if name == 'CLIP':
            with open('memcat_CLIP_reps_dict_layer_LAST.json', 'r') as f:
                image_representations = json.load(f)
        elif name == 'SigLIP2':
            with open('memcat_SIGLIP2_reps_dict_layer_GETIMGS.json', 'r') as f:
                image_representations = json.load(f)

        image_data = []
        for img in image_representations:
            image_data.append(torch.tensor(image_representations[img], dtype=torch.float32))

        mean = torch.stack(image_data).mean(dim=0)
        std = torch.stack(image_data).std(dim=0)
        image_data = [(x - mean) / (std + 1e-8) for x in image_data]

        # shuffle image data
        random.shuffle(image_data)
        train_data = image_data #[:int(len(image_data) * 0.8)]
        #val_data = image_data[int(len(image_data) * 0.8):]

        debug = False

        if debug:
            training_dataset = ImageRepsDataset(train_data[:100])
        else:
            training_dataset = ImageRepsDataset(train_data)

        #val_dataset = ImageRepsDataset(val_data)

        train_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True)

        # if debug:
        #     val_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=4, shuffle=False)
        # else:
        #     val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

        print(str(len(train_dataloader))) # + ' ' + str(len(val_dataloader)))

        model = AutoEncoderModel(input_size=image_data[0].shape[1], hidden_size=100)
        # train the model
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        model = model.to(device)
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train()
        num_epochs = 20
        weight = 1e-5 # sparsity weight
        best_loss = math.inf
        best_correlation = -math.inf

        for epoch in range(num_epochs):
            print("Epoch " + str(epoch))
            model.train()
            for batch in train_dataloader:
                # get the inputs
                input_batch, output_batch = batch
                input_seq = input_batch.to(device)
                label = output_batch.to(device)

                # forward pass
                outputs_dec, outputs_enc = model(input_seq)
                loss = criterion(outputs_dec, label) + weight * outputs_enc.abs().sum() # for sparsity

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses = 0.0

            with torch.no_grad():
                model.eval()

                memcat_all = []
                ae_losses = []

                with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        stripped_img_id = row[1].split('.')[0]
                        memcat_all.append(float(row[-1]))

                        image_rep = image_representations[stripped_img_id]
                        image_rep = torch.tensor(image_rep, dtype=torch.float32).to(device)
                        outputs_dec, outputs_enc = model(image_rep)
                        loss = (criterion(outputs_dec, image_rep)).item() # only reconstruction loss

                        ae_losses.append(loss)

                print('correlation between human memorability and SPARSE autoencoder loss (all data)')
                corr = spearmanr(memcat_all, ae_losses)
                print(len(memcat_all), len(ae_losses))
                print(corr)
                print()
                epochs_sigs[epoch].append(corr.statistic)
                
                # model is small, save every epoch
                torch.save(model.state_dict(), 'SPARSE_autoencoder_model_FULLmemcat_nobatchnosplit1e4_' + name + '_' + str(seed) + '_' + str(epoch) +  '.pt')
                
    # save stats for this model
    with open('autoencoder_loss_correlation_stats_' + name + '_nobatchNOSPLIT_20epochs_seed0.json', 'w') as f:
        json.dump(epochs_sigs, f)
    	

   
