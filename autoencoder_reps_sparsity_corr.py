import json
from scipy.stats import spearmanr
import csv

memcat = []
sparsities = []

with open('autoencoder_reps_sparsity_SigLIP2_nobatchnosplit1e4_0_4.json', 'r') as f:
    autoenc_sparsities = json.load(f)


with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        img_id = row[1]
        memcat.append(float(row[-1]))
        sparsities.append(autoenc_sparsities[img_id.split('.')[0]])

# corr
corr = spearmanr(memcat, sparsities)
print('correlation between human memorability and SPARSE autoencoder vector sum')
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')