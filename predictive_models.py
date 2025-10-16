# predict human memorability from model-internal representations and statistics
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm, glm
import pandas as pd
import json
import csv
import numpy as np


def fit_generalized_linear_model(data, formula):
    model = glm(formula, data, family=sm.families.Gaussian())
    result = model.fit()
    return result


mems = []
groups = []
with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        mem = float(row[-1])
        mems.append(mem)
        cat = row[2]
        groups.append(cat)

# encode categories as a categorical variable
groups_id = {cat: idx for idx, cat in enumerate(set(groups))}
groups = [groups_id[cat] for cat in groups]

# load the model internal signals
model_types = ['clip', 'dinov2', 'siglip2']
mode_strs = {'clip': 'CLIP', 'dinov2': 'DINOv2', 'siglip2': 'SigLIP2'}

for model_type in model_types:
    model_str = mode_strs[model_type]
    print(f'Processing {model_str} model...')

    # 13 layers, but the first one is the same as we are looking at CLS
    with open(model_type + '_activations_mean.json', 'r') as f:
        activations_mean = json.load(f)
    with open(model_type + '_activations_max.json', 'r') as f:
        activations_max = json.load(f)
    with open(model_type + '_activations_max_abs.json', 'r') as f:
        activations_max_abs = json.load(f)

    # 13 layers
    with open(model_type + '_patch_regularity.json', 'r') as f:
        patch_regularity = json.load(f)

    # 12 layers
    with open(model_type + '_attentions_entropy.json', 'r') as f:
        attentions_entropy = json.load(f)

    # just the last layer
    # activation_max_data = activations_max[str(len(activations_max) - 1)]
    # activation_mean_data = activations_mean[str(len(activations_mean) - 1)]
    # activation_max_abs_data = activations_max_abs[str(len(activations_max_abs) - 1)]
    # patch_regularity_data = patch_regularity[str(len(patch_regularity) - 1)]
    # attentions_entropy_data = attentions_entropy[str(len(attentions_entropy) - 1)]

    # just the first layer
    ## but skip activations at layer 0, use 1st instead (as it is the same for CLS)
    # activation_max_data = activations_max['1']
    # activation_mean_data = activations_mean['1']
    # activation_max_abs_data = activations_max_abs['1']
    # patch_regularity_data = patch_regularity['1']
    # attentions_entropy_data = attentions_entropy['1']

    # layer 10
    activation_max_data = activations_max['10']
    activation_mean_data = activations_mean['10']
    activation_max_abs_data = activations_max_abs['10']
    patch_regularity_data = patch_regularity['10']
    attentions_entropy_data = attentions_entropy['10']

    # # layer 8
    # activation_max_data = activations_max['8']
    # activation_mean_data = activations_mean['8']
    # activation_max_abs_data = activations_max_abs['8']
    # patch_regularity_data = patch_regularity['8']
    # attentions_entropy_data = attentions_entropy['8']

    # # normalize the data (z-score normalization)
    activation_max_data = (activation_max_data - np.mean(activation_max_data)) / np.std(activation_max_data)
    activation_mean_data = (activation_mean_data - np.mean(activation_mean_data)) / np.std(activation_mean_data)
    activation_max_abs_data = (activation_max_abs_data - np.mean(activation_max_abs_data)) / np.std(activation_max_abs_data)
    patch_regularity_data = (patch_regularity_data - np.mean(patch_regularity_data)) / np.std(patch_regularity_data)
    attentions_entropy_data = (attentions_entropy_data - np.mean(attentions_entropy_data)) / np.std(attentions_entropy_data)
    #
    # # # normalize the data (min-max normalization)
    # activation_max_data = (activation_max_data - np.min(activation_max_data)) / (np.max(activation_max_data) - np.min(activation_max_data))
    # activation_mean_data = (activation_mean_data - np.min(activation_mean_data)) / (np.max(activation_mean_data) - np.min(activation_mean_data))
    # activation_max_abs_data = (activation_max_abs_data - np.min(activation_max_abs_data)) / (np.max(activation_max_abs_data) - np.min(activation_max_abs_data))
    # patch_regularity_data = (patch_regularity_data - np.min(patch_regularity_data)) / (np.max(patch_regularity_data) - np.min(patch_regularity_data))
    # attentions_entropy_data = (attentions_entropy_data - np.min(attentions_entropy_data)) / (np.max(attentions_entropy_data) - np.min(attentions_entropy_data))


    # create a DataFrame
    df = pd.DataFrame({
        'activation_max': activation_max_data,
        'activation_mean': activation_mean_data,
        'activation_max_abs': activation_max_abs_data,
        'patch_regularity': patch_regularity_data,
        'attentions_entropy': attentions_entropy_data,
        'group': groups,
        'mem_score': mems
    })

    # Fit generalized linear model
    glm_formula = 'mem_score ~ activation_max + activation_mean + activation_max_abs + patch_regularity + attentions_entropy'
    glm_result = fit_generalized_linear_model(df, glm_formula)
    print(f'Generalized Linear Model Results for {model_str}:')
    print(glm_result.summary())
    print("\n")

    # df.to_csv(f'{model_type}_data_layer10_normalized_layer.csv', index=False)
    # print(f'Data saved to {model_type}_data_layer10_normalized_layer.csv\n')
