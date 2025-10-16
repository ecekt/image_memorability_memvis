import json
import numpy as np
import csv
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

with open('dict_memcat_ALL_vitmae_BASE_losses.json') as f:
    dict_vm_base_losses = json.load(f)

with open('dict_memcat_ALL_vitmae_LARGE_losses.json') as f:
    dict_vm_large_losses = json.load(f)

memcat = []
vm_losses = []
vm_large_losses = []

memcat_coco = []
vm_losses_coco = []
vm_large_losses_coco = []

memcat_imagenet = []
vm_losses_imagenet = []
vm_large_losses_imagenet = []

with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        img_id = row[1]
        memcat.append(float(row[-1]))
        vm_losses.append(dict_vm_base_losses[str(img_id)])
        vm_large_losses.append(dict_vm_large_losses[str(img_id)])

        if row[6] == 'coco':
            memcat_coco.append(float(row[-1]))
            vm_losses_coco.append(dict_vm_base_losses[str(img_id)])
            vm_large_losses_coco.append(dict_vm_large_losses[str(img_id)])

        if row[6] != 'imagenet':
            memcat_imagenet.append(float(row[-1]))
            vm_losses_imagenet.append(dict_vm_base_losses[str(img_id)])
            vm_large_losses_imagenet.append(dict_vm_large_losses[str(img_id)])

print(len(memcat))
print(len(vm_losses), len(vm_large_losses))
print(len(memcat_coco))
print(len(vm_losses_coco), len(memcat_coco))

# ALL MEMCAT
print('ALL MEMCAT IMAGES')
print('correlation between human memorability and autoencoder loss')
corr = spearmanr(memcat, vm_losses)
#print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()

print('correlation between human memorability and autoencoder loss (LARGE)')
corr = spearmanr(memcat, vm_large_losses)
#print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()

print('correlation between autoencoder loss and autoencoder loss (LARGE)')
corr = spearmanr(vm_losses, vm_large_losses)
#print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')

print()


# COCO ONLY
print('COCO ONLY')
print('correlation between human memorability and autoencoder loss')
corr = spearmanr(memcat_coco, vm_losses_coco)
#print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()

print('correlation between human memorability and autoencoder loss (LARGE)')
corr = spearmanr(memcat_coco, vm_large_losses_coco)
#print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()

print('correlation between autoencoder loss and autoencoder loss (LARGE)')
corr = spearmanr(vm_losses_coco, vm_large_losses_coco)
# print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')

print()

# NOT-IMAGENET
print('NOT-IMAGENET')
print('correlation between human memorability and autoencoder loss')
corr = spearmanr(memcat_imagenet, vm_losses_imagenet)
#print(corr)

# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()
print('correlation between human memorability and autoencoder loss (LARGE)')
corr = spearmanr(memcat_imagenet, vm_large_losses_imagenet)
#print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()

print('correlation between autoencoder loss and autoencoder loss (LARGE)')
corr = spearmanr(vm_losses_imagenet, vm_large_losses_imagenet)
# print(corr)
# print corr nicely
print(f'{corr[0]:.4f}\t{corr[1]:.4f}')
print()

print(len(memcat_coco), len(memcat_imagenet))