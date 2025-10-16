import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

memcat_all = defaultdict(float)

with open('data/memcat/MemCat_data/memcat_image_data.csv', mode='r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        img_id = row[1]
        cat = row[2]
        obj = row[3]
        path = 'data/memcat/MemCat_images/MemCat/' + cat + '/' + obj + '/' + img_id
        memcat_all[path] = float(row[-1])

# top 10 most memorable images
# plot images in a grid
sorted_memcat_all = sorted(memcat_all.items(), key=lambda x: x[1], reverse=True)

print('Top 10 most memorable images:')
plt.figure(figsize=(10, 10))
for i, (img_id, score) in enumerate(sorted_memcat_all[:10]):
    print(f'Image ID: {img_id}, Score: {score}')
    img = plt.imread(img_id)
    plt.subplot(5, 2, i + 1)
    plt.imshow(img)
    plt.title(f'{score:.2f}')
    plt.axis('off')
    plt.tight_layout()
plt.savefig('top_10_memorable_images.png')
plt.show()
plt.close()

# top 10 least memorable images
print('Top 10 least memorable images:')
plt.figure(figsize=(10, 10))
for i, (img_id, score) in enumerate(sorted_memcat_all[-10:]):
    print(f'Image ID: {img_id}, Score: {score}')
    img = plt.imread(img_id)
    plt.subplot(5, 2, i + 1)
    plt.imshow(img)
    plt.title(f'{score:.2f}')
    plt.axis('off')
    plt.tight_layout()
plt.savefig('top_10_least_memorable_images.png')
plt.show()
plt.close()
