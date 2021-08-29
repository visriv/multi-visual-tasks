"""
Split retail training data a into train/val subset
"""
import json
from pathlib import Path


subset = 'a'
data_dir = Path('data/train/')
gt_file = data_dir /  '{}_annotations.json'.format(subset)

with open(str(gt_file), 'r') as f:
    gts = json.load(f)

imgs = gts['images']
annotations = gts['annotations']
categories = gts['categories']

num_train = int(len(imgs) * 0.8)

imgs_train = imgs[:num_train]
imgs_val = imgs[num_train:]
print('Number of training images = ', len(imgs_train))
print('Number of validation images = ', len(imgs_val))

save_dict = {}
save_dict['annotations'] = annotations
save_dict['categories'] = categories

save_dict['images'] = imgs_train
gt_train_file = data_dir /  '{}_train_annotations.json'.format(subset)
with open(str(gt_train_file), 'w') as f:
    json.dump(save_dict, f)
print('Saved at ', str(gt_train_file))

save_dict['images'] = imgs_val
gt_val_file = data_dir /  '{}_val_annotations.json'.format(subset)
with open(str(gt_val_file), 'w') as f:
    json.dump(save_dict, f)
print('Saved at ', str(gt_val_file))
