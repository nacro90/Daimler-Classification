import json
import os
from random import shuffle

dataset_folder = 'processed_dataset/'

train_pile = []
test_pile = []

positives = os.listdir(dataset_folder + 'positive/')
negatives = os.listdir(dataset_folder + 'negative/')

divider = int((len(positives) + len(negatives)) * 0.8)

for i, name in enumerate(positives):
    row = {
        'image': dataset_folder + 'positive/' + name,
        'label': 1
    }
    if i < divider / 2:
        train_pile.append(row)
    else:
        test_pile.append(row)

for i, name in enumerate(negatives):
    row = {
        'image': dataset_folder + 'negative/' + name,
        'label': 0
    }
    if i < divider / 2:
        train_pile.append(row)
    else:
        test_pile.append(row)

shuffle(train_pile)
shuffle(test_pile)

train_json = json.dumps(train_pile, indent=4, sort_keys=True)
test_json = json.dumps(test_pile, indent=4, sort_keys=True)

with open(dataset_folder + 'train.json', 'w') as json_file:
    json_file.write(train_json)

with open(dataset_folder + 'test.json', 'w') as json_file:
    json_file.write(test_json)
