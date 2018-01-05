import json
import os
from random import shuffle

import numpy as np
from PIL import Image

POSITIVE_ROOT = "positive/"
NEGATIVE_ROOT = "negative/"

DATASET_FOLDER = "processed_dataset/"

TRAINING_JSON = "train.json"
TEST_JSON = "test.json"


def training_batch_generator(batch_size):
    training_data = json.load(open(DATASET_FOLDER + TRAINING_JSON))

    for batch_index in range(int(len(training_data) / batch_size)):

        image_list = []
        labels = []

        for training_row in training_data[batch_index * batch_size:(batch_index + 1) * batch_size]:
            im = Image.open(training_row['image'])
            width, height = im.size
            
            image_np = np.array(im.getdata(), dtype=np.float32)
            image_np = image_np.reshape(width, height, 1)

            image_list.append(image_np)

            labels.append([training_row['label']])

        yield batch_index, np.array(image_list), np.array(labels)


def get_training_length():
    return len(json.load(open(DATASET_FOLDER + TRAINING_JSON)))


def test_batch_generator(batch_size):
    test_data = json.load(open(DATASET_FOLDER + TEST_JSON))

    for batch_index in range(len(test_data) // batch_size):

        image_list = []
        labels = []

        for test_row in test_data[batch_index * batch_size: (batch_index + 1) * batch_size]:
            im = Image.open(test_row['image'])
            width, height = im.size

            image_np = np.array(im.getdata(), dtype=np.float32)
            image_np = image_np.reshape(width, height, 1)
            
            image_list.append(image_np)

            labels.append([test_row['label']])

        yield batch_index, np.array(image_list), np.array(labels)


def main():
    for batch_index, images, labels in training_batch_generator(1):
        print(images)
        print(labels)
        if batch_index == 1:
            break


if __name__ == '__main__':
    main()
