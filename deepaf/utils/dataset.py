#!/usr/bin/env python3

import os
import csv
import numpy as np

from deepaf.utils.data_augment import rotate3D, relocate


def generate_train(dataset, batch_size, augment=False):
    while 1:
        for i in range(0, len(dataset), batch_size):
            ds = dataset[i:i + batch_size]
            x = []
            y = []
            for d in ds:
                data = np.load(d[0])
                if augment:
                    data = rotate3D(data)
                    data = relocate(data)

                label = [0, 1] if int(d[1]) == 1 else [1, 0]

                x.append(data)
                y.append(label)

            x = np.array(x)
            y = np.array(y)
            yield (x, y)

def generate_query(dataset, batch_size, augment=False):
    while 1:
        for i in range(0, len(dataset), batch_size):
            ds = dataset[i:i + batch_size]
            x = []
            for d in ds:
                data = np.load(d[0])
                if augment:
                    data = rotate3D(data)
                    data = relocate(data)

                x.append(data)

            x = np.array(x)
            yield [x, )
            


def get_dataset_from_csv(csv_path, data_dir):
    data = []
    with open(csv_path) as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            x = os.path.join(data_dir, row[0])
            y = row[1]
            data.append([x, y])
    return data


def save_dataset_to_csv(csv_path, dataset):
    contents = []
    for item in dataset:
        label = 1 if item['type'] == 'positive' else 0
        name = item['name'] + '.npy'
        contents.append([name, label])

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(contents)
