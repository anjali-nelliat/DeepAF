#!/usr/bin/env python3

""" Run model on tensors calculated from AlphaPulldown-predicted complexes"""

import math
import os
import random
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from utils.densenet_model import get_model
from utils.dataset import gen, get_dataset_from_csv


print(tf.test.gpu_device_name())
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print('---------------loading dataset---------------')
    input_size = [args.alength, args.alength, args.alength, args.ndims]
    query_data = get_dataset_from_csv(args.query_data, args.datapath)

    print('---------------begin query---------------')

    model = get_model(args.model, input_size)

    print("Trained weights from", args.weights)
    model.load_weights(args.weights)
    # model.summary()

    pred = model.predict_generator(generator=gen(query_data, args.batch, augment=False),
                                     steps=math.ceil(len(query_data) / args.batch),
                                      workers=1,
                                      verbose=1)

    pred = np.array(pred)
    np.save(args.output, pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='./data/sample_dataset/tensors', help='Path to tensors')
    parser.add_argument('--weights', type=str, default='./models/sample_model/test/best_model.hdf5')
    parser.add_argument('--output', type=str, default='./models/sample_model/test/preds.npy')
    parser.add_argument('--query_data', type=str, default='./data/sample_dataset/query_data.csv')
    parser.add_argument('--batch', default=32)
    parser.add_argument('--alength', default=64)
    parser.add_argument('--ndims', default=8)
    parser.add_argument('--seed', default=2032)
    args, unknown = parser.parse_known_args()

    main(args)