#!/usr/bin/env python3

"""Generate the 3D tensor representations and csv list of predicted 
protein complexes """

import argparse
import os
import csv
import random

from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import pandas as pd

from deepaf.utils.get_interface import cal_interface_worker
from deepaf.utils.distance_encoding import encoding_worker
from deepaf.utils.dataset import save_dataset_to_csv


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    with open(args.dataset, 'r', encoding="utf-8") as f:
        csv_dict = csv.DictReader(f)
        dataset = [row for row in csv_dict]

    # prepare data
    pname = []
    labels = []
    pdb_paths = []
    for item in dataset:
        pdb_path = f"{args.data_dir}/{item['name']}/ranked_0.pdb"
        assert os.path.exists(pdb_path)

        labels.append(1 if item['type'] == 'positive' else 0)
        pname.append(item['name'])
        pdb_paths.append(pdb_path)


    # calculate interface
    interface_file = os.path.join(args.output_dir, 'interfaces.npy')
    if os.path.exists(interface_file) and not args.enforce_calculation:
        interfaces = np.load(interface_file, allow_pickle=True)
        print("interface file exists")
    else:
        print("calculate interface")
        with Pool(processes=args.threads) as pool:
            packed_data = [[pdb_paths[i],
                            args.interface_threshold,
                            args.interface_least_residue_n,
                            args.relaxed] for i in range(len(pname))]
            interfaces = list(tqdm(pool.imap(cal_interface_worker, packed_data), total=len(packed_data)))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        np.save(interface_file, interfaces)

    # distance encoding
    print("encoding tensors")
    with Pool(processes=args.threads) as pool:
        packed_data = []
        outpath = os.path.join(args.output_dir)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for i in range(len(pname)):
            packed_data.append([pdb_paths[i],
                                interfaces[i],
                                args.edge_length,
                                method,
                                True,
                                args.interface_threshold,
                                os.path.join(outpath, pname[i] + '.npy'),
                                args.relaxed])
        result = list(tqdm(pool.imap_unordered(encoding_worker, packed_data), total=len(packed_data)))

    save_dataset_to_csv(os.path.join(args.output_dir, 'complex_list.csv'), dataset)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./data/dataset.csv', help='CSV dataset')
    parser.add_argument('--data_dir', type=str, default='./data/afp_models', help='Prefix to the Alphapulldown prediction result folder')
    parser.add_argument('--output_dir', type=str, default='./data/output_dataset', help='Output dir to save the tensors')
    parser.add_argument('--interface_threshold', type=float, default=12.0, help='Distance threshold to define two residue contact')
    parser.add_argument('--interface_least_residue_n', type=int, default=12, help='Increase the distance threshold to ensure at least n residue included in the nearst area')
    parser.add_argument('--enforce_calculation', action='store_true', help='Enforce the calculation even if the file exists')
    parser.add_argument('--relaxed', action='store_true', help='Use relaxed models')
    parser.add_argument('--edge_length', type=int, default=64, help='Edge length of the tensor')
    parser.add_argument('--threads', type=int, default=12, help='Number of threads running')
    parser.add_argument('--seed', default=2032, type=int, help='Random seeds')
    args, unknown = parser.parse_known_args()
    main(args)
