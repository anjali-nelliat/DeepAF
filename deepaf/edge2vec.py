#!/usr/bin/env python3

"""Run edge2vec on the input transition matrix to obtain embeddings"""

import argparse
import networkx as nx
import random
import numpy as np
import math
from scipy import stats
from scipy import spatial
from gensim.models import Word2Vec

from deepaf.utils.embed import read_edge_type_matrix, read_graph, simulate_edge2vecwalks

def main(args):
    trans_matrix = read_edge_type_matrix(args.matrix)
    edge_graph = read_graph(args.input, args.weighted, args.directed)
    walks = simulate_edge2vecwalks(edge_graph, args.num_walks, args.walk_length, trans_matrix, args.directed, args.p, args.q)
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.wv.save_word2vec_format(args.output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run edge2vec.")
    parser.add_argument('--input', nargs='?', default='weighted_graph.txt', help='Input data file')
    parser.add_argument('--matrix', nargs='?', default='matrix.txt', help='Input transition matrix file')
    parser.add_argument('--output', nargs='?', default='embeddings.txt', help='Embeddings path')
    parser.add_argument('--dimensions', type=int, default=128, help='Number of dimensions. Default is 128.')
    parser.add_argument('--walk-length', type=int, default=10, help='Length of walk per source. Default is 10.')
    parser.add_argument('--num-walks', type=int, default=30, help='Number of walks per source. Default is 30.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 2.')
    parser.add_argument('--iter', default=50, type=int, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=0.25, help='Return hyperparameter. Default is 0.25.')
    parser.add_argument('--q', type=float, default=0.25, help='Inout hyperparameter. Default is 0.25.')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true', help='edge_graphraph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)
    args, unknown = parser.parse_known_args()
    main(args)