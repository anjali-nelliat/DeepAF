#!/usr/bin/env python3

"""Generate an edge transition matrix from a heterogenous network using randowm walk along edges based
on for input to the edge2vec algorithm"""

import argparse
import networkx as nx
import random
import numpy as np
import math

from deepaf.utils.get_matrix import simulate_walks, update_trans_matrix


def main(args):
    trans_matrix = initialize_edge_type_matrix(args.type_size)
    edge_graph = read_graph(args.input, args.weighted, args.directed)

    for i in range(args.em_iteration):
        walks = simulate_walks(edge_graph, args.num_walks, args.walk_length, trans_matrix, args.directed, args.p,
                               args.q)  # M step
        print(str(i), "th iteration for Upating transition matrix!")
        trans_matrix = update_trans_matrix(walks, args.type_size, args.e_step)  # E step

    np.savetxt(args.output, trans_matrix)

if __name__ == "__main__":
    """Parses the transition matrix arguments."""
    parser = argparse.ArgumentParser(description="Run edge transition matrix.")
    parser.add_argument('--input', nargs='?', default='data.txt', help='Input data path')
    parser.add_argument('--output', nargs='?', default='matrix.txt', help='store transition matrix')
    parser.add_argument('--type_size', type=int, default=6, help='Number of edge types. Default is 6.')
    parser.add_argument('--em_iteration', default=5, type=int, help='EM iterations for transition matrix')
    parser.add_argument('--e_step', default=3, type=int, help='E step in the EM algorithm: there are four expectation metrics')
    parser.add_argument('--dimensions', type=int, default=50, help='Number of dimensions. Default is 50.')
    parser.add_argument('--walk-length', type=int, default=10, help='Length of walk per source. Default is 10.')
    parser.add_argument('--num-walks', type=int, default=30, help='Number of walks per source. Default is 30.')
    parser.add_argument('--window-size', type=int, default=10, help='Context size for optimization. Default is 10.')
    parser.add_argument('--iter', default=50, type=int, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=0.25, help='Return hyperparameter. Default is 0.25.')
    parser.add_argument('--q', type=float, default=0.25, help='Inout hyperparameter. Default is 0.25.')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)
    args, unknown = parser.parse_known_args()
    main(args)