#!/usr/bin/env python3

"""Utility scripts to generate embeddings from a transition matrix using edge2vec"""

import argparse
import networkx as nx
import random
import numpy as np
import math
from scipy import stats
from scipy import spatial

def read_graph(edgeList, weighted=False, directed=False):
    """Reads the input network in networkx."""
    if weighted:
        edge_graph = nx.read_edgelist(edgeList, nodetype=str, data=(('type', int), ('weight', float), ('id', int)),
                             create_using=nx.Diedge_graphraph())
    else:
        edge_graph = nx.read_edgelist(edgeList, nodetype=str, data=(('type', int), ('id', int)), create_using=nx.Diedge_graphraph())
        for edge in edge_graph.edges():
            edge_graph[edge[0]][edge[1]]['weight'] = 1.0

    if not directed:
        edge_graph = edge_graph.to_undirected()

    return edge_graph


def read_edge_type_matrix(file):
    """load transition matrix"""
    matrix = np.loadtxt(file, delimiter=' ')
    return matrix


def simulate_edge2vecwalks(edge_graph, num_walks, walk_length, matrix, is_directed, p, q):
    """generate random walk paths constrained by transition matrix"""
    walks = []
    nodes = list(edge_graph.nodes())
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter + 1), '/', str(num_walks))
        random.shuffle(nodes)
        for node in nodes:
            walks.append(edge2vec(edge_graph, walk_length, node, matrix, is_directed, p, q))
    return walks


def edge2vec(edge_graph, walk_length, start_node, matrix, is_directed, p, q):
    """return a random walk path"""
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        cur_nbrs = sorted(edge_graph.neighbors(cur))
        random.shuffle(cur_nbrs)
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                rand = int(np.random.rand() * len(cur_nbrs))
                next_node = cur_nbrs[rand]
                walk.append(next_node)
            else:
                prev = walk[-2]
                pre_edge_type = edge_graph[prev][cur]['type']
                distance_sum = 0
                for neighbor in cur_nbrs:
                    neighbor_link = edge_graph[cur][neighbor]
                    neighbor_link_type = neighbor_link['type']
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type - 1][neighbor_link_type - 1]

                    if edge_graph.has_edge(neighbor, prev) or edge_graph.has_edge(prev, neighbor):  # undirected graph

                        distance_sum += trans_weight * neighbor_link_weight / p  # +1 normalization
                    elif neighbor == prev:  # decide whether it can random walk back
                        distance_sum += trans_weight * neighbor_link_weight
                    else:
                        distance_sum += trans_weight * neighbor_link_weight / q

                rand = np.random.rand() * distance_sum
                threshold = 0
                for neighbor in cur_nbrs:
                    neighbor_link = edge_graph[cur][neighbor]
                    neighbor_link_type = neighbor_link['type']
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type - 1][neighbor_link_type - 1]

                    if edge_graph.has_edge(neighbor, prev) or edge_graph.has_edge(prev, neighbor):  # undirected graph

                        threshold += trans_weight * neighbor_link_weight / p
                        if threshold >= rand:
                            next_node = neighbor
                            break
                    elif neighbor == prev:
                        threshold += trans_weight * neighbor_link_weight
                        if threshold >= rand:
                            next_node = neighbor
                            break
                    else:
                        threshold += trans_weight * neighbor_link_weight / q
                        if threshold >= rand:
                            next_node = neighbor
                            break
                walk.append(next_node)
        else:
            break  # if the node only has 1 neighbour

    return walk