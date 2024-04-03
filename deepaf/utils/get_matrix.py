#!/usr/bin/env python3

"""Utility scripts to generate an edge transition matrix from a heterogenous network using randowm walk along edges based
on for input to the edge2vec algorithm"""

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
from scipy import stats


def read_graph(edgeList, weighted=False, directed=False):
    """Reads the input network in networkx."""
    if weighted:
        e_graph = nx.read_edgelist(edgeList, nodetype=str, data=(('type', int), ('weight', float), ('id', int)),
                             create_using=nx.Die_graphraph())
    else:
        e_graph = nx.read_edgelist(edgeList, nodetype=str, data=(('type', int), ('id', int)), create_using=nx.Die_graphraph())
        for edge in e_graph.edges():
            e_graph[edge[0]][edge[1]]['weight'] = 1.0

    if not directed:
        e_graph = e_graph.to_undirected()
    return e_graph


def initialize_edge_type_matrix(type_num):
    """Initialize a transition matrix with equal values"""
    initialized_val = 1.0 / (type_num * type_num)
    matrix = [[initialized_val for i in range(type_num)] for j in range(type_num)]
    return matrix


def simulate_walks(e_graph, num_walks, walk_length, matrix, is_directed, p, q):
    """e_graphenerate random walk paths constrained by transition matrix"""
    walks = []
    links = list(e_graph.edges(data=True))
    print('Walk iteration:')
    for walk_iter in range(num_walks):
        print(str(walk_iter + 1), '/', str(num_walks))
        random.shuffle(links)
        count = 1000
        for link in links:
            walks.append(edge2vec_walk(e_graph, walk_length, link, matrix, is_directed, p, q))
            count = count - 1
            if count == 0 and len(links) > 1000:  # control the pairwise list length
                break
    return walks


def edge2vec_walk(e_graph, walk_length, start_link, matrix, is_directed, p, q):
    """return a random walk path"""
    walk = [start_link]
    result = [str(start_link[2]['type'])]
    while len(walk) < walk_length:  # here we may need to consider some dead end issues
        cur = walk[-1]
        start_node = cur[0]
        end_node = cur[1]
        cur_edge_type = cur[2]['type']

        """Consider the hub nodes and reduce the hub influence"""
        if is_directed:  # directed graph has random walk direction already
            direction_node = end_node
            left_node = start_node
        else:  # for undirected graph, first consider the random walk direction by choosing the start node
            start_direction = 1.0 / e_graph.degree(start_node)
            end_direction = 1.0 / e_graph.degree(end_node)
            prob = start_direction / (start_direction + end_direction)

            rand = np.random.rand()

            if prob >= rand:
                direction_node = start_node
                left_node = end_node
            else:
                direction_node = end_node
                left_node = start_node
        neighbors = e_graph.neighbors(direction_node)

       """Calculate sum of distances, with +1 normalization"""
        distance_sum = 0
        for neighbor in neighbors:
            neighbor_link = e_graph[direction_node][neighbor]  # get candidate link's type
            neighbor_link_type = neighbor_link['type']
            neighbor_link_weight = neighbor_link['weight']
            trans_weight = matrix[cur_edge_type - 1][neighbor_link_type - 1]
            if e_graph.has_edge(neighbor, left_node) or e_graph.has_edge(left_node, neighbor):
                distance_sum += trans_weight * neighbor_link_weight / p
            elif neighbor == left_node:  # decide if we can random walk back
                distance_sum += trans_weight * neighbor_link_weight
            else:
                distance_sum += trans_weight * neighbor_link_weight / q

        """Pick up the next step link"""
        rand = np.random.rand() * distance_sum
        threshold = 0
        neighbors2 = e_graph.neighbors(direction_node)
        for neighbor in neighbors2:
            neighbor_link = e_graph[direction_node][neighbor]  # get candidate link's type
            neighbor_link_type = neighbor_link['type']
            neighbor_link_weight = neighbor_link['weight']
            trans_weight = matrix[cur_edge_type - 1][neighbor_link_type - 1]
            if e_graph.has_edge(neighbor, left_node) or e_graph.has_edge(left_node, neighbor):
                threshold += trans_weight * neighbor_link_weight / p
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break
            elif neighbor == left_node:
                threshold += trans_weight * neighbor_link_weight
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break
            else:
                threshold += trans_weight * neighbor_link_weight / q
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break

        if distance_sum > 0:
            next_link = e_graph[direction_node][next_link_end_node]
            next_link_tuple = tuple()
            next_link_tuple += (direction_node,)
            next_link_tuple += (next_link_end_node,)
            next_link_tuple += (next_link,)
            walk.append(next_link_tuple)
            result.append(str(next_link_tuple[2]['type']))
        else:
            break
    return result


def update_trans_matrix(walks, type_size, evaluation_metric):
    """E step, update transition matrix"""
    #use list of list to store all edge type numbers and use KL divergence to update
    matrix = [[0 for i in range(type_size)] for j in range(type_size)]
    repo = dict()
    for i in range(type_size):  # initialize empty list to hold edge type vectors
        repo[i] = []

    for walk in walks:
        curr_repo = dict()  # store each type number in current walk
        for edge in walk:
            edge_id = int(edge) - 1
            if edge_id in curr_repo:
                curr_repo[edge_id] = curr_repo[edge_id] + 1
            else:
                curr_repo[edge_id] = 1

        for i in range(type_size):
            if i in curr_repo:
                repo[i].append(curr_repo[i])
            else:
                repo[i].append(0)

    for i in range(type_size):
        for j in range(type_size):
            if evaluation_metric == 1:
                sim_score = wilcoxon_test(repo[i], repo[j])
                matrix[i][j] = sim_score
            elif evaluation_metric == 2:
                sim_score = entroy_test(repo[i], repo[j])
                matrix[i][j] = sim_score
            elif evaluation_metric == 3:
                sim_score = spearmanr_test(repo[i], repo[j])
                matrix[i][j] = sim_score
            elif evaluation_metric == 4:
                sim_score = pearsonr_test(repo[i], repo[j])
                matrix[i][j] = sim_score
            else:
                raise ValueError('not correct evaluation metric! You need to choose from 1-4')
    return matrix


"""Different ways to calculate correlation between edge-types"""
# pairwised judgement
def wilcoxon_test(v1, v2):  # the smaller the more similar
    result = stats.wilcoxon(v1, v2).statistic
    return 1 / (math.sqrt(result) + 1)


def entroy_test(v1, v2):  # the smaller the more similar
    result = stats.entropy(v1, v2)
    return result


def spearmanr_test(v1, v2):  # the larger the more similar
    result = stats.mstats.spearmanr(v1, v2).correlation
    return sigmoid(result)


def pearsonr_test(v1, v2):  # the larger the more similar
    result = stats.mstats.pearsonr(v1, v2)[0]
    return sigmoid(result)


def cos_test(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def standardization(x):
    return (x + 1) / 2


def relu(x):
    return (abs(x) + x) / 2