#!/usr/bin/env python3

"""Encode atomic coordinates in the 3D tensor representation of the protein complex
   using Euclidian distance from the center of the space to the atom """

import math
import random
import numpy as np
import os

from tqdm import tqdm
from math import ceil, floor, sqrt
from deepaf.utils.get_PDB import readPDB, readPDB2Pd


def resize_data(xyz, content, edge_length, relaxed=True):
    if relaxed:
        chainAchar = 'A'
        chainBchar = 'B'
    else:
        chainAchar = 'B'
        chainBchar = 'C'
    compete = None
    x = floor(xyz.min(axis=0)[0])
    # for i in tqdm(range(floor(xyz.min(axis=0)[0]), floor(xyz.max(axis=0)[0]))):
    while True:
        y = floor(xyz.min(axis=0)[1])
        while True:
            z = floor(xyz.min(axis=0)[2])
            while True:
                leagel_x = (xyz[:, 0] > x - 0.5) & (xyz[:, 0] < x + edge_length - 0.5)
                leagel_y = (xyz[:, 1] > y - 0.5) & (xyz[:, 1] < y + edge_length - 0.5)
                leagel_z = (xyz[:, 2] > z - 0.5) & (xyz[:, 2] < z + edge_length - 0.5)
                leagel = leagel_x & leagel_y & leagel_z
                nowt = [np.count_nonzero(leagel), leagel, x, y, z]
                if compete is None:
                    compete = nowt
                elif nowt[0] > compete[0]:
                    compete = nowt
                elif nowt[0] == compete[0]:
                    dx = np.count_nonzero(content[leagel][:, 0] == chainAchar) - np.count_nonzero(content[leagel][:, 0] == chainBchar)
                    dy = np.count_nonzero(content[compete[1]][:, 0] == chainAchar) - np.count_nonzero(content[compete[1]][:, 0] == chainBchar)
                    if abs(dx) < abs(dy):
                        compete = nowt
                # print(np.count_nonzero(leagel), x, y, z)
                if z + edge_length - 1 >= xyz.max(axis=0)[2]:
                    break
                else:
                    z += 1
            if y + edge_length - 1 >= xyz.max(axis=0)[1]:
                break
            else:
                y += 1
        if x + edge_length - 1 >= xyz.max(axis=0)[0]:
            break
        else:
            x += 1

    return xyz[compete[1]], content[compete[1]]


def coord_transform(PDB_path, edge_length, interface, ignore_hydrogen=False, relaxed=True):
    if relaxed:
        chainAchar = 'A'
        chainBchar = 'B'
    else:
        chainAchar = 'B'
        chainBchar = 'C'
    PDB_content = readPDB(PDB_path)
    content = []
    xyz = []
    pdb = readPDB2Pd(PDB_path)
    if len(interface[0]) != pdb[(pdb['CHAIN'] == chainAchar)]['resSeq'].max() or len(interface[1]) != pdb[(pdb['CHAIN'] == chainBchar)]['resSeq'].max():
        print(len(interface[0]), len(interface[1]))
        print(pdb[(pdb['CHAIN'] == chainAchar)]['resSeq'].max())
        print(pdb[(pdb['CHAIN'] == chainBchar)]['resSeq'].max())
        print(chainAchar, chainBchar)
        print(PDB_path)
    for item in PDB_content:
        if interface[0 if item['CHAIN'] == chainAchar else 1][item['resSeq'] - 1]:
            if ignore_hydrogen and item['ELEMENT'] == 'H':
                continue
            else:
                xyz.append(np.array([item['X'], item['Y'], item['Z']]))
                content.append([item['CHAIN'], item['ELEMENT']])

    xyz = np.array(xyz)
    content = np.array(content)
    center_size = xyz.max(axis=0) - xyz.min(axis=0)
    if center_size.max() > edge_length:
        xyz, content = resize_data(xyz, content, edge_length, relaxed)

    center_size = xyz.max(axis=0) - xyz.min(axis=0)
    if center_size.max() > edge_length:
        print(f"WARN: At job {PDB_path}, interface size {center_size} is larger than edge size {edge_length}")

    xyz -= (xyz.max(axis=0) + xyz.min(axis=0)) / 2
    xyz += int(edge_length / 2)
    l = (xyz.min(axis=1) > -0.5) & (xyz.max(axis=1) < edge_length - 0.5)
    # print(f"use {l.sum()} of {len(content)} for encoding")
    # print(f"interface size {center_size}, edge size {edge_length}")
    xyz = xyz[l]
    content = content[l]

    return xyz, content

def encoding(PDB_path,
             interface,
             edge_length=128,
             ignore_hydrogen=False,
             dis_thre=12,
             relaxed=True):

    if relaxed:
        chainAchar = 'A'
        chainBchar = 'B'
    else:
        chainAchar = 'B'
        chainBchar = 'C'

    unit_length = 8 if ignore_hydrogen else 10
    tensor = np.zeros((edge_length, edge_length, edge_length, unit_length), dtype=np.float16)
    if interface[0].sum() + interface[1].sum() == 0:
        return tensor
    coords, contents = coord_transform(PDB_path, edge_length, interface, ignore_hydrogen, relaxed)
    subscripts = np.round(coords, 0).astype(np.uint8)

    # print(len(contents))
    # print(interface[0].sum(), interface[1].sum())

    ELEMENTSD = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4}
    if ignore_hydrogen:
        ELEMENTSD = {'C': 0, 'N': 1, 'O': 2, 'S': 3}
    for i in range(contents.shape[0]):
        index = 0 if contents[i][0] == chainAchar else int(unit_length / 2)
        index += ELEMENTSD[contents[i][1]]

        for x in range(edge_length):
            if abs(coords[i][0] - x) > dis_thre:
                continue
            for y in range(edge_length):
                if abs(coords[i][1] - y) > dis_thre:
                    continue
                if sqrt(pow(x - coords[i][0], 2) + pow(y - coords[i][1], 2)) > dis_thre:
                    continue
                for z in range(edge_length):
                    if abs(coords[i][2] - z) > dis_thre:
                        continue
                    distance = sqrt(pow(x - coords[i][0], 2) + pow(y - coords[i][1], 2) + pow(z - coords[i][2], 2))
                    if distance < dis_thre:
                        if tensor[x][y][z][index] == 0:
                            tensor[x][y][z][index] = distance
                        else:
                            tensor[x][y][z][index] = min(tensor[x][y][z][index], distance)

    return tensor


def encoding_worker(arg):
    tensor = encoding(
        PDB_path=arg[0],
        interface=arg[1],
        edge_length=arg[2],
        ignore_hydrogen=arg[4],
        dis_thre=arg[5],
        relaxed=arg[7]
    )
    np.save(arg[6], tensor)
    return 0
