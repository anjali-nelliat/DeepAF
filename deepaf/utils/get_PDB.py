#!/usr/bin/env python3

"""Extract PDB features from predicted complexes."""

import numpy as np
import pandas as pd
import json

PDBFS = {
    'ATOM': [0, 4],
    'atype': [13, 16],
    'resName': [17, 20],
    'CHAIN': 21,
    'resSeq': [22, 26],
    'X': [30, 38],
    'Y': [38, 46],
    'Z': [46, 54],
    'plddt': [61, 66],
    'ELEMENT': 77
}


def extractPDB(row, target):
    if target in ('ATOM', 'resName', 'atype'):
        return row[PDBFS[target][0]:PDBFS[target][1]].replace(' ', '')
    elif target in ('X', 'Y', 'Z', 'plddt'):
        return float(row[PDBFS[target][0]:PDBFS[target][1]])
    elif target in ('ELEMENT', 'CHAIN'):
        return row[PDBFS[target]]
    elif target in ('resSeq'):
        return int(row[PDBFS[target][0]:PDBFS[target][1]])


def readPDB(PDBfilename):
    file = open(PDBfilename, 'r', encoding="utf-8")
    PDBfile = [i.replace('\n', '') for i in file.readlines()]
    file.close()
    pdb = []
    for i in PDBfile:
        if extractPDB(i, 'ATOM') == 'ATOM':
            pdb.append({j: extractPDB(i, j) for j in PDBFS.keys()})
    return pdb


def readPDB2Pd(PDBfilename):
    file = open(PDBfilename, 'r', encoding="utf-8")
    PDBfile = [i.replace('\n', '') for i in file.readlines()]
    file.close()
    pdb = {i: [] for i in list(PDBFS.keys())[1:]}

    for i in PDBfile:
        if extractPDB(i, 'ATOM') == 'ATOM':
            for j in list(PDBFS.keys())[1:]:
                pdb[j].append(extractPDB(i, j))

    return pd.DataFrame(pdb)