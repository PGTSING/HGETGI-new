import numpy as np
import random
import time
from sklearn import metrics
import tqdm
import dgl
import sys
import os
import pandas as pd
import scipy.sparse as sp
import torch
from Metapath import Metapath2VecTrainer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import networkx as nx
from scipy import interp
import matplotlib.pyplot as plt

path = "new_data"

def construct_graph():
    TF_ids = []
    TF_names = []
    Target_ids = []
    Target_names = []
    Disease_ids = []
    Disease_names = []
    f_3 = open(os.path.join(path, "id_TF.txt"), encoding="gbk")
    f_4 = open(os.path.join(path, "id_Target.txt"), encoding="gbk")
    f_5 = open(os.path.join(path, "id_Disease.txt"), encoding="gbk")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        TF_ids.append(identity)
        TF_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break;
        w = w.strip().split()
        identity = int(w[0]) + len(TF_names)
        Target_ids.append(identity)
        Target_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break;
        v = v.strip().split()
        identity = int(v[0]) + len(TF_names) + len(Target_names)
        disease_name = v[1]
        Disease_ids.append(identity)
        Disease_names.append(disease_name)
    f_3.close()
    f_4.close()
    f_5.close()

    TF_ids_invmap = {x: i for i, x in enumerate(TF_ids)}
    Target_ids_invmap = {x: i for i, x in enumerate(Target_ids)}
    Disease_ids_invmap = {x: i for i, x in enumerate(Disease_ids)}

    print(Target_ids_invmap)
    print(Disease_ids_invmap)
    TF_Target_src = []
    TF_Target_dst = []
    Target_Disease_src = []
    Target_Disease_dst = []
    TF_Disease_src = []
    TF_Disease_dst = []
    f_1 = open(os.path.join(path, "TF_Target.txt"), "r")
    f_2 = open(os.path.join(path, "Target_Disease.txt"), "r")
    # f_2 = open(os.path.join(path, "New_Target_Disease.txt"), "r")
    f_0 = open(os.path.join(path, "TF_Disease.txt"), "r")

    matrix = [([0] * 8881) for i in range(8881)]
    for x in f_1:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n')) + len(TF_names)
        matrix[x1][x2] = 1
        matrix[x2][x1] = 1

    for x in f_0:
        x = x.strip().split()
        x1 = int(x[0])
        x2 = int(x[1].strip('\n')) + len(TF_names) + len(Target_names)
        matrix[x1][x2] = 1
        matrix[x2][x1] = 1

    for x in f_2:
        x = x.strip().split()
        x1 = int(x[0]) + len(TF_names)
        x2 = int(x[1].strip('\n')) + len(TF_names) + len(Target_names)
        matrix[x1][x2] = 1
        matrix[x2][x1] = 1
    sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                sum = sum+1
    print(sum)
    # print(len(matrix))
    # print(len(matrix[0]))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                if i < len(TF_names) and j >= len(TF_names) and j < len(TF_names)+len(Target_names):
                    TF_Target_src.append(TF_ids_invmap[i])
                    TF_Target_dst.append(Target_ids_invmap[j])
                elif i < len(TF_names) and j >= (len(TF_names) + len(Target_names)):
                    TF_Disease_src.append(TF_ids_invmap[i])
                    TF_Disease_dst.append(Disease_ids_invmap[j])
                elif i >= len(TF_names) and i < len(TF_names)+len(Target_names) and j >= (len(TF_names) + len(Target_names)):
                    Target_Disease_src.append(Target_ids_invmap[i])
                    Target_Disease_dst.append(Disease_ids_invmap[j])
    matrix = sp.csr_matrix(matrix)

    g = nx.from_scipy_sparse_matrix(matrix)

    print("连通分量数：",nx.number_connected_components(g))

    print(nx.is_connected(g))
    print(nx.number_of_isolates(g))
    print("孤立节点：",list(nx.isolates(g)))
    print(len(TF_names))
    print(len(Target_names))
    # for c in nx.connected_components(g):
    #     s = g.subgraph(c).copy()
    #     print(s.number_of_nodes())
    #     print(s.number_of_edges())
    #     for n in s.nodes():
    #         print(n)
    #     for e in s.edges():
    #         print(e)


    f_1.close()
    f_2.close()
    f_0.close()

    hg = dgl.heterograph({
        ('TF', 'zt', 'Target'): (TF_Target_src, TF_Target_dst),
        ('Target', 'tz', 'TF'): (TF_Target_dst, TF_Target_src),
        ('Target', 'td', 'Disease'): (Target_Disease_src, Target_Disease_dst),
        ('Disease', 'dt', 'Target'): (Target_Disease_dst, Target_Disease_src),
        ('TF', 'zd', 'Disease'): (TF_Disease_src, TF_Disease_dst),
        ('Disease', 'dz', 'TF'): (TF_Disease_dst, TF_Disease_src)})

    return hg, TF_names, Target_names, Disease_names,matrix

hg, TF_names, Target_names, Disease_names ,matraix = construct_graph()
print(hg)
print(hg.edges('all', etype = 'zt'))
print(hg.edges('all', etype = 'tz'))
print(hg.edges('all', etype = 'td'))
print(hg.edges('all', etype = 'dt'))
print(hg.edges('all', etype = 'zd'))
print(hg.edges('all', etype = 'dz'))
print(hg.find_edges(1,etype='zt'))
