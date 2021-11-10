import numpy as np
import random
import time
import tqdm
import dgl
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
num_walks_per_node = 50
walk_length = 10
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
        identity = int(w[0])
        Target_ids.append(identity)
        Target_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break;
        v = v.strip().split()
        identity = int(v[0])
        paper_name = v[1]
        Disease_ids.append(identity)
        Disease_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    TF_ids_invmap = {x: i for i, x in enumerate(TF_ids)}
    Target_ids_invmap = {x: i for i, x in enumerate(Target_ids)}
    Disease_ids_invmap = {x: i for i, x in enumerate(Disease_ids)}

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
    for x in f_1:
        # print(x)
        x = x.strip().split()
        x[0] = int(x[0])
        x[1] = int(x[1].strip('\n'))
        TF_Target_src.append(TF_ids_invmap[x[0]])
        TF_Target_dst.append(Target_ids_invmap[x[1]])
    for y in f_2:
        y = y.strip().split()
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        Target_Disease_src.append(Target_ids_invmap[y[0]])
        Target_Disease_dst.append(Disease_ids_invmap[y[1]])
    for ss in f_0:
        ss = ss.strip().split()
        ss[0] = int(ss[0])
        ss[1] = int(ss[1].strip('\n'))
        TF_Disease_src.append(TF_ids_invmap[ss[0]])
        TF_Disease_dst.append(Disease_ids_invmap[ss[1]])
    f_1.close()
    f_2.close()
    f_0.close()

    hg = dgl.heterograph({
        ('TF', 'zt', 'Target') : (TF_Target_src, TF_Target_dst),
        ('Target', 'tz', 'TF') : (TF_Target_dst, TF_Target_src),
        ('Target', 'td', 'Disease') : (Target_Disease_src, Target_Disease_dst),
        ('Disease', 'dt', 'Target') : (Target_Disease_dst, Target_Disease_src),
        ('TF', 'zd', 'Disease') : (TF_Disease_src, TF_Disease_dst),
        ('Disease', 'dz', 'TF') : (TF_Disease_dst, TF_Disease_src)
    })
    return hg, TF_names, Target_names, Disease_names



#"TF - Target - Disease - Target - TF" metapath sampling
# metapath=['zt', 'td', 'dt', 'tz']
#'zd', 'dz', 'zt', 'tz'
def generate_metapath():
    output_path = open(os.path.join(path, "output_path_ztdtz_N50_L10_new.txt"), "w")
    count = 0

    hg, TF_names, Target_names, Disease_names = construct_graph()
    print(hg)
    for TF_idx in tqdm.trange(hg.number_of_nodes('TF')):
        traces, _ = dgl.sampling.random_walk(
                hg, [TF_idx] * num_walks_per_node, metapath=['zt', 'td', 'dt', 'tz'] * walk_length)

        # for tr in traces:
        #     outline = ' '.join(
        #             (TF_names if i % 2 == 0 else Target_names)[tr[i]]
        #             for i in range(0, len(tr)))  # skip paper
        #     print(outline, file=output_path)
        # print(traces) #['zt', 'tz']
        # for tr in traces:
        #     outline=""
        #     k = 0
        #     for i in range(0, len(tr)):
        #         if i % 2 == 0:
        #             tt = TF_names[tr[i]]
        #         elif k % 2 == 0:
        #             #tt = ""
        #             k = k + 1
        #             tt = Disease_names[tr[i]]
        #         else:
        #             k = k + 1
        #             tt = Target_names[tr[i]]
        #
        #         outline = outline +' '+tt  #skip Disease
        #
        #     print(outline, file=output_path) #'zd', 'dz', 'zt', 'tz'
        for tr in traces:
            outline=""
            for i in range(0, len(tr)):
                if i % 4 == 0:
                    tt = TF_names[tr[i]]
                elif i % 4 == 2:
                    tt = ""
                    # tt = Disease_names[tr[i]]
                else:
                    tt = Target_names[tr[i]]

                outline = outline +' '+tt  #skip Disease

            print(outline, file=output_path)
    output_path.close()

def toHomo():
    hg, TF_names, Target_names, Disease_names = construct_graph()
    print(hg)
    hg = dgl.to_homogeneous(hg)
    print(hg)
    print(hg.ndata[dgl.NID])
    print(hg.edata[dgl.ETYPE])
    print(hg.edata[dgl.EID])
    nx_g = hg.to_networkx().to_undirected()
    print(nx_g)
    print(nx.number_connected_components(nx_g))
    print(list(nx.isolates(nx_g)))
    print(nx.is_connected(nx_g))
    print(nx_g.edges())
#     plt.figure(figsize=(20, 6))
#
#     plt.subplot(122)
#     plt.title('Directed graph ,DGL', fontsize=20)
#
#     nx.draw(nx_g)
#     plt.show()

if __name__ == "__main__":
    # generate_metapath()
    toHomo()
