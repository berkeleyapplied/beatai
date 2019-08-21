import networkx as nx
import numpy as np
from progressbar import print_progress
import matplotlib.pyplot as plt
import community
from cluster import *
import time
from annotations import getAnnotations, extractBeatAnnotationLocations


def create_signal_network(dataset, depth=5):
    # plt.ion()
    ind = range(dataset.shape[0])

    G = nx.Graph()

    for i in ind:
        G.add_node(i)

    print('Building Network with {} nodes'.format(len(ind)))
    ntot = len(ind)
    n = 0

    conn = []

    for i in ind:
        dd = dataset - dataset[i][:]
        dist = np.sqrt(np.sum(np.power(dd,2.0),1))
        dist[i] = np.max(dist)
        sarr = []
        for j in range(len(dist)):
            sarr.append((j,dist[j]))
        sarr.sort(key=lambda tup: tup[1])

        for j in range(depth):
            connTup = (min([i,sarr[j][0]]),max([i,sarr[j][0]]))
            if not connTup in conn:
                G.add_edge(connTup[0],connTup[1])
                conn.append(connTup)


        n+=1
        print_progress(n, ntot)

    return G


def cluster_signals(dset, depth=5):
    G = create_signal_network(dset, depth=depth)
    partition = community.best_partition(G)

    # Convert to indices
    cls = []
    for i in range(len(G.nodes)):
        cls.append(partition[i])
    C = {}
    for c in range(max(cls)):
        C[c] = []
        for i in range(len(G.nodes)):
            if partition[i] == c:
                C[c].append(i)

    return G, C, partition

def label_community_nodes(G, part):
    for i in range(len(G.nodes)):
        G.node[i]['community'] = part[i]
    return G

def label_community_nodes_by_class(G,cls):

    for k in cls:
        for i in cls[k]:
            G.node[i]['community'] = k
    return G


def community_envelope(dataset, cls, ncls, plotflag=False):

    ind = cls[ncls]

    S = []
    nsr = int(dataset.shape[1]/2)

    for i in ind:
        S.append(list(dataset[i][:nsr]))
    S = np.array(S)
    Sm = np.mean(S,0)
    Sstd = np.std(S,0)

    if plotflag:
        plt.plot(Sm,'k')
        plt.plot(Sm+Sstd,'r')
        plt.plot(Sm-Sstd,'r')
        plt.show()

    return Sm, Sstd

def plot_community_envelope(dataset, cls, ncls, nstd=0.0, clr='k'):
    ind = cls[ncls]

    if type(clr)==list or type(clr)==tuple:
        if len(clr) == 3:
            if clr[0]>1 or clr[1]>1 or clr[2]>1:
                clr[0] = float(clr[0])/255.
                clr[1] = float(clr[1])/255.
                clr[2] = float(clr[2])/255.
    S = []
    nsr = int(dataset.shape[1] / 2)

    for i in ind:
        S.append(list(dataset[i][:nsr]))
    S = np.array(S)
    Sm = np.mean(S, 0)
    Sstd = np.std(S, 0)

    plt.plot(Sm, color=clr, label='Class {}'.format(ncls))
    if nstd > 0:
        plt.plot(Sm + nstd*Sstd, '--',color=clr)
        plt.plot(Sm - nstd*Sstd, '--',color=clr)
    plt.legend()

def community_joint_distribution(dataset, cls, acls, bcls, plotflag=False):

    sma, stda = community_envelope(dataset, cls, acls)
    smb, stdb = community_envelope(dataset, cls, bcls)


def runNetworkCluster(root, datan, useAnn=True, filter=30.0, depth=5):

    datfile = '{}/{}.dat'.format(root,datan)
    atrfile = '{}/{}.atr'.format(root,datan)

    ann = getAnnotations(atrfile)
    if useAnn:
        pk, sym = extractBeatAnnotationLocations(ann)
        sr, si, pk1 = generateSignalMatrix(datfile, pk=pk, datafile=True, filter=filter)
    else:
        sr, si, pk1 = generateSignalMatrix(datfile, datafile=True, filter=filter)

    dset = convertSignalMatrixToDataset(sr,si)
    G, cls, part = cluster_signals(dset, depth=depth)

    return dset, G, cls










