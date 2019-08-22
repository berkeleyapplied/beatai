import networkx as nx
import numpy as np
from progressbar import print_progress
import matplotlib.pyplot as plt
import community
from cluster import *
import time
from annotations import getAnnotations, extractBeatAnnotationLocations

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def create_signal_network(dataset, depth=5, rng=200):
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
        r1 = i-int(rng/2)
        r2 = i+int(rng/2)
        if r1<0:
            r1 = 0
            r2 = rng
        if r2 > len(ind):
            r1 = len(ind)-rng
            r2 = len(ind)

        sind = np.arange(r1,r2)

        dd = dataset[sind][:] - dataset[i][:]
        #dd = dataset - dataset[i][:]
        dist = np.sqrt(np.sum(np.power(dd,2.0),1))
        dist[sind==i] = np.max(dist)
        sarr = []
        for j in range(len(sind)):
            sarr.append((sind[j],dist[j]))
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
    for c in range(max(cls)+1):
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

def communityZScore(dataset, cls, ncls, n, plotflag=False):

    nx = int(dataset.shape[1]/2)
    Sm, Sstd = community_envelope(dataset, cls, ncls);
    z = np.abs(dataset[n][:nx]-Sm)/Sstd
    if plotflag:
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        plot_community_envelope(dataset, cls, ncls, nstd=1.0)
        ax2.plot(dataset[n][:nx],'r')
        ax1.plot(z)
        plt.show()
    return np.mean(z), z




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

def plot_community_symbols(root, datan, cls, pk1):
    datfile = '{}/{}.dat'.format(root, datan)
    atrfile = '{}/{}.atr'.format(root, datan)
    ann = getAnnotations(atrfile)
    pk, sym = extractBeatAnnotationLocations(ann)
    sym = sym[1:-1]

    D1, D2 = retrieveBatch(datfile, None, None, True)
    s1, p, c = D1

    plt.plot(s1,'b')
    ncls = len(cls.keys())
    cmap = get_cmap(ncls)
    nclr = 0
    for k in cls:
        ind = cls[k]
        plt.plot(pk1[ind],s1[pk1[ind]],'.',color=cmap(nclr))
        nclr += 1
        for i in ind:
            plt.text(pk[i],s1[pk[i]]+20.0,'{}-{}'.format(sym[i],k))
    plt.show()


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
    # TODO: re-cluster...combine...etc...

    G = label_community_nodes(G, part)

    return dset, G, cls, part, pk1










