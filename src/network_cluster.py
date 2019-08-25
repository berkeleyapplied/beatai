import networkx as nx
import numpy as np
from progressbar import print_progress
import matplotlib.pyplot as plt
import community
from cluster import *
import time
from annotations import getAnnotations, extractBeatAnnotationLocations
from copy import deepcopy

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def create_signal_network(dataset, ind=None, depth=5, rng=200):
    # plt.ion()
    if ind is None:
        ind = range(dataset.shape[0])

    dset = {}
    for i in range(len(ind)):
        dset[ind[i]] = dataset[i][:]

    G = nx.Graph()

    for i in ind:
        G.add_node(i)

    print('Building Network with {} nodes'.format(len(ind)))
    ntot = len(ind)
    n = 0

    conn = []

    for i in ind:
        r1 = n-int(rng/2)
        r2 = n+int(rng/2)
        if r1<0:
            r1 = 0
            r2 = rng
        if r2 > len(ind):
            r1 = len(ind)-rng
            r2 = len(ind)

        sind = np.arange(r1,r2)

        dd = dataset[sind][:] - dataset[n][:]
        #dd = dataset - dataset[i][:]
        dist = np.sqrt(np.sum(np.power(dd,2.0),1))
        dist[sind==n] = np.max(dist)
        sarr = []
        for j in range(len(sind)):
            sarr.append((sind[j],dist[j]))
        sarr.sort(key=lambda tup: tup[1])

        for j in range(depth):
            connTup = (min([i,ind[sarr[j][0]]]),max([i,ind[sarr[j][0]]]))
            if not connTup in conn:
                G.add_edge(connTup[0],connTup[1])
                conn.append(connTup)

        n+=1
        print_progress(n, ntot)

    return G


def cluster_signals(dataset, ind=None, depth=5, beat_range=300):
    G = create_signal_network(dataset, ind=ind, depth=depth, rng=beat_range)
    partition = community.best_partition(G)

    # Convert to indices
    cls = []
    for i in G.nodes:
        cls.append(partition[i])

    C = {}
    for c in range(max(cls)+1):
        C[c] = []
        for i in G.nodes:
            if partition[i] == c:
                C[c].append(i)

    return G, C, partition

def label_community_nodes(G, part):
    for i in range(len(G.nodes)):
        if i in part:
            G.node[i]['community'] = part[i]
        else:
            print('WARNING: Beat {} not within partition!'.format(i))
            G.node[i]['community'] = -1
    return G


def generate_partition_from_class(cls):

    part = {}
    for c in cls:
        for i in cls[c]:
            if i in part:
                print('WARNING: Duplicate partition index ({}) in class structure'.format(i))
            part[i] = c

    return part


def community_envelope(dataset, cls, ncls, ind=None, plotflag=False):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    # Setup dataset dict
    dset = {}
    for i in range(len(ind)):
        dset[ind[i]] = dataset[i][:]

    cind = cls[ncls]

    S = []
    nsr = int(dataset.shape[1]/2)

    for i in cind:
        S.append(list(dset[i][:nsr]))
    S = np.array(S)
    Sm = np.mean(S,0)
    Sstd = np.std(S,0)

    if plotflag:
        plt.plot(Sm,'k')
        plt.plot(Sm+Sstd,'r')
        plt.plot(Sm-Sstd,'r')
        plt.show()

    return Sm, Sstd

def signal_ZScore(dataset, cls, ncls, n, ind=None, plotflag=False):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    dset = {}
    for i in range(len(ind)):
        dset[ind[i]] = dataset[i][:]

    nx = int(dataset.shape[1]/2)
    Sm, Sstd = community_envelope(dataset, cls, ncls, ind=ind)
    z = np.abs(dset[n][:nx]-Sm)/Sstd
    if plotflag:
        fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)
        plot_community_envelope(dataset, cls, ncls, nstd=1.0)
        ax2.plot(dset[n][:nx],'r')
        ax1.plot(z)
        plt.show()
    return np.mean(z), z

def community_prune_highz(dataset, cls, ncls, nstd=2.0, ind=None, plotflag=False):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    dset = {}
    for i in range(len(ind)):
        dset[ind[i]] = dataset[i][:]

    cind = np.array(cls[ncls]).astype(int)

    nx = int(dataset.shape[1] / 2)
    Sm, Sstd = community_envelope(dataset, cls, ncls, ind=ind)
    z = []
    for n in cind:
        z.append(np.mean(np.abs(dset[n][:nx] - Sm) / (nstd * Sstd)))
    z = np.array(z)

    pruneList = cind[np.where(z>1.0)[0]]
    keepList = cind[np.where(z<=1.0)[0]]

    if plotflag:
        plot_community_envelope(dataset, cls, ncls, nstd=1.0, clr='r', traces=True)
        for p in pruneList:
            plt.plot(dset[p][:nx],'k')
        plt.show()
    return np.array(keepList).astype(int), np.array(pruneList).astype(int)


def community_prune(dataset, cls, nstd=2.0, ind=None):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    if nstd < 1.5:
        print('Warning: Pruning at low sigma values may not be stable!')

    pruned = np.array([])
    print('Pruning with {} sigma threshold'.format(nstd))
    n = 0
    while(True):
        p_iter = np.array([])
        for c in cls:
            k, p = community_prune_highz(dataset, cls, c, nstd=nstd, ind=ind)
            if c==6:
                print('Class 6 Keep: {}'.format(k))
                print('Class 6 Prune: {}'.format(p))

            p_iter = np.append(p_iter,p)
            cls[c] = k
        pruned = np.append(pruned,p_iter)
        n += 1
        print('Iteration {}: pruned {} beats'.format(n, len(p_iter)))
        if len(p_iter) == 0:
            break
    return cls, pruned


def community_reintegrate(dataset, cls, pruned, ind=None, nstd=2.0):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    #setup dataset dictionary due to changing indices
    dset = {}
    for i in range(len(ind)):
        dset[ind[i]] = dataset[i][:]

    outliers = []
    SM = {}
    SSTD = {}
    for c in cls:
        Sm, Sstd = community_envelope(dataset, cls, c, ind=ind)
        SM[c] = Sm
        SSTD[c] = Sstd

    nx = int(dataset.shape[1] / 2)
    for i in pruned:
        ztup = []
        for c in cls:
            z = np.abs(dset[int(i)][:nx] - SM[c]) / SSTD[c]
            ztup.append((c, np.mean(z)))

        ztup.sort(key=lambda tup: tup[1])
        if ztup[0][1] < nstd:  # this is best bit
            c = ztup[0][0]
            cls[c] = np.append(cls[c],np.array([i])).astype(int)
            # print('Process beat {}: class {}'.format(i,c))
        else:
            outliers.append(i)
            # print('Process beat {}: Outlier'.format(i))

    return cls, np.array(outliers)


def community_restructure(dataset, cls, nstd=2.0, ind=None):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    cls, pruned = community_prune(dataset, cls, nstd=nstd, ind=ind)
    cls, outliers = community_reintegrate(dataset, cls, pruned, nstd=nstd, ind=ind)
    # Create new outliers class:
    # newkey = max(cls.keys())+1
    # cls[newkey] = outliers
    return cls, outliers


def signal_envelope_test(dataset, cls, ncls, n, nstd=1.0):

    nx = int(dataset.shape[1] / 2)
    Sm, Sstd = community_envelope(dataset, cls, ncls)

    z = np.abs(dataset[n][:nx] - Sm) / (nstd*Sstd)
    metric = np.zeros(z.shape)
    metric[z<1.0] = 1.0/len(z)
    return np.sum(metric)


def plot_community_envelope(dataset, cls, ncls, nstd=0.0, ind=None, clr='k', traces=False, alpha=0.2):

    if ind is None:
        ind = np.arange(dataset.shape[0])

    dset = {}
    for i in range(len(ind)):
        dset[ind[i]] = dataset[i][:]

    cind = cls[ncls]

    if type(clr)==list or type(clr)==tuple:
        if len(clr) == 3:
            if clr[0]>1 or clr[1]>1 or clr[2]>1:
                clr[0] = float(clr[0])/255.
                clr[1] = float(clr[1])/255.
                clr[2] = float(clr[2])/255.
    S = []
    nsr = int(dataset.shape[1] / 2)

    for i in cind:
        if traces:
            plt.plot(dset[i][:nsr],color=clr,linewidth=0.5,alpha=alpha)
        S.append(list(dset[i][:nsr]))
    S = np.array(S)
    Sm = np.mean(S, 0)
    Sstd = np.std(S, 0)

    plt.plot(Sm, color=clr, label='Class {}'.format(ncls))
    if nstd > 0:
        plt.plot(Sm + nstd*Sstd, '--',color=clr,linewidth=2.0)
        plt.plot(Sm - nstd*Sstd, '--',color=clr,linewidth=2.0)
    plt.legend()


def plot_community_symbols(root, datan, cls, pk1, nclass=None, srange=None):
    # NOTE: pk1 input must be related to annotation peak locations from atrfile!!!
    datfile = '{}/{}.dat'.format(root, datan)
    atrfile = '{}/{}.atr'.format(root, datan)
    ann = getAnnotations(atrfile)
    pk, sym = extractBeatAnnotationLocations(ann)
    sym = sym[1:-1]

    D1, D2 = retrieveBatch(datfile, None, None, True)
    s1, p, c = D1

    sind = np.array([])
    if type(srange)==list or type(srange)==tuple:
        if len(srange) == 2:
            sind = np.arange(srange[0],srange[1]+1)
    if len(sind)==0:
        sind = np.arange(len(s1))


    plt.plot(sind, s1[sind],'b')
    ncls = len(cls.keys())
    cmap = get_cmap(ncls)
    nclr = 0
    for k in cls:
        if nclass is not None:
            if k != nclass:
                continue
        ind = cls[k]
        pk = np.intersect1d(pk1[ind],sind)
        plt.plot(pk, s1[pk], '.',color=cmap(nclr))
        nclr += 1
        for i in ind:
            if pk1[i]>=sind[0] and pk1[i]<=sind[-1]:
                plt.text(pk1[i],s1[pk1[i]]+20.0,'{}-{}'.format(sym[i],k))
    plt.show()

def barchart_community(root, datan, cls, pk1, clsList=None):
    # NOTE: pk1 input must be related to annotation peak locations from atrfile!!!
    datfile = '{}/{}.dat'.format(root, datan)
    atrfile = '{}/{}.atr'.format(root, datan)
    ann = getAnnotations(atrfile)
    pk, sym = extractBeatAnnotationLocations(ann)
    sym = sym[1:-1]

    if clsList is None:
        clsList = cls.keys()

    us = tuple(set(sym))
    fig, ax = plt.subplots(len(clsList),1)
    xp = np.arange(len(us))

    nplt = 0
    for c in clsList:
        ind = cls[c]
        n = []

        for u in us:
            n.append(len([i for i in ind if sym[i]==u]))

        ax[nplt].bar(xp, n, align='center', alpha=0.5)
        ax[nplt].set_xticks(xp, us)
        nplt+=1

    plt.show()

def report_community(root, datan, cls):
    # NOTE: pk1 input must be related to annotation peak locations from atrfile!!!
    datfile = '{}/{}.dat'.format(root, datan)
    atrfile = '{}/{}.atr'.format(root, datan)
    ann = getAnnotations(atrfile)
    pk, sym = extractBeatAnnotationLocations(ann)
    sym = sym[1:-1]

    clsList = cls.keys()

    us = tuple(set(sym))

    print('****************************************')
    print('      {} Community Class Report         \n'.format(datan))
    for c in clsList:
        print('Class {}: '.format(c), end="")
        ind = cls[c]

        for u in us:
            i2 = [i for i in ind if sym[i]==u]
            print('   {}:{} '.format(u,len(i2)), end="")
        print('')
    print('\n****************************************\n')


def combine_graphs(G,g,outliers):

    #remove outliers
    for o in outliers:
        if o in g.nodes:
            g.remove_node(o)
        if o in G.nodes:
            G.remove_node(o)

    # first remove all nodes in G that are in g
    for n in g.nodes:
        if n in G.nodes:
            G.remove_node(n)

    # return the composed graph
    return nx.compose(G,g)


def runNetworkCluster(root, datan, useAnn=True, filter=30.0, depth=5, beat_range=300, nstd=2.0):

    datfile = '{}/{}.dat'.format(root,datan)
    atrfile = '{}/{}.atr'.format(root,datan)

    cls_tot = {}  # this is the continuous class structure

    ann = getAnnotations(atrfile)
    if useAnn:
        pk, sym = extractBeatAnnotationLocations(ann)
        sr, si, pk1 = generateSignalMatrix(datfile, pk=pk, datafile=True, filter=filter)
    else:
        sr, si, pk1 = generateSignalMatrix(datfile, datafile=True, filter=filter)

    dataset = convertSignalMatrixToDataset(sr,si)
    dset = deepcopy(dataset)
    ind = np.arange(dataset.shape[0])
    G = nx.Graph()

    while(True):

        print(dset.shape)
        g, cls, part = cluster_signals(dset, ind=ind, depth=depth, beat_range=beat_range)
        cls, outliers = community_restructure(dset, cls, nstd=nstd, ind=ind)
        outliers = np.array(outliers).astype(int)
        G = combine_graphs(G,g,outliers)

        c = cls_tot.keys()
        if len(c) > 0:
            cmax = max(c)+1
        else:
            cmax = 0

        for ctemp in cls.keys():
            cls_tot[ctemp+cmax] = cls[ctemp]

        # Catch outliers in the total class structure...
        cls_tot, outliers_tot = community_restructure(dataset, cls_tot, nstd=nstd)
        outliers = np.append(outliers, outliers_tot)

        part_tot = generate_partition_from_class(cls_tot)

        if len(outliers) <= 1:
            if len(outliers) == 1:
                ncls = max(cls_tot.keys())+1
                cls_tot[ncls] = outliers
                part_tot[outliers[0]] = ncls
                G.add_node(outliers[0])
            break
        else:
            outliers = np.array(outliers).astype(int)
            # rebuild dataset based on outliers
            # outliers are the original node numbers...need to maintain that indexing
            dnext = []
            for o in outliers:
                oind = np.where(ind==o)[0][0]
                dnext.append(dset[oind][:])
            dset = to_time_series_dataset(dnext)
            beat_range = len(outliers)
            ind = outliers

    G = label_community_nodes(G, part_tot)

    return dataset, G, cls_tot, part_tot, pk1

if __name__ == '__main__':
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filterlowpass', type=float, required=False, default=30.0, help='Filter Low Pass Hz (def=30.0)')
    parser.add_argument('-d','--directory', required=True, help='data directory')
    parser.add_argument('-n', '--dataset', required=True, help='dataset number')
    parser.add_argument('-s', '--nstd', required=False, type=float, default=1.5, help='Prune Sigma (def=1.5)')
    parser.add_argument('-c', '--connection_depth', required=False, type=int, default=5, help='Network Connection Depth')
    parser.add_argument('-b', '--beat_range', required=False, type=int, default=300, help='Local Beat Range')
    parser.add_argument('-o', '--output_file', required=False, help='Output Pickle File for Data')
    parser.add_argument('--beat_detection', action='store_true', help='Use internal beat detection')


    args = parser.parse_args()

    dset, G, cls, part, pk1 = runNetworkCluster(
                                args.directory,
                                args.dataset,
                                useAnn=not args.beat_detection,
                                filter=args.filterlowpass,
                                nstd=args.nstd,
                                depth=args.connection_depth,
                                beat_range=args.beat_range)

    print('************* BEATAI SEGMENTATION REPORT *************')
    print('File: {}/{}.dat'.format(args.directory,args.dataset))
    if args.beat_detection:
        print('Internal Beat Detection Used')
    else:
        print('Beat Annotation: {}/{}.atr'.format(args.directory, args.dataset))

    report_community(args.directory, args.dataset, cls)

    if args.output_file is not None:
        data = (args.directory,
                args.dataset,
                args.beat_detection,
                args.filterlowpass,
                args.nstd,
                args.connection_depth,
                args.beat_range,
                dset, G, cls, part, pk1)
        pickle.dump(data, open(args.output_file,'wb'))






