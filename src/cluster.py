from beatdetection import *
from scale import *
import numpy as np
import matplotlib.pyplot as plt
from morphologyStudy import retrieveBatch
from filters import *
from scipy.signal import hilbert
from scipy.interpolate import interp1d, interp2d
import math
from annotations import getAnnotations
import sys
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from detection import update, rescale, butter_bandpass_filter, butter_lowpass_filter, removeBaseline

def generateRRInterval(infile, datafile=True):
    fs = 360

    srec = 0
    BATCH_NUM = 10000

    batchNum = 0
    resetCounter = 200

    RR = np.array([])

    pkLast = -1
    while (True):

        D1, D2 = retrieveBatch(infile, BATCH_NUM, srec, datafile)
        srec += BATCH_NUM

        if type(D1) == type(None):
            break

        (S1, pk1, c1) = D1

        if pkLast > 0:
            pk1 = np.concatenate((np.array([pkLast]), pk1))
        pkLast = pk1[-1]

        rint = pk1[1:]-pk1[:-1]
        RR = np.concatenate((RR,np.array(rint)/fs))
    RR = RR[:-1] # last interval
    return RR


def generateSignalMatrix(infile, pk=None, datafile=True, filter=10.0):

    fs = 360
    corner_freq = filter

    srec = 0
    BATCH_NUM = 10000

    batchNum = 0
    resetCounter = 200
    interpRange = 1024

    SR1 = []
    SI1 = []


    D1, D2 = retrieveBatch(infile, None, None, datafile)
    srec += BATCH_NUM

    (S1, pk1, c1) = D1

    if pk is not None:
        pk1 = pk

    S1F = removeBaseline(S1, highcut=filter)
    pk_out = []
    n1 = 0
    while True:
        n1 += 1

        if n1 >= len(pk1)-1:
            break

        if resetCounter > 0:
            resetCounter -= 1

        s = rescale(S1F, pk1, n1, scaleLength=interpRange)

        if len(s) == interpRange:

            xa = hilbert(s)

            si = butter_lowpass_filter(xa.imag, corner_freq, fs, order=4)
            sr = butter_lowpass_filter(xa.real, corner_freq, fs, order=4)

            xc = np.arange(len(sr))
            xint = np.linspace(0.01, len(sr)-1.01, 2000)

            fr = interp1d(xc, sr, kind='cubic')
            fi = interp1d(xc, si, kind='cubic')

            sr = fr(xint)
            si = fi(xint)

            SR1.append(sr[500:1500])
            SI1.append(si[500:1500])

            pk_out.append(pk[n1])

    return np.array(SR1), np.array(SI1), np.array(pk_out)

def convertSignalMatrixToDataset(sr, si):

    M = []
    for i in range(len(sr)):
        M.append(list(sr[i])+list(si[i]))
    return to_time_series_dataset(M)

def signalMahobolisDistance(S, n):

    D = []
    for s in S:
        ds = s - S[n]
        mdist = np.sqrt(np.sum(np.power(ds,2.0)))
        D.append(mdist)
    return D

def signalDistance(S, n):

    D = []
    for s in S:
        ds = s-S[n]
        mdist = np.sum(np.abs(ds))
        D.append(mdist)
    return D


def clusterSignalData(datfile, clusterNum,filter=12.0, verbose=False, plotflag=False):

    sr, si, pk = generateSignalMatrix(datfile, filter=filter)
    dset = convertSignalMatrixToDataset(sr,si)

    ncluster = clusterNum


    km = TimeSeriesKMeans(n_clusters=ncluster, metric="softdtw", verbose=False)
    res = km.fit(dset)
    cls = res.predict(dset)

    ind = {}
    for i in range(ncluster):
        ind[i] = np.where(cls==i)[0]

    if verbose:
        print('Signal Clustering Report')
        print('NPeaks: {}'.format(len(pk)))
        print('NClusters: {}'.format(len(ind)))
        for i in ind:
            print('Cluster {}: {}'.format(i, len(ind[i])))



    return sr, si, pk, cls, ind, res

# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-f', '--filterlowpass', type=float, required=False, default=10.0, help='Filter Low Pass Hz')
#     parser.add_argument('-d','--directory', required=True, help='data directory')
#     parser.add_argument('-n', '--dataset', required=True, help='dataset number')
#     args = parser.parse_args()
#
#
#     datafile = args.directory + '/' + args.dataset + '.dat'
#     afile = args.directory + '/' + args.dataset + '.atr'
#
#     run_datafile(datafile, annfile=afile, filter=args.filterlowpass, datafile=True)





