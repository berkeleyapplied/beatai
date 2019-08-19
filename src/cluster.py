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



def generateSignalMatrix(infile, datafile=True, filter=10.0):

    fs = 360
    corner_freq = filter

    srec = 0
    BATCH_NUM = 10000

    batchNum = 0
    resetCounter = 200
    interpRange = 1024

    SR = []
    SI = []

    while(True):

        D1, D2 = retrieveBatch(infile, BATCH_NUM, srec, datafile)
        srec += BATCH_NUM

        if type(D1) == type(None):
            break

        (S1, pk1, c1) = D1
        S1F = removeBaseline(S1, highcut=filter)

        n1 = 0

        while True:
            n1 += 1

            if n1 > len(pk1)-2:
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

                SR.append(sr)
                SI.append(si)

        batchNum+=1

    return np.array(SR), np.array(SI)









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





