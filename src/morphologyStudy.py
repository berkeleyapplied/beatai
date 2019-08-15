from parse import *
from beatdetection import *
from scale import *
import numpy as np
import matplotlib.pyplot as plt
from generateModelECG import fitECGModel


def updateEWMA(m, sig, S, pk, c, n, filterSNR, snrThresh = 2.0, fil=60):


    if n > 1 and n < len(pk) - 2:
        if c[n]:
            s = skewBaseline(S, pk, n)
            if not type(s) == type(None):

                if filterSNR:
                    snr = np.mean(np.abs(s-m)/sig)
                    print('SNR: {} - {}'.format(snr,snr < snrThresh))
                    if snr > snrThresh:
                        return [] # Too noisy

                var = np.power((s-m),2.0)
                sig += (np.sqrt(var)-sig)/fil
                m += (s - m) / fil
                return s
    return []



def updateDensityMatrix(M, S, pk, c, n, beatfilter=60):

    xedges = np.linspace(-0.5, 1024.5, 1026)
    yedges = np.linspace(-0.5, 1.5, 1026)

    if n > 1 and n < len(pk)-2:
        if c[n]:
            s = ppSkew(S, pk, n)
            if not type(s)==type(None):
                H, xedges, yedges = np.histogram2d(range(len(s)), s, bins=(xedges, yedges))
                M += (H.T-M)/beatfilter
    return M


def updateEnvelope(M, S, pk, c, n, envCtr):

    if n > 1 and n < len(pk) - 2:
        if c[n]:
            s = ppSkew(S, pk, n)
            if not type(s) == type(None):
                i = envCtr % M.shape[0]
                M[i,:] = ppSkew(S, pk, n)
                return True

    return False


def retrieveBatch(filename, nrec, srec, datafile=False):

    if not datafile:
        S1, S2 = getECGData(filename, nrec, srec)
    else:
        S1, S2 = parse212File(filename, nrec, srec)

    if len(S1)==0:
        return None, None

    pk1 = findPeakLocations(S1)
    pk2 = findPeakLocations(S2)
    c1 = classifyPeaks(S1, pk1)
    c2 = classifyPeaks(S2, pk2)

    return (S1, pk1, c1), (S2, pk2, c2)


def run_envelope(dbfile, depth=180):
    srec = 0
    M1 = np.zeros((depth, 1024))
    BATCH_NUM = 10000
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    framenum = 0
    batchNum = 0
    traceCounter = 0

    while(True):
        D1, D2 = retrieveBatch(dbfile, BATCH_NUM, srec)
        srec += BATCH_NUM

        if type(D1) == type(None):
            break

        baseTime = batchNum*BATCH_NUM  # seconds

        (S1, pk1, c1) = D1
        #(S2, pk2, c2) = D2  # TODO: Sync both signals

        # Doing S1 for now
        for n1 in range(2,len(pk1)-2):
            if(updateEnvelope(M1, S1, pk1, c1, n1, traceCounter)):
                traceCounter += 1

            ts = baseTime + float(pk1[n1])/250.

            if n1 % 10 == 0:

                med = np.median(M1,axis=0)
                sdev = M1.std(axis=0)

                ax.cla()

                x = range(1024)
                ax.plot(x,med,'k')
                ax.plot(x,med+sdev,'r.',markersize=1.0)
                ax.plot(x,med-sdev,'r.',markersize=1.0)

                fname = 'frames/tmp_{:05d}.png'.format(framenum)
                framenum+=1

                hr = int(ts)/3600
                min = (int(ts) % 3600)/60
                sec = (int(ts) % 3600) % 60

                ax.set_title('Time: {} hr {} min {} sec'.format(hr,min,sec))
                ax.set_xlim((0,1024))
                ax.set_ylim((-0.2,1.0))
                ax.grid(True)
                plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                plt.pause(0.1)
        batchNum+=1



def run_density(dbfile):
    srec = 0
    xedges = np.linspace(-0.5, 1024.5, 1026)
    yedges = np.linspace(-0.5, 1.5, 1026)
    X, Y = np.meshgrid(xedges, yedges)
    H, xedges, yedges = np.histogram2d([0], [0], bins=(xedges, yedges))
    H = H.T
    M1 = np.zeros(np.shape(H))
    M2 = np.zeros(np.shape(H))
    BATCH_NUM = 10000
    plt.ion()
    fig, ax = plt.subplots(1,1)
    framenum = 0
    batchNum = 0

    while(True):
        D1, D2 = retrieveBatch(dbfile, BATCH_NUM, srec)
        srec += BATCH_NUM
        if type(D1) == type(None):
            break

        baseTime = batchNum*BATCH_NUM  # seconds

        (S1, pk1, c1) = D1
        #(S2, pk2, c2) = D2  # TODO: Sync both signals

        # Doing S1 for now
        for n1 in range(2,len(pk1)-2):
            M1 = updateDensityMatrix(M1, S1, pk1, c1, n1)
            ts = baseTime + float(pk1[n1])/250.

            if n1 % 10 == 0:
                ax.cla()
                Mn = M1/M1.sum()
                plt.pcolormesh(X, Y, np.power(Mn, 0.2), cmap='jet')
                fname = 'frames/tmp_{:05d}.png'.format(framenum)
                framenum+=1

                hr = int(ts)/3600
                min = (int(ts) % 3600)/60
                sec = (int(ts) % 3600) % 60

                plt.title('Time: {} hr {} min {} sec'.format(hr,min,sec))

                plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None)
                plt.pause(0.1)
        batchNum+=1

def run_ewma(dbfile, filtertime):

    srec = 0
    m1 = np.zeros(1024)
    sig1 = np.zeros(1024)
    BATCH_NUM = 10000

    plt.ion()
    fig, (ax,ax2) = plt.subplots(2, 1, sharex=True)

    framenum = 0
    batchNum = 0
    traceCounter = 0
    ctr = 0
    resetCounter = 200

    while(True):
        D1, D2 = retrieveBatch(dbfile, BATCH_NUM, srec)
        srec += BATCH_NUM

        if type(D1) == type(None):
            break

        baseTime = batchNum*BATCH_NUM  # seconds

        (S1, pk1, c1) = D1

        #(S2, pk2, c2) = D2  # TODO: Sync both signals

        # Doing S1 for now
        params = None
        for n1 in range(2,len(pk1)-2):

            filterSNR = resetCounter <= 0
            if resetCounter > 0:
                resetCounter -= 1

            s = updateEWMA(m1, sig1, S1, pk1, c1, n1, filterSNR, snrThresh = 3.0, fil=filtertime)

            if(len(s) > 0):
                traceCounter += 1
            ctr+=1
            ts = baseTime + float(pk1[n1])/250.

            if traceCounter % 10 == 0:

                ax.cla()
                ax2.cla()
                x = range(1024)
                ax.plot(x,m1,'k')
                ax.plot(x,m1+sig1,'r.',markersize=0.5)
                ax.plot(x,m1-sig1,'r.',markersize=0.5)

                if len(s) == len(m1):
                    snr = np.abs(s-m1)/sig1
                    ax2.plot(x, snr)
                    ax2.set_xlim((0, 1024))
                    ax2.set_ylim((-0.2, 3.0))
                    ax2.grid(True)

                fname = 'frames/tmp_{:05d}.png'.format(framenum)
                framenum+=1

                hr = int(ts)/3600
                min = (int(ts) % 3600)/60
                sec = (int(ts) % 3600) % 60

                ax.set_title('Time: {} hr {} min {} sec'.format(hr,min,sec))
                ax.set_xlim((0,1024))
                ax.set_ylim((-0.5,1.0))
                ax.grid(True)
                # plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
                #     orientation='portrait', papertype=None, format=None,
                #     transparent=False, bbox_inches=None, pad_inches=0.1,
                #     frameon=None)
                plt.pause(0.01)


            # if n1 > 100:
            #     params = fitECGModel(m1, params)

        batchNum+=1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input db file')
    parser.add_argument('-t', '--filtertime', type=float, required=True, help='EWMA time filter...')

    args = parser.parse_args()
    run_ewma(args.input, args.filtertime)
    #run_density(args.input)





