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
from mpl_toolkits.mplot3d import Axes3D

state = 0  # Running is Default
beatreq=0
annreq=''

def update(S, pk, c, n):
    if n > 1 and n < len(pk) - 2:
        if c[n]:
            s = skewBaseline(S, pk, n)
            if not type(s) == type(None):
                # if filterSNR:
                #     snr = np.mean(np.abs(s - m) / sig)
                #     if snr > snrThresh:
                #         return []  # Too noisy
                return s
    return []


def ewma(s, k):
    sout = [s[0]]
    curr = s[0]
    for n in range(1,len(s)):
        curr += (s[n] - curr)  / k
        sout.append(curr)
    return np.array(sout)



def rescale(S1F, pk1, n1, scaleLength=256):

    s = S1F[pk1[n1-1]:pk1[n1+1]+1]
    s = s/S1F[pk1[n1]]
    xi1 = np.linspace(0, pk1[n1]-pk1[n1-1]-1, int(scaleLength/2))
    xi2 = np.linspace(pk1[n1]-pk1[n1-1], len(s), int(scaleLength/2))

    s1 = np.interp(xi1, np.arange(len(s)), s)
    s2 = np.interp(xi2, np.arange(len(s)), s)

    return np.concatenate((s1,s2))


def figure_close(evt):
    exit()


def key_press(event):
    global state
    global beatreq
    global annreq

    sys.stdout.flush()

    if event.key == ' ':
        if state <= 0:
            state = 1
        else:
            state = 0

    if event.key == 'left':
        state = -1

    if event.key == 'right':
        state = 2

    if event.key == 'escape':
        plt.close(event.canvas.figure)
        exit()

    if event.key == 'b':
        bstr = input('Enter Beat#: ')
        try:
            beatnum = int(bstr)
            if beatnum > 1:
                beatreq = beatnum
                state=3
            else:
                print('ERROR: beat number must be greater than 1!')
        except:
            print('ERROR converting {} to integer beat number'.format(bstr))

    if event.key == 'a':
        bstr = input('Enter Annotation: ')
        if len(bstr) == 1:
            ann = 'NV.LRAaJSFO!ejEPfpQ'
            if bstr in ann:
                annreq = bstr
                state = 4

    # if event.key == 'x':
    #     visible = xl.get_visible()
    #     xl.set_visible(not visible)
    #     fig.canvas.draw()


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a

def butter_lowpass_filter(data,highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    low = lowcut / nyq
    b, a = butter(order, [low,high], btype='band')
    return b, a

def butter_bandpass_filter(data,lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def removeBaseline(s, lowcut=0.5, highcut=10.0, fs=250):
     return butter_bandpass_filter(s, lowcut, highcut, fs)



def run_datafile(dbfile, annfile=None, datafile=False, filter=10.0):

    global state
    global beatreq
    global annreq

    if annfile is not None:
        annData = getAnnotations(annfile)
        asamp, asym, asub = annData
    else:
        annData = None

    fs = 250
    corner_freq = filter

    srec = 0
    BATCH_NUM = 10000
    plt.ion()

    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', figure_close)
    fig.canvas.mpl_connect('key_press_event', key_press)

    ax0 = plt.subplot2grid((3,4),(0,0),colspan=4)
    ax1 = plt.subplot2grid((3,4),(1,0),colspan=2)
    ax2 = plt.subplot2grid((3,4),(2,0),colspan=2)
    ax3 = plt.subplot2grid((3,4),(1,2),rowspan=2, colspan=2)
    # ax4 = plt.subplot2grid((6,3),(3,1),rowspan=3)
    # ax5 = plt.subplot2grid((6,3),(3,2),rowspan=3)

    batchNum = 0
    traceCounter = 0
    ctr = 0
    resetCounter = 200
    interpRange = 1024

    while(True):

        D1, D2 = retrieveBatch(dbfile, BATCH_NUM, srec, datafile)
        srec += BATCH_NUM

        if type(D1) == type(None):
            break


        (S1, pk1, c1) = D1
        #(S2, pk2, c2) = D2  # TODO: Sync both signals


        S1F = removeBaseline(S1, highcut=filter)


        # Doing S1 for now

        n1 = 1
        while True:
            n1 += 1
            if n1 > len(pk1)-2:
                break

            if resetCounter > 0:
                resetCounter -= 1

            # s = update(S1, pk1, c1, n1)
            s = rescale(S1F, pk1, n1, scaleLength=interpRange)


            if(len(s) > 0):
                traceCounter += 1
            ctr+=1

            x = range(interpRange)

            if len(s) == interpRange:

                xa = hilbert(s)

                # si = xa.imag
                # sr = xa.real

                si = butter_lowpass_filter(xa.imag, corner_freq, fs, order=4)
                sr = butter_lowpass_filter(xa.real, corner_freq, fs, order=4)


                ax0.cla()
                ind = np.arange(pk1[n1]-1000,pk1[n1]+1000)
                ax0.plot(ind, S1[ind],'k')
                ax0.axvline(x=pk1[n1-1])
                ax0.axvline(x=pk1[n1+1])
                ax0.set_title('Beat Number: {}'.format(n1))

                if annData is not None:
                    ylim = ax0.get_ylim()
                    annInd = np.where(np.logical_and(asamp > min(ind)-1, asamp < max(ind)+1))[0]
                    if len(annInd) > 0:
                        ax0.plot(asamp[annInd],S1[asamp[annInd]],'ro')
                        for ai in annInd:
                            ax0.text(asamp[ai], ylim[1],
                                     '{}:{}'.format(asym[ai],asub[ai]))
                    ax0.set_ylim(ylim[0],ylim[1]+100)


                dsi = si[1:] - si[:-1]
                dsr = sr[1:] - sr[:-1]
                ds = np.sqrt(np.power(dsi, 2.0) + np.power(dsr, 2.0))
                ds = np.insert(ds, 0, 0.0)


                amp = np.sqrt(np.power(sr,2.0)+np.power(si,2.0))
                # Phasor Angular Speed
                sr_n = sr / amp
                si_n = si / amp
                dsi_n = si_n[1:] - si_n[:-1]
                dsr_n = sr_n[1:] - sr_n[:-1]
                ds_n = np.sqrt(np.power(dsi_n, 2.0) + np.power(dsr_n, 2.0))
                ds_n = np.insert(ds_n, 0, 0.0)
                # ax2.plot(dsr,'b')
                # ax2.plot(dsi,'r')
                # ax2.plot(ds,'k')


                # ax2.plot(x,sr,'k')
                # ax2.plot(x,si,'r')


                midpt = interpRange/2
                rangeInd = np.arange(midpt-midpt/2,midpt+midpt/2).astype(int)

                # sr = sr[rangeInd]
                # si = si[rangeInd]
                # ds = ds[rangeInd]

                # sr = sr[512-256:512+255]
                # si = si[512-256:512+255]
                # ds = ds[512-256:512+255]

                xc = np.arange(len(sr))
                xint = np.linspace(0.01, len(sr)-1.01, 2000)

                fr = interp1d(xc, sr, kind='cubic')
                fi = interp1d(xc, si, kind='cubic')
                fds = interp1d(xc, ds, kind='cubic')

                sr = fr(xint)
                si = fi(xint)

                ax1.cla()
                ax2.cla()
                ax1.plot(sr, 'k')
                ax2.plot(si, 'r')
                ax1.grid(True)
                ax2.grid(True)


                ax3.cla()
                ax3.plot(sr,si,'r.')
                ax3.axis('equal')
                ax3.grid(True)



            if state == 0:
                n1 += -1

            if state == -1:
                n1 += -2
                state = 0

            if state == 2:
                state = 0

            if state == 3:
                n1 = beatreq-1
                state = 0

            if state == 4:
                if annData is not None:
                    found = False
                    for i in range(len(asym)):
                        if asym[i] == annreq and asamp[i] > pk1[n1+1]:
                            dsamp = np.abs(pk1 - asamp[i])
                            indpk = np.arange(len(pk1))
                            n1 = indpk[dsamp==min(dsamp)][0]
                            print('Annotation found: {}'.format(n1))
                            found=True
                            break
                    if not found:
                        print('{} not found starting from beat {}'.format(annreq, n1))
                else:
                    print('ERROR: Annotation data not loaded!')
                state=0
                n1 += -1


            plt.pause(0.01)

        batchNum+=1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filterlowpass', type=float, required=False, default=10.0, help='Filter Low Pass Hz')
    parser.add_argument('-d','--directory', required=True, help='data directory')
    parser.add_argument('-n', '--dataset', required=True, help='dataset number')
    args = parser.parse_args()


    datafile = args.directory + '/' + args.dataset + '.dat'
    afile = args.directory + '/' + args.dataset + '.atr'

    run_datafile(datafile, annfile=afile, filter=args.filterlowpass, datafile=True)





