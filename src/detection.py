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

state = 0  # Running is Default
beatreq=0

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


def rescale(S1F, pk1, n1):

    s = S1F[pk1[n1-1]:pk1[n1+1]]
    s = s/S1F[pk1[n1]]
    xi = np.linspace(0,len(s),1024)
    return np.interp(xi, np.arange(len(s)),s)


def figure_close(evt):
    exit()


def key_press(event):
    global state
    global beatreq

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

    ax0 = plt.subplot2grid((6,1),(0,0))
    ax1 = plt.subplot2grid((6,1),(1,0))
    ax2 = plt.subplot2grid((6,1),(2,0))
    ax3 = plt.subplot2grid((6,1),(3,0),rowspan=3)

    batchNum = 0
    traceCounter = 0
    ctr = 0
    resetCounter = 200


    while(True):

        D1, D2 = retrieveBatch(dbfile, BATCH_NUM, srec, datafile)
        srec += BATCH_NUM

        if type(D1) == type(None):
            break


        (S1, pk1, c1) = D1
        #(S2, pk2, c2) = D2  # TODO: Sync both signals


        S1F = removeBaseline(S1, highcut=filter)

        # f2 = plt.figure()
        # plt.plot(S1,'k')
        # plt.plot(SLP,'c')
        # plt.plot(pk1[c1==1], S1[pk1[c1==1]], 'g.')
        # plt.plot(pk1[c1==0], S1[pk1[c1==0]], 'r.')
        # plt.show()

        # Doing S1 for now

        n1 = 1
        while True:
            n1 += 1
            if n1 > len(pk1)-2:
                break

            if resetCounter > 0:
                resetCounter -= 1

            # s = update(S1, pk1, c1, n1)
            s = rescale(S1F, pk1, n1)


            if(len(s) > 0):
                traceCounter += 1
            ctr+=1

            x = range(1024)
            if len(s) == 1024:

                xa = hilbert(s)

                # si = xa.imag
                # sr = xa.real

                si = butter_lowpass_filter(xa.imag, corner_freq, fs, order=4)
                sr = butter_lowpass_filter(xa.real, corner_freq, fs, order=4)

                # si = savgol_filter(xa.imag, 31, 3)
                # sr = savgol_filter(xa.real,31, 3)

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


                ax1.cla()
                ax2.cla()
                ax1.plot(x,s,'k')
                ax2.plot(x,sr,'k')
                ax2.plot(x,si,'r')


                sr = sr[512-256:512+255]
                si = si[512-256:512+255]

                xc = np.arange(len(sr))
                xint = np.linspace(0.01,len(sr)-1.01,2500)
                fr = interp1d(xc, sr, kind='cubic')
                fi = interp1d(xc, si, kind='cubic')

                sr = fr(xint)
                si = fi(xint)

                ax3.cla()
                ax3.plot(sr,si,'r.')
                ax3.axis('equal')

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

            plt.pause(0.01)

        batchNum+=1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='input db file')
    parser.add_argument('-f', '--filterlowpass', type=float, required=False, default=10.0, help='Filter Low Pass Hz')
    parser.add_argument('-a', '--annotationfile', required=False, default=None, help='Annotation File')
    parser.add_argument('--datafile', action='store_true')



    args = parser.parse_args()
    run_datafile(args.input, datafile=args.datafile, annfile=args.annotationfile, filter=args.filterlowpass)





