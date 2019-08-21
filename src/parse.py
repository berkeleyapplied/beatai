#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import csv

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '#' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def int12(x):
    if x>0xFFF:
        print(x)
        raise OverflowError
    if x>0x7FF:
        x = -1*int(0x1000-x)
    return x

def int12r(x):

    x = int(x)

    if x >= 2**12/2:
        print(x)
        raise OverflowError

    if x < -1 * 2**12/2:
        print(x)
        raise OverflowError

    if x < 0:
        x = 0x800 | (0x7FF & x)
    else:
        x = 0x7FF & x

    return x



def write212File(fname, S1, S2, progress=False):

    L = len(S1)

    with open(fname, 'wb') as rb:
        t1 = time.time()
        for i in range(L):
            if i % 1000 == 0 and progress:
                t2 = time.time()
                r = (t2 - t1) / 100.
                if (R > 0):
                    R += (r - R) / 10
                else:
                    R = r
                time_left = (L - i) / R

                if (time_left >= 3600):
                    timestr = ' {:.2f} hr.'.format(time_left / 3600.)
                elif (time_left >= 60):
                    timestr = ' {:.2f} min.'.format(time_left / 60.)
                else:
                    timestr = ' {:.2f} sec.'.format(time_left)

                print_progress(i, L, suffix=timestr)

            v1 = int12r(S1[i])
            v2 = int12r(S2[i])

            b = []
            b.append(0xFF & v1)
            b.append( (v1 & 0xF00) >> 8 | (v2 & 0xF00) >> 4 )
            b.append(0xFF & v2)

            rb.write(bytearray(b))

def parse212File(fname, nrec=None, srec=None):

    if nrec is not None and srec is not None:
        nsamp = int(nrec)*250
        stsamp = int(srec-1)*250 + 1
    else:
        nsamp = -1
        stsamp = 0

    with open(fname,'rb') as bfile:
        bytes = bytearray(bfile.read())

    blen = len(bytes)

    s = range(0,blen,3)  # 24 bit values
    S1 = []
    S2 = []

    ns = 0
    for ind in s:
        if ns >= stsamp:
            b = bytes[ind:ind+3]
            S1.append(int12(b[0] | ((b[1] & 0x0F) << 8)))
            S2.append(int12(((b[1] & 0xF0) << 4) | b[2]))
        ns += 1
        if ns >= nsamp and nsamp >= 0:
            break

    S1 = np.array(S1)
    S2 = np.array(S2)
    plt.show()

    return S1,S2

def parse212(bytes):

    blen = len(bytes)

    s = range(0,blen,3)  # 24 bit values
    S1 = []
    S2 = []

    for ind in s:
        b = bytes[ind:ind+3]
        S1.append(int12(b[0] | ((b[1] & 0x0F) << 8)))
        S2.append(int12(((b[1] & 0xF0) << 4) | b[2]))

    S1 = np.array(S1)
    S2 = np.array(S2)

    return S1,S2


def connectdb(fname):
    try:
        conn = sqlite3.connect(fname)
    except sqlite3.Error as e:
        print(e)
        return
    finally:
        return conn
    return

def parseACCPacket(data):
    # already bytearray
    # convert to signed integer
    c1, c2, c3 = [], [], []
    n = 0
    for b in data:
        if b > 127:
            out = int(b)-256
        else:
            out = int(b)

        c = n%3
        n+=1

        if c == 0:
            c1.append(out)
        if c == 1:
            c2.append(out)
        if c == 2:
            c3.append(out)

    return c1, c2, c3

def getACCData(dbfile,nrec,srec,progress=False):
    conn = connectdb(dbfile)
    c = conn.cursor()
    sql = 'select AccData from EcgData'
    c.execute(sql)
    print('Querying database...')
    rows = c.fetchall()
    conn.close()

    nrows = len(rows)
    print('{} records'.format(len(rows)))


    if nrec is None:
        nrec = nrows - srec

    print('Initializing arrays: {} elements'.format(nrec*250))
    C1 = np.zeros(nrec * 32)
    C2 = np.zeros(nrec * 32)
    C3 = np.zeros(nrec * 32)

    t1 = time.time()
    R = -1
    n = 0
    print('\nParsing elements...\n')
    for row in rows:

        if n >= srec:
            i1 = (n-srec) * 32
            i2 = i1 + 32
            d = bytearray(row[0])
            c1, c2, c3 = parseACCPacket(d)
            C1[i1:i2] = c1
            C2[i1:i2] = c2
            C3[i1:i2] = c3

            R = 0

            if n % 1000 == 0 and progress:
                t2 = time.time()
                r = 1000 / (t2 - t1)
                if R == 0:
                    R = r
                else:
                    R += (r - R) / 50.0
                t1 = time.time()
                time_left = (nrec - (n-srec)) / R

                if (time_left >= 3600):
                    timestr = ' {:.2f} hr.'.format(time_left / 3600.)
                elif (time_left >= 60):
                    timestr = ' {:.2f} min.'.format(time_left / 60.)
                else:
                    timestr = ' {:.2f} sec.'.format(time_left)

                print_progress(n, nrows, suffix=timestr)
        n+=1

        if n >= nrec+srec:
            break

    # C1 = np.array(C1)
    # C2 = np.array(C2)
    # C3 = np.array(C3)

    return C1, C2, C3

def getECGData(dbfile, nrec, srec, progress=False):

    conn = connectdb(dbfile)
    c = conn.cursor()
    sql = 'select EcgData from EcgData'
    c.execute(sql)
    rows = c.fetchall()

    S1 = []
    S2 = []

    n = 0
    nrows = len(rows)

    if nrec is None:
        nrec = nrows - srec

    S1 = np.zeros(nrec*250)
    S2 = np.zeros(nrec*250)

    t1 = time.time()
    R = -1

    for row in rows:

        if n >= srec:
            i1 = (n-srec) * 250
            i2 = i1 + 250
            d = bytearray(row[0])
            s1t, s2t = parse212(d)
            S1[i1:i2] = s1t
            S2[i1:i2] = s2t

            R = 0

            if n % 1000 == 0 and progress:
                t2 = time.time()
                r = 1000 / (t2 - t1)
                if R == 0:
                    R = r
                else:
                    R += (r - R) / 50.0
                t1 = time.time()
                time_left = (nrec - (n - srec)) / R

                if (time_left >= 3600):
                    timestr = ' {:.2f} hr.'.format(time_left / 3600.)
                elif (time_left >= 60):
                    timestr = ' {:.2f} min.'.format(time_left / 60.)
                else:
                    timestr = ' {:.2f} sec.'.format(time_left)

                print_progress(n, nrows, suffix=timestr)

        n+=1

        if n >= nrec+srec:
            break

    conn.close()
    return S1, S2


def writeECGData(dbfile, outputfile, nrec, srec, progress=False):

    print('Connecting to database...')
    conn = connectdb(dbfile)
    c = conn.cursor()
    sql = 'select EcgData from EcgData'
    print('Querying database...')
    c.execute(sql)
    print('Fetching...')
    rows = c.fetchall()

    S1 = []
    S2 = []

    n = 0
    nrows = len(rows)
    print('{} records'.format(len(rows)))

    if nrec is None:
        nrec = nrows - srec

    t1 = time.time()
    R = -1

    print('\nWriting records to {}...\n'.format(outputfile))
    with open(outputfile, 'wb') as rb:
        for row in rows:
            if n >= srec:
                d = bytearray(row[0])
                rb.write(d)
            n += 1

            if n % 10000 == 0 and progress:
                t2 = time.time()
                r = 10000.0/(t2-t1)
                t1 = time.time()
                time_left = (nrows-n)/r

                if(time_left >= 3600):
                    timestr = ' {:.2f} hr.'.format(time_left/3600.)
                elif(time_left >= 60):
                    timestr = ' {:.2f} min.'.format(time_left/60.)
                else:
                    timestr = ' {:.2f} sec.'.format(time_left)

                print_progress(n, nrows, suffix=timestr)

            if n >= nrec+srec:
                break
    conn.close()

def getNumPackets(dbfile):
    conn = connectdb(dbfile)
    c = conn.cursor()
    sql = 'select TimeStamp from EcgData'
    c.execute(sql)
    rows = c.fetchall()
    numPackets = len(rows)
    conn.close()
    return numPackets

def getNUTSData(csvfile, nrec, srec):
    maxVariance=[]
    minVariance=[]
    numPeaks=[]
    numAbsPeaks=[]
    cost=[]
    turningPoint=[]
    numTPPeaks=[]
    numAbsMaxPeaks=[]

    with open(csvfile, 'r') as cfile:
        reader = csv.reader(cfile)
        hdr = True
        n = srec

        for row in reader:
            if hdr:
                hdr = False
            else:
                maxVariance.append(float(row[0]))
                minVariance.append(float(row[1]))
                numPeaks.append(float(row[2]))
                numAbsPeaks.append(float(row[3]))
                cost.append(float(row[4]))
                turningPoint.append(float(row[5]))
                numTPPeaks.append(float(row[6]))
                numAbsMaxPeaks.append(float(row[7]))
                n+=1
                if n >= (nrec-srec):
                    break

    return (np.array(maxVariance), np.array(minVariance), np.array(numPeaks), np.array(numAbsPeaks),
            np.array(cost), np.array(turningPoint), np.array(numTPPeaks), np.array(numAbsMaxPeaks))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dbfile', required=True, help='Database SQLITE3 File')
    parser.add_argument('-n', '--numrec', required=False, type=int, help='Number of records')
    parser.add_argument('-s', '--startrec', required=False, type=int, default=0, help='Number of Start Record')
    parser.add_argument('-o', '--output', required=False, help='Output mit212 data strip file')
    parser.add_argument('--plotdata', action='store_true')
    args = parser.parse_args()

    if args.plotdata:
        S1, S2 = getECGData(args.dbfile, args.numrec, args.startrec, progress=True)
        C1, C2, C3 = getACCData(args.dbfile, args.numrec, args.startrec)

        ts = np.array(range(len(S1)))/250.
        ta = np.array(range(len(C1)))/32.

        fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(ts, S1,'r')
        ax1.plot(ts, S2,'b')
        ax2.plot(ta, C1,'g')
        ax2.plot(ta, C2,'k')
        ax2.plot(ta, C3,'c')
        plt.show()

    if args.output:
        writeECGData(args.dbfile, args.output, args.numrec, args.startrec, progress=True)