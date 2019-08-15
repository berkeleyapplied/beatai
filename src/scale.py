import numpy as np
import matplotlib.pyplot as plt

def rescaleECG(S, pklocs, n ):

    if n > 1 and n < len(pklocs)-2:
        pkloc = pklocs[n]
        s2 = S[pklocs[n-2]:pklocs[n+2]]


        maxS = S[pkloc]
        minS = min(S[pkloc-10:pkloc+10])
        s2 = (s2-minS)/(maxS-minS)


        xs = np.arange(len(s2))
        x = np.linspace(0,len(s2)-1,1024)
        s3 = np.interp(x,xs,s2)
        return s3

    return []

def skewBaseline(S, pklocs, n):

    if n > 0 and n < len(pklocs):
        # *** LHS
        s1 = np.array(S[pklocs[n-1]:pklocs[n]])

        sa = s1
        sa = np.append(sa[0], sa)
        sa = np.append(sa, sa[-1])

        d1 = sa[1:-1] - sa[:-2]
        d2 = sa[1:-1] - sa[2:]
        ind1 = np.arange(len(s1))
        negp = ind1[np.logical_and(d1 < 0, d2 < 0)]
        posp = ind1[np.logical_and(d1 > 0, d2 > 0)]

        n1 = negp[1:3]
        p1 = posp[0:2]

        if not (len(n1)==2 and len(p1)==2):
            return

        x1 = np.mean(np.append(ind1[n1], ind1[p1]))
        y1 = np.mean(np.append(s1[n1], s1[p1]))

        n2 = negp[-3:-1]
        p2 = posp[-2:]

        if not (len(n2)==2 and len(p2)==2):
            return
        x2L = np.mean(np.append(ind1[n2], ind1[p2]))
        y2L = np.mean(np.append(s1[p2], s1[p2]))


        # ****** RHS
        s2 = np.array(S[pklocs[n]:pklocs[n + 1]])

        sa = s2
        sa = np.append(sa[0],sa)
        sa = np.append(sa,sa[-1])

        d1 = sa[1:-1] - sa[:-2]
        d2 = sa[1:-1] - sa[2:]
        ind2 = np.arange(len(s2))
        negp = ind2[np.logical_and(d1<0,d2<0)]
        posp = ind2[np.logical_and(d1>0,d2>0)]

        n1=negp[1:3]
        p1=posp[0:2]

        if not (len(n1)==2 and len(p1)==2):
            return
        x2R = np.mean(np.append(ind2[n1],ind2[p1]))
        y2R= np.mean(np.append(s2[n1],s2[p1]))

        n2 = negp[-3:-1]
        p2 = posp[-2:]
        if not (len(n2)==2 and len(p2)==2):
            return
        x3 = np.mean(np.append(ind2[n2], ind2[p2]))
        y3 = np.mean(np.append(s2[p2], s2[p2]))

        x2 = np.mean([x2L,x2R])
        y2 = np.mean([y2L,y2R])

        m1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - m1 * x1
        os1 = b1 + m1*ind1
        s1 = s1 - os1
        s1 = s1/s1[-1]

        m2 = (y3 - y2) / (x3 - x2)
        b2 = y2 - m2 * x2
        os2 = b2 + m2 * ind2
        s2 = s2 - os2
        s2 = s2 / s2[0]

        xi1 = np.linspace(ind1[0], ind1[-1], 512)
        s1i = np.interp(xi1, ind1, s1)
        xi2 = np.linspace(ind2[0], ind2[-1], 513)
        s2i = np.interp(xi2, ind2, s2)

        s12 = np.append(s1i,s2i[1:])

        return s12



def ppSkew(S, pklocs, n):

    if n > 0 and n < len(pklocs):
        s = np.array(S[pklocs[n]:pklocs[n+1]])
        d1 = s[1:-1] - s[:-2]
        d2 = s[1:-1] - s[2:]

        ind = np.arange(1,len(s)-1)
        i1 = ind[np.logical_and(d1<0,d2<0)]
        if len(i1):
            i1 = i1[0]
        else:
            return

        ineg = ind[np.logical_and(d1>0,d2>0)]
        if len(ineg):
            i2 = ineg[ineg > i1]
            if len(i2):
                i2 = i2[0]
            else:
                return
        else:
            return

        x1 = (i1 + i2)/2.0
        y1 = (s[i1] + s[i2])/2.0


        s2 = s[::-1]
        d1 = s2[1:-1] - s2[:-2]
        d2 = s2[1:-1] - s2[2:]
        ind = np.arange(1, len(s2) - 1)
        i1 = ind[np.logical_and(d1 < 0, d2 < 0)]
        if len(i1):
            i1 = i1[0]
        else:
            return

        ineg = ind[np.logical_and(d1 > 0, d2 > 0)]
        if len(ineg):
            i2 = ineg[ineg > i1]
            if len(i2):
                i2 = i2[0]
            else:
                return
        else:
            return

        y2 = (s2[i1] + s2[i2]) / 2.0
        x2 = ((len(s)-i1) + (len(s)-i2))/2.0

        m = (y2-y1)/(x2-x1)
        b = y1 - m*x1
        x = np.arange(len(s))
        os = m*x + b

        s3 = s-os
        s3 = s3/s3[0]
        xi = np.linspace(x[0],x[-1],1024)
        s4 = np.interp(xi, x, s3)

        return s4


def pkMetric(S, pklocs, n):

    if n > 1 and n < len(pklocs)-2:

        d11 = pklocs[n-1] - pklocs[n-2]
        d12 = pklocs[n] - pklocs[n-1]
        d1 = float(d11 + d12)/2.0

        d21 = pklocs[n + 2] - pklocs[n + 1]
        d22 = pklocs[n+1] - pklocs[n]
        d2 = float(d21 + d22) / 2.0

        return abs(d1-d2)/(d1+d2)

def pkAmpMetric(S, pklocs, n):

    if n > 1 and n < len(pklocs) - 2:
        S1 = S[pklocs[n-2]-10:pklocs[n-2]+10]
        S2 = S[pklocs[n-1]-10:pklocs[n-1]+10]
        S3 = S[pklocs[n+1]-10:pklocs[n+1]+10]
        S4 = S[pklocs[n+2]-10:pklocs[n+2]+10]
        if len(S1)>0 and len(S2)>0 and len(S3)>0 and len(S4)>0:
            a1 = S[pklocs[n-2]]-np.min(S1)
            a2 = S[pklocs[n-1]]-np.min(S2)
            a4 = S[pklocs[n+1]]-np.min(S3)
            a5 = S[pklocs[n+2]]-np.min(S4)
            a12 = (a1+a2)/2.0
            a45 = (a4+a5)/2.0
            return abs(a12-a45) / (a12 + a45)

def clusterPkMetric(met, amet, thresh, athresh, window=5):

    cls = np.zeros(len(met))
    cls[np.where(met <= thresh)] += 1.0
    cls[np.where(amet <= athresh)] += 1.0

    f = np.convolve(cls, np.ones(window), mode='same')/window
    res = np.zeros(len(met))
    res[np.where(f>1.5)] = 1

    return res

def classifyPeaks(S, pklocs, spacingThresh=0.07, AmpThresh=0.2, window=5):

    smet = [0.0, 0.0]
    amet = [0.0, 0.0]

    for i in range(2,len(pklocs)-2):
        smet.append(pkMetric(S, pklocs,i))
        amet.append(pkAmpMetric(S, pklocs, i))
    smet.append(0.0); smet.append(0.0)
    amet.append(0.0); amet.append(0.0)

    smet = np.array(smet)
    amet = np.array(amet)

    cls = clusterPkMetric(smet, amet, spacingThresh, AmpThresh, window)
    return cls

