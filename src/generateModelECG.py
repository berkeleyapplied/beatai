import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

def applyFitConstraints(p, ind):
    qrs_sig, qpk, qr_int, spk, rs_int = (p[0], p[1], p[2], p[3], p[4])
    tpk, rt_int, tsig, tskew = (p[5], p[6], p[7], p[8])
    ppk, pr_int, psig = (p[9], p[10], p[11])

    if ind == 0:
        if qrs_sig < 1:
            qrs_sig = 1.0
        if qrs_sig > 30:
            qrs_sig = 30.0
        dp = qrs_sig - p[0]


    elif ind == 8:

        if tskew < 0.0:
            tskew = 0.0

        if tskew > 10.0:
            tskew = 10.0

        dp = tskew - p[8]

    else:
        dp = 0.0


    res = np.array([qrs_sig, qpk, qr_int, spk, rs_int, tpk, rt_int, tsig, tskew,ppk, pr_int, psig])
    res = res/np.linalg.norm(res)

    return res, dp



def fitECGModel(s, params):

    x = np.arange(1024)

    if type(params) == type(None):
        qrs_sig, qpk, spk = (20.0, 0.4, 0.4)
        tpk, rt_int, tsig, tskew = (0.05, 120., 70., 4.0)
        ppk, pr_int, psig = (0.1, 100, 20.0)
        p = np.array([qrs_sig, qpk, spk, tpk, rt_int, tsig, tskew, ppk, pr_int, psig])
    else:
        p = params

    # Good Fit
    # qrs_sig, qpk, spk = (6.0, 0.15,  0.1)
    # tpk, rt_int, tsig, tskew = (0.1, 160., 50., 2.0)
    # ppk, pr_int, psig = (0.04, 85., 15.0)


    # Current State
    m_0 = createModelNormalizedECG((s[0], s[-1]),
                                   (p[0], p[1], p[2]),
                                   (p[3], p[4], p[5], p[6]),
                                   (p[7], p[8], p[9]), 0.0)
    e_0 = np.sum(np.power(s-m_0, 2.0))/1024.


    # Derive Partials
    dedp = []
    fit = np.ones(len(p))
    fit[1] = 0
    fit[2] = 0

    for i in range(len(p)):
        dp = p[i]*0.01

        if fit[i]:
            p[i] += dp
        # p, ddp = applyFitConstraints(p, i)
        # dp += ddp

        m_t = createModelNormalizedECG((s[0], s[-1]),
                                       (p[0], p[1], p[2]),
                                       (p[3], p[4], p[5], p[6]),
                                       (p[7], p[8], p[9]),
                                       0.0)
        e_t = np.sum(np.power(s-m_t, 2.0))/1024.
        dedp.append((e_0-e_t))

        if fit[i]:
            p[i] -= dp

    dedp = np.array(dedp)
    dedp_n = np.linalg.norm(dedp)


    p += p*0.001*dedp/dedp_n

    plt.ion()
    fig, ax = plt.subplots(1,1)
    exitCriteria = False
    fitState = 0
    niter = 0
    while not exitCriteria:
        ax.cla()

        m = createModelNormalizedECG((s[0], s[-1]),
                                     (p[0], p[1], p[2]),
                                     (p[3], p[4], p[5], p[6]),
                                     (p[7], p[8], p[9]), 0.0)
        e_t = np.sum(np.power(s - m, 2.0)) / 1024.

        ax.plot(x,s)
        ax.plot(x,m)
        plt.pause(0.01)
        #plt.show()
        dedp = []
        e_last = e_t
        for i in range(len(p)):
            dp = p[i] * 0.02

            if fit[i] == 1:
                p[i] += dp

            m_t = createModelNormalizedECG((s[0], s[-1]),
                                           (p[0], p[1], p[2]),
                                           (p[3], p[4], p[5], p[6]),
                                           (p[7], p[8], p[9]), 0.0)

            e_t = np.sum(np.power(s - m_t, 2.0)) / 1024.
            dedp.append((e_last - e_t))

            if fit[i] == 1:
                p[i] -= dp

        dedp = np.array(dedp)
        dedp_last = dedp_n
        dedp_n = np.linalg.norm(dedp)
        print('QS: {} {}'.format(p[1],p[2]))
        print('DEDP2: {}'.format(dedp_n-dedp_last))
        dedp2 = dedp_n-dedp_last

        if dedp2 > 0.0 and niter>10:
            if fitState==0:
                niter = 0
                fit[1] = 1
                fit[2] = 1
                fit[0] = 1
                fit[3:] = 0
                fitState = 1
                print('Fitting QRS')



            # else:
            #     exitCriteria = True
            #     print('Finished')

        p += p * 0.01 * dedp / dedp_n
        niter += 1

    fig.close()
    return p





def createModelNormalizedECG(pkamp, qrs, tdata, pdata, noise_sig):

    p1, p2 = pkamp
    qrs_sig, qpk, spk = qrs
    tpk, rt_int, tsig, tskew = tdata
    ppk, pr_int, psig = pdata
    qr_int = 1.75*qrs_sig
    rs_int = 1.75*qrs_sig

    pksig = qrs_sig
    s = np.zeros(1024)
    x = np.arange(-512,512)
    s += np.exp(-1.0*np.power(x,2.0)/(2*pksig**2.0))
    s += p2*np.exp(-1.0*np.power((x-512),2.0)/(2*pksig**2.0))
    s += p1*np.exp(-1.0*np.power((x+512),2.0)/(2*pksig**2.0))

    s += -qpk * np.exp(-1.0*np.power((x+qr_int),2.0)/(2*pksig**2.0))
    s += -qpk * np.exp(-1.0*np.power((x+qr_int-512),2.0)/(2*pksig**2.0))

    s += -spk * np.exp(-1.0*np.power((x-rs_int),2.0)/(2*pksig**2.0))
    s += -spk * np.exp(-1.0*np.power((x-rs_int+512),2.0)/(2*pksig**2.0))

    s += ppk * np.exp(-1.0*np.power((x+pr_int),2.0)/(2*psig**2.0))
    s += ppk * np.exp(-1.0*np.power((x+pr_int-512),2.0)/(2*psig**2.0))


    sn = skewnorm.pdf(x, tskew, rt_int, tsig)
    s += sn/np.max(sn) * tpk
    sn = skewnorm.pdf(x, tskew, -512+rt_int, tsig)
    s += sn / np.max(sn) * tpk

    s += np.random.normal(0,noise_sig, 1024)

    return s/s[512]




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True, help='output csv file')
    parser.add_argument('-n', '--numrecords', required=False, type=int, help='number of records to process')


    args = parser.parse_args()
    run_signal(args.input, args.output, args.numrecords, args.filtertime)
