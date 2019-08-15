import numpy as np

CBDFiltCoeffSHF = [302, 2052, 3892, 3892, 3892, 2052, 302]
SHFPreFilterDelay = 123
SHFResultFilterDelay = 19

CBDFiltCoeffSLF = [6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 13, 13, 14, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 37, 38, 39, 41, 42, 43, 45, 46, 48, 49, 51, 52, 54, 55, 57, 58, 60, 61, 63, 64, 66, 68, 69, 71, 72, 74, 76, 77, 79, 80, 82, 83, 85, 87, 88, 90, 91, 93, 94, 96, 97, 98, 100, 101, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 122, 123, 124, 125, 125, 126, 126, 127, 127, 128, 128, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 130, 130, 129, 129, 129, 129, 128, 128, 127, 127, 126, 126, 125, 125, 124, 123, 122, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 110, 109, 108, 107, 105, 104, 103, 101, 100, 98, 97, 96, 94, 93, 91, 90, 88, 87, 85, 83, 82, 80, 79, 77, 76, 74, 72, 71, 69, 68, 66, 64, 63, 61, 60, 58, 57, 55, 54, 52, 51, 49, 48, 46, 45, 43, 42, 41, 39, 38, 37, 35, 34, 33, 32, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 17, 16, 15, 14, 14, 13, 13, 12, 11, 11, 10, 10, 10, 9, 9, 8, 8, 8, 8, 7, 7, 7, 6, 6, 6, 6]
SLFPreFilterDelay = 0
SLFTrendFilterDelay = 19


CBDFiltCoeffHF = [40, 117, 278, 562, 973, 1440, 1822, 1971, 1972, 1971, 1822, 1440, 973, 562, 278, 117, 40]
syncHFPreFilterDelay = 20-8


CBDFiltCoeffLF = [4, 9, 15, 25, 38, 58, 86, 123, 171, 232, 305, 389, 483, 582, 682, 777, 859, 923, 964, 978, 978, 978, 964, 923, 859, 777, 682, 582, 483, 389, 305, 232, 171, 123, 86, 58, 38, 25, 15, 9, 4]
syncLFPreFilterDelay = 20 - 20

HilbertCoef = [56, 0, 96, 0, 221, 0, 463, 0, 879, 0, 1611, 0, 3182, 0, 10360, 0,	-10360, 0, -3182, 0, -1611, 0, -879, 0,	-463, 0, -221, 0, -96, 0, -56]
syncHilbertDelay = 140-140
syncHilbertResultDelay = 144 - 140
HilbertPreSpeedDelay = 140 - 125

syncResultsSAngleDelay = 144-144

syncAfterHilbertPreAmpSpeedCompDelay = 35 - 35
syncBeforeHilbertPreAmpSpeedCompDelay = 35 - 20

def hilbertFilterFunction():
    return firFilter(HilbertCoef)

class firFilter:
    def __init__(self, mask, scale=float(1<<14)):
        self.mask = mask
        self.queue = []
        self.scale = scale

    def reset(self):
        self.queue = []

    def addSample(self, s):
        if len(self.queue) < len(self.mask):
            self.queue.append(s)
        else:
            self.queue[0:-1] = self.queue[1:]
            self.queue[-1] = s

    def result(self):
        a = np.array(self.mask[:len(self.queue)])
        b = np.array(self.queue)
        return np.sum(a*b)/self.scale

class filterDiff:
    def __init__(self, size):
        self.queue = []
        self.size = size

    def addSample(self, s):
        if len(self.queue) < self.size:
            self.queue.append(s)
        else:
            self.queue[0:-1] = self.queue[1:]
            self.queue[-1] = s

    def result(self):
        return self.queue[-1]-self.queue[0]


class delayFilter:
    def __init__(self, delay):
        self.delay = delay
        self.queue = []

    def addSample(self, s):
        if len(self.queue) < self.delay+1:
            self.queue.append(s)
        else:
            self.queue[0:-1] = self.queue[1:]
            self.queue[-1] = s

    def result(self):
        if len(self.queue) < self.delay+1:
            return 0
        else:
            return self.queue[0]

class results:

    def __init__(self):
        self.data = []
        self.sAngle = []
        self.trend = []
        self.spAmp = []
        self.hf = []
        self.hhf = []
        self.ampPhase = []


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from mit212 import parse212File

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputfile', required=True, help='File Input')
    parser.add_argument('-n', '--numSamples', type=int, required=False, default=None, help='Number of samples to process')
    parser.add_argument('-s', '--startSample', type=int, required=False, default=None, help='Starting point sample number')
    parser.add_argument('-p', '--noisePercent', type=float, required=False, default=None, help='Noise Percentage')
    parser.add_argument('-z', '--zoom', type=float, required=False, default=500., help='Zoom extent for Hilbert plots')
    parser.add_argument('--animate', action='store_true')

    args = parser.parse_args()
    plot_ext = args.zoom

    S1, S2 = parse212File(args.inputfile)

    SHF = firFilter(CBDFiltCoeffSHF)
    SHFPreFilter = delayFilter(SHFPreFilterDelay)
    SHFResultFilter = delayFilter(SHFResultFilterDelay)

    SLF = firFilter(CBDFiltCoeffSLF)
    SLFPreFilter = delayFilter(SLFPreFilterDelay)
    SLFTrendFilter = delayFilter(SLFTrendFilterDelay)

    filterHilbertTop = firFilter(HilbertCoef)
    syncAfterHilbertPreSpeedCompFilter = delayFilter(syncHilbertDelay)
    syncResultHFFilter = delayFilter(syncHilbertResultDelay)
    syncResultHHFFilter = delayFilter(syncHilbertResultDelay)
    syncBeforeHilbertPreSpeedCompFilter = delayFilter(HilbertPreSpeedDelay)

    speedCompImFilter = filterDiff(9)
    speedCompReFilter = filterDiff(9)

    syncResultsSAngleFilter = delayFilter(syncResultsSAngleDelay)


    HF = firFilter(CBDFiltCoeffHF)
    syncHFPreFilterDiff = delayFilter(syncHFPreFilterDelay)

    LF = firFilter(CBDFiltCoeffLF)
    syncLFPreFilterDiff = delayFilter(syncLFPreFilterDelay)

    filterHilbertBottom = firFilter(HilbertCoef)

    syncAfterHilbertPreAmpSpeedCompFilter = delayFilter(syncAfterHilbertPreAmpSpeedCompDelay)
    syncBeforeHilbertPreAmpSpeedCompFilter = delayFilter(syncBeforeHilbertPreAmpSpeedCompDelay)

    ampSpeedCompReFilter = filterDiff(9)
    ampSpeedCompImFilter = filterDiff(9)
    ampSpeedDTDelayFilter = delayFilter(4)
    ampSpeedHDTDelayFilter = delayFilter(4)
    syncResultsSPAmpFilter = delayFilter(144-39)

    SHF_res = []
    SLF_res = []
    SHF_SLF_res = []
    SHF_SLF_hil = []
    HilbertRe = []
    HilbertIm = []
    HilbertRe2 = []
    HilbertIm2 = []
    speed = []
    sangle = []
    n = 0

    res1 = results()
    res2 = results()

    nsamp = len(S1)
    if args.numSamples:
        nsamp = int(args.numSamples)

    nstart = 0
    if args.startSample:
        nstart = args.startSample

    nend = nstart+nsamp
    if nend > len(S1):
        nend = len(S1)

    S1 = np.array(S1[nstart:nend]).astype(float)
    S2 = np.array(S2[nstart:nend]).astype(float)

    if args.noisePercent:
        sig = abs(np.mean(S1))*args.noisePercent/100.0
        noise = np.random.normal(0.0, sig, len(S1))
        S1 += noise
        noise = np.random.normal(0.0, sig, len(S2))
        S2 += noise


    for sample in S1:
        n+=1

        if args.numSamples:
            if n > args.numSamples:
                break

        SHF.addSample(sample)
        SHFPreFilter.addSample(SHF.result()); SHF_res.append(SHFPreFilter.result())
        SHFResultFilter.addSample(SHFPreFilter.result())

        SLF.addSample(sample)
        SLFPreFilter.addSample(SLF.result()); SLF_res.append(SLFPreFilter.result())
        SLFTrendFilter.addSample(SLFPreFilter.result())

        SHF_SLF = SHFPreFilter.result() - SLFPreFilter.result()
        SHF_SLF_res.append(SHF_SLF)
        filterHilbertTop.addSample(SHF_SLF); SHF_SLF_hil.append(filterHilbertTop.result())

        syncAfterHilbertPreSpeedCompFilter.addSample(filterHilbertTop.result())
        syncResultHHFFilter.addSample(syncAfterHilbertPreSpeedCompFilter.result())

        syncBeforeHilbertPreSpeedCompFilter.addSample(SHF_SLF)
        syncResultHFFilter.addSample(syncBeforeHilbertPreSpeedCompFilter.result())

        speedCompReFilter.addSample(syncBeforeHilbertPreSpeedCompFilter.result())
        speedCompImFilter.addSample(syncAfterHilbertPreSpeedCompFilter.result())

        spd = np.sqrt(speedCompReFilter.result()**2.0 + speedCompImFilter.result()**2.0)
        speed.append(spd)

        if spd > 0:
            sa = 1000*speedCompImFilter.result()/spd
        else:
            sa = 0.0

        sangle.append(sa)

        HilbertRe.append(speedCompReFilter.result())
        HilbertIm.append(speedCompImFilter.result())

        syncResultsSAngleFilter.addSample(sa)


        # Lower
        HF.addSample(sample)
        syncHFPreFilterDiff.addSample(HF.result())

        LF.addSample(sample)
        syncLFPreFilterDiff.addSample(LF.result())

        HF_LF = syncHFPreFilterDiff.result()-syncLFPreFilterDiff.result()

        filterHilbertBottom.addSample(HF_LF)
        syncAfterHilbertPreAmpSpeedCompFilter.addSample(filterHilbertBottom.result())

        syncBeforeHilbertPreAmpSpeedCompFilter.addSample(HF_LF)

        ampSpeedCompReFilter.addSample(syncBeforeHilbertPreAmpSpeedCompFilter.result())
        HilbertRe2.append(ampSpeedCompReFilter.result())
        ampSpeedCompImFilter.addSample(syncAfterHilbertPreAmpSpeedCompFilter.result())
        HilbertIm2.append(ampSpeedCompImFilter.result())

        ampSpeedDTDelayFilter.addSample(syncBeforeHilbertPreAmpSpeedCompFilter.result())
        ampSpeedHDTDelayFilter.addSample(syncAfterHilbertPreAmpSpeedCompFilter.result())

        spd2 = np.sqrt(ampSpeedCompReFilter.result()**2.0 + ampSpeedCompImFilter.result()**2.0)
        amplitude = np.sqrt(ampSpeedDTDelayFilter.result()**2.0 + ampSpeedHDTDelayFilter.result()**2.0)
        spAmp = spd2 * amplitude

        syncResultsSPAmpFilter.addSample(spAmp)

        ampPhase = syncResultsSAngleFilter.result() * (
            syncResultHFFilter.result()/10. + syncResultHHFFilter.result()/10.
        ) / 10.0

        # load results
        res1.data.append(SHFResultFilter.result())
        res1.sAngle.append(syncResultsSAngleFilter.result())
        res1.trend.append(SLFTrendFilter.result())
        res1.spAmp.append(syncResultsSPAmpFilter.result())
        res1.hf.append(syncResultHFFilter.result())
        res1.hhf.append(syncResultHHFFilter.result())
        res1.ampPhase.append(ampPhase)

    SHF_res1 = np.array(SHF_res)
    SLF_res1 = np.array(SLF_res)
    SHF_SLF_res1 = np.array(SHF_SLF_res)
    SHF_SLF_hil1 = np.array(SHF_SLF_hil)
    t1 = np.arange(len(S2)).astype(float)/250.
    HilbertRe_1 = np.array(HilbertRe)
    HilbertIm_1 = np.array(HilbertIm)
    HilbertRe2_1 = np.array(HilbertRe2)
    HilbertIm2_1 = np.array(HilbertIm2)

    #### Channel 2
    SHF_2 = firFilter(CBDFiltCoeffSHF)
    SHFPreFilter_2 = delayFilter(SHFPreFilterDelay)
    SHFResultFilter_2 = delayFilter(SHFResultFilterDelay)

    SLF_2 = firFilter(CBDFiltCoeffSLF)
    SLFPreFilter_2 = delayFilter(SLFPreFilterDelay)
    SLFTrendFilter_2 = delayFilter(SLFTrendFilterDelay)

    filterHilbertTop_2 = firFilter(HilbertCoef)
    syncAfterHilbertPreSpeedCompFilter_2 = delayFilter(syncHilbertDelay)
    syncResultHFFilter_2 = delayFilter(syncHilbertResultDelay)
    syncResultHHFFilter_2 = delayFilter(syncHilbertResultDelay)
    syncBeforeHilbertPreSpeedCompFilter_2 = delayFilter(HilbertPreSpeedDelay)

    speedCompImFilter_2 = filterDiff(9)
    speedCompReFilter_2 = filterDiff(9)

    syncResultsSAngleFilter_2 = delayFilter(syncResultsSAngleDelay)

    HF_2 = firFilter(CBDFiltCoeffHF)
    syncHFPreFilterDiff_2 = delayFilter(syncHFPreFilterDelay)

    LF_2 = firFilter(CBDFiltCoeffLF)
    syncLFPreFilterDiff_2 = delayFilter(syncLFPreFilterDelay)

    filterHilbertBottom_2 = firFilter(HilbertCoef)

    syncAfterHilbertPreAmpSpeedCompFilter_2 = delayFilter(syncAfterHilbertPreAmpSpeedCompDelay)
    syncBeforeHilbertPreAmpSpeedCompFilter_2 = delayFilter(syncBeforeHilbertPreAmpSpeedCompDelay)

    ampSpeedCompReFilter_2 = filterDiff(9)
    ampSpeedCompImFilter_2 = filterDiff(9)
    ampSpeedDTDelayFilter_2 = delayFilter(4)
    ampSpeedHDTDelayFilter_2 = delayFilter(4)
    syncResultsSPAmpFilter_2 = delayFilter(144 - 39)



    SHF_res_2 = []
    SLF_res_2 = []
    SHF_SLF_res_2 = []
    SHF_SLF_hil_2 = []
    HilbertRe_2 = []
    HilbertIm_2 = []
    HilbertRe2_2 = []
    HilbertIm2_2 = []
    speed_2 = []
    sangle_2 = []
    n = 0

    for sample in S2:
        n+=1

        if args.numSamples:
            if n > args.numSamples:
                break

        SHF_2.addSample(sample);
        SHFPreFilter_2.addSample(SHF_2.result()); SHF_res_2.append(SHFPreFilter_2.result())
        SHFResultFilter_2.addSample(SHFPreFilter_2.result())

        SLF_2.addSample(sample);
        SLFPreFilter_2.addSample(SLF_2.result()); SLF_res_2.append(SLFPreFilter_2.result())
        SLFTrendFilter_2.addSample(SLFPreFilter_2.result())

        SHF_SLF_2 = SHFPreFilter_2.result() - SLFPreFilter_2.result()
        SHF_SLF_res_2.append(SHF_SLF_2)
        filterHilbertTop_2.addSample(SHF_SLF_2); SHF_SLF_hil_2.append(filterHilbertTop_2.result())

        syncAfterHilbertPreSpeedCompFilter_2.addSample(filterHilbertTop_2.result())
        syncResultHHFFilter_2.addSample(syncAfterHilbertPreSpeedCompFilter_2.result())

        syncBeforeHilbertPreSpeedCompFilter_2.addSample(SHF_SLF_2)
        syncResultHFFilter_2.addSample(syncBeforeHilbertPreSpeedCompFilter_2.result())

        speedCompReFilter_2.addSample(syncBeforeHilbertPreSpeedCompFilter_2.result())
        speedCompImFilter_2.addSample(syncAfterHilbertPreSpeedCompFilter_2.result())

        spd_2= np.sqrt(speedCompReFilter_2.result()**2.0 + speedCompImFilter_2.result()**2.0)
        speed.append(spd)

        if spd_2 > 0:
            sa_2 = 1000*speedCompImFilter_2.result()/spd_2
        else:
            sa_2 = 0.0

        sangle_2.append(sa_2)

        HilbertRe_2.append(speedCompReFilter_2.result())
        HilbertIm_2.append(speedCompImFilter_2.result())

        syncResultsSAngleFilter_2.addSample(sa_2)


        # Lower
        HF_2.addSample(sample)
        syncHFPreFilterDiff_2.addSample(HF_2.result())

        LF_2.addSample(sample)
        syncLFPreFilterDiff_2.addSample(LF_2.result())

        HF_LF_2 = syncHFPreFilterDiff_2.result()-syncLFPreFilterDiff_2.result()

        filterHilbertBottom_2.addSample(HF_LF_2)
        syncAfterHilbertPreAmpSpeedCompFilter_2.addSample(filterHilbertBottom_2.result())

        syncBeforeHilbertPreAmpSpeedCompFilter_2.addSample(HF_LF_2)

        ampSpeedCompReFilter_2.addSample(syncBeforeHilbertPreAmpSpeedCompFilter_2.result())
        HilbertRe2_2.append(ampSpeedCompReFilter_2.result())
        ampSpeedCompImFilter_2.addSample(syncAfterHilbertPreAmpSpeedCompFilter_2.result())
        HilbertIm2_2.append(ampSpeedCompImFilter_2.result())

        ampSpeedDTDelayFilter_2.addSample(syncBeforeHilbertPreAmpSpeedCompFilter_2.result())
        ampSpeedHDTDelayFilter_2.addSample(syncAfterHilbertPreAmpSpeedCompFilter_2.result())

        spd2_2 = np.sqrt(ampSpeedCompReFilter_2.result()**2.0 + ampSpeedCompImFilter_2.result()**2.0)
        amplitude_2 = np.sqrt(ampSpeedDTDelayFilter_2.result()**2.0 + ampSpeedHDTDelayFilter_2.result()**2.0)
        spAmp_2 = spd2_2 * amplitude_2

        syncResultsSPAmpFilter_2.addSample(spAmp_2)

        ampPhase_2 = syncResultsSAngleFilter_2.result() * (
            syncResultHFFilter_2.result()/10. + syncResultHHFFilter_2.result()/10.
        ) / 10.0

        # load results
        res2.data.append(SHFResultFilter_2.result())
        res2.sAngle.append(syncResultsSAngleFilter_2.result())
        res2.trend.append(SLFTrendFilter_2.result())
        res2.spAmp.append(syncResultsSPAmpFilter_2.result())
        res2.hf.append(syncResultHFFilter_2.result())
        res2.hhf.append(syncResultHHFFilter_2.result())
        res2.ampPhase.append(ampPhase_2)



    SHF_res2 = np.array(SHF_res)
    SLF_res2 = np.array(SLF_res)
    SHF_SLF_res2 = np.array(SHF_SLF_res)
    SHF_SLF_hil2 = np.array(SHF_SLF_hil)
    t2 = np.arange(len(S2)).astype(float)/250.
    HilbertRe_2 = np.array(HilbertRe)
    HilbertIm_2 = np.array(HilbertIm)
    HilbertRe2_2 = np.array(HilbertRe2)
    HilbertIm2_2 = np.array(HilbertIm2)

    if not args.animate:

        fig, ax= plt.subplots(3,2,sharex=True)
        ax[0][0].plot(t1,S1, 'b')
        ax[0][1].plot(t2,S2 + max(S1)*1.1, 'r')
        ax[0][0].set_title('Signal: S1')
        ax[0][1].set_title('Signal: S2')
        ax[0][0].set_ylabel('Raw Signal')

        ax[1][0].plot(t1,SHF_res1,'b')
        ax[1][0].plot(t1,SLF_res1,'b')
        ax[1][1].plot(t2,SHF_res2+max(SHF_res1)*1.1,'r')
        ax[1][1].plot(t2,SLF_res2+max(SHF_res1)*1.1,'r')
        ax[1][0].set_ylabel('HF and LF Signals')

        ax[2][0].plot(t1,SHF_SLF_res1,'b')
        ax[2][1].plot(t2,SHF_SLF_res2+max(SHF_SLF_res1)*1.1,'r')
        ax[2][0].set_ylabel('Filtered Signal SHF-SLF')



        fig2, (ax21, ax22) = plt.subplots(1,2)
        ax21.plot(HilbertRe,HilbertIm,'b.',markersize=1.0)
        ax21.set_title('Upper Hilbert Space')
        ax22.plot(HilbertRe2,HilbertIm2,'b.',markersize=1.0)
        ax22.set_title('Lower Hilbert Space')

        plt.show()

    else:
        dtwin = 1.0
        Hz = 360
        nwin = int(Hz*dtwin)
        nhilbert = 50

        plt.ion()
        fig, ax = plt.subplots(2,4)
        line1, = ax[0][0].plot(t1[:nwin],SHF_SLF_res1[:nwin],'b')
        line2, = ax[1][0].plot(t2[:nwin],SHF_SLF_res2[:nwin],'b')

        line3, = ax[0][1].plot(HilbertRe_1[nwin-50:nwin],HilbertIm_1[nwin-nhilbert:nwin],'b')
        line4, = ax[0][2].plot(HilbertRe2_1[nwin-50:nwin],HilbertIm2_1[nwin-nhilbert:nwin],'b')
        ax[0][1].set_xlim(-1*plot_ext,plot_ext)
        ax[0][1].set_ylim(-plot_ext,plot_ext)
        ax[0][2].set_xlim(-plot_ext,plot_ext)
        ax[0][2].set_ylim(-plot_ext,plot_ext)

        line5, = ax[1][1].plot(HilbertRe_2[nwin - 50:nwin], HilbertIm_2[nwin - nhilbert:nwin], 'b')
        line6, = ax[1][2].plot(HilbertRe2_2[nwin - 50:nwin], HilbertIm2_2[nwin - nhilbert:nwin], 'b')
        ax[1][1].set_xlim(-plot_ext, plot_ext)
        ax[1][1].set_ylim(-plot_ext, plot_ext)
        ax[1][2].set_xlim(-plot_ext, plot_ext)
        ax[1][2].set_ylim(-plot_ext, plot_ext)

        line7, = ax[0][3].plot(t1[:nwin], res1.spAmp[:nwin], 'b')
        line8, = ax[1][3].plot(t2[:nwin], res2.spAmp[:nwin], 'b')

        for n in range(len(t1)-nwin):
            line1.set_ydata(SHF_SLF_res1[n:n+nwin])
            line2.set_ydata(SHF_SLF_res2[n:n+nwin])

            line3.set_xdata(HilbertRe_1[n+nwin-nhilbert:n+nwin])
            line3.set_ydata(HilbertIm_1[n+nwin-nhilbert:n+nwin])
            line4.set_xdata(HilbertRe2_1[n+nwin-nhilbert:n+nwin])
            line4.set_ydata(HilbertIm2_1[n+nwin-nhilbert:n+nwin])

            line5.set_xdata(HilbertRe_2[n + nwin - nhilbert:n + nwin])
            line5.set_ydata(HilbertIm_2[n + nwin - nhilbert:n + nwin])
            line6.set_xdata(HilbertRe2_2[n + nwin - nhilbert:n + nwin])
            line6.set_ydata(HilbertIm2_2[n + nwin - nhilbert:n + nwin])

            line7.set_ydata(res1.spAmp[n:n + nwin])
            line8.set_ydata(res2.spAmp[n:n + nwin])
            ax[0][3].set_ylim(min(res1.spAmp[n:n + nwin]),max(res1.spAmp[n:n + nwin]))
            ax[1][3].set_ylim(min(res2.spAmp[n:n + nwin]),max(res2.spAmp[n:n + nwin]))

            plt.pause(0.001)
