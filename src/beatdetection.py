from scipy.signal import butter, filtfilt, lfilter
import matplotlib.pyplot as plt
import numpy as np

def findPeakLocations(S):

    SamPerSec = 250

    ######### Band Pass Filter  ##################################
    lowCut = .5  # Low pass in Hz
    highCut = 15  # High pass in Hz
    filterOrder = 1
    nyFreq = 0.5 * SamPerSec
    low = lowCut / nyFreq
    high = highCut / nyFreq
    b, a = butter(filterOrder, [low, high], btype="band", analog=False)
    filterSig = lfilter(b, a, S)
    ###############################################################

    ######### Take Derivative and Square Signal  #################
    filterSig[:10] = filterSig[10]  # Clean the beginning to avoid abrupt start of signal
    derivFilterSig = np.ediff1d(filterSig)  # Take derivative of bandpass filtered signal
    derivFilterSig2 = derivFilterSig ** 2  # Square Derivative Signal
    ##############################################################

    ######## Apply Smoothing Filter on Squared Signal ##############
    window = 15  # Window width in number of samples
    windowFilterSig = np.convolve(derivFilterSig2, np.ones(window))
    ################################################################

    ############  Find Peaks in Window Filtered Signal ####################
    peakLocations = np.empty(windowFilterSig.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(windowFilterSig.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(windowFilterSig.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays
    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = windowFilterSig.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if windowFilterSig[i - 1] < windowFilterSig[i]:
            i_ahead = i + 1  # Index to look ahead of current sample
            # Find next sample that is unequal to windowFilterSig[i]
            while i_ahead < i_max and windowFilterSig[i_ahead] == windowFilterSig[i]:
                i_ahead += 1

            # Maxima is found if next unequal sample is smaller than windowFilterSig[i]
            if windowFilterSig[i_ahead] < windowFilterSig[i]:
                left_edges[m] = i
                right_edges[m] = i_ahead - 1
                peakLocations[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                # Skip samples that can't be maximum
                i = i_ahead
        i += 1
    peakLocations.resize(m, refcheck=False)
    left_edges.resize(m, refcheck=False)
    right_edges.resize(m, refcheck=False)

    peaks_size = peakLocations.shape[0]
    priority = windowFilterSig[peakLocations]
    distance = 25  # Used to limit the number of noise peaks returned
    keep = np.ones(peaks_size, dtype=np.uint8)  # Prepare array of flags
    priority_to_position = np.argsort(priority)

    ## Highest priority first -> iterate in reverse order (decreasing)
    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if keep[j] == 0:
            # Skip evaluation for peak already marked as "don't keep"
            continue
        k = j - 1
        # Flag "earlier" peaks for removal until minimal distance is exceeded
        while 0 <= k and peakLocations[j] - peakLocations[k] < distance:
            keep[k] = 0
            k -= 1
        k = j + 1
        # Flag "later" peaks for removal until minimal distance is exceeded
        while k < peaks_size and peakLocations[k] - peakLocations[j] < distance:
            keep[k] = 0
            k += 1
    keep = keep.view(dtype=np.bool)  # Convert 1's and 0's to Boolean array
    peakLocations = peakLocations[keep]  # Array of peak locations (sample #) in windowFilterSig
    peakHeights = windowFilterSig[peakLocations]  # Array of peak amplitudes at each returned peak location

    if len(peakLocations) == 0:
        exit('No Peaks Found')

        ########## Classify Found Peaks as QRS or Noise ##################################

    refracPeriodMax = 55  # Minimum number of samples between QRS labeled peaks
    qrsFiltFactor = 0.125  # Adaptive Threshold Parameter
    noisePeakFactor = 0.125  # Adaptive Threshold Parameter
    qrsNoiseDif = 0.2  # Adaptive Threshold Parameter
    numPeaks = len(peakLocations)  # Number of Peaks to Iterate through
    qrsPeakVal = 0.0  # Initialize placeholder parameter
    noisePeakVal = 0.0  # Initialize placeholder parameter
    lastQRSindex = 0  # Initialize placeholder parameter
    qrsPeaks = []  # Initialize Array for QRS Peaks
    noisePeaks = []  # Initialize Array for noise Peaks
    warmUp = 500  # Duration of initial warm up period in samples, Used to set initial parameters

    if peakLocations[0] > warmUp:  # Eval Start of signal to get initial parameters
        initThres = 1.25 * (np.average(peakHeights))  # No Peak in WarmUp Period, Defaults to 1.25*average Peak Value
        initPeakLocation = 0
    else:
        initPeakLocation = np.max(np.where(peakLocations < warmUp))
        if initPeakLocation == 0:
            initThres = 1.25 * (np.average(peakHeights))  # 1 peak in WarmUp, Defualts to 1.25*average Peak Value
        else:
            initPeakMax = np.max(
                peakHeights[:initPeakLocation])  # Good warm up period, multiple peaks found, use 0.25*max
            initThres = 0.25 * initPeakMax

    thres = initThres
    for x in range(numPeaks):
        if x <= initPeakLocation:  ### Hold the refac period and thres fixed in warm up
            refracPeriod = 0
            thres = initThres
        else:
            refracPeriod = refracPeriodMax

        if (peakLocations[x] - lastQRSindex) >= refracPeriod:
            if peakHeights[x] > thres:
                qrsPeaks = np.append(qrsPeaks, peakLocations[x])
                qrsPeakVal = qrsFiltFactor * peakHeights[x] + (1 - qrsFiltFactor) * qrsPeakVal
                lastQRSindex = peakLocations[x]
            else:
                noisePeaks = np.append(noisePeaks, peakLocations[x])
                noisePeakVal = noisePeakFactor * peakHeights[x] + (1 - noisePeakFactor) * noisePeakVal
            thres = noisePeakVal + qrsNoiseDif * (qrsPeakVal - noisePeakVal)

            if thres > initThres:
                thres = initThres

    qrsPeaks = qrsPeaks.astype(int)
    noisePeaks = noisePeaks.astype(int)

    ########## Find sample location of QRS peaks in the original S signal

    for x in range(len(qrsPeaks)):
        win = S[(qrsPeaks[x] - window):qrsPeaks[x]]
        winFilt = filterSig[(qrsPeaks[x] - window):qrsPeaks[x]]
        filtMax = max(winFilt)
        filtMin = min(winFilt)
        if filtMax > abs(filtMin):
            tmp_max_ind = np.argmax(win)
        else:
            tmp_max_ind = np.argmin(win)
        qrsPeaks[x] = (qrsPeaks[x] - window) + tmp_max_ind

    for x in range(len(noisePeaks)):
        win = S[(noisePeaks[x] - window):noisePeaks[x]]
        tmp_max_ind = np.argmax(win)
        noisePeaks[x] = (noisePeaks[x] - window) + tmp_max_ind

    ###################################################################
    # print('Called BeatDetection')
    # fig, ax = plt.subplots(figsize=(12, 6))
    # plt.plot(S)
    # plt.plot(filterSig)
    # plt.plot(qrsPeaks, S[qrsPeaks], "o", color='lime')
    # plt.show()



    return qrsPeaks