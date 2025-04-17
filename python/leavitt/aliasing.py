import os
import numpy as np

# From Peter Yoachim's cepheid fitting code

def fold(indates, period):
    ind = indates/period
    result = ind-np.floor(ind)
    return result

def period_simp(t, y, rr=None, min_sl=None, npts=None,
                      harms=None, length=None):
    # given a time series of points, find the period that minimizes the
    #      string length

    # t and y input
    time = t-np.min(t)
    if npts is None:
        npts = 1000
    np = len(y)
    if rr is None:
        rr = [np.min(time),np.max(time)]

    # set up the folds to check
    folds = np.arange(npts)/(npts-1)*(np.max(rr)-np.min(rr))+np.min(rr)

    # variable to hold the string length
    sl = np.zeros(npts)

    # fold up the light-curve and calc the string length, normally this
    #     would be better done with something like a FT, but this data is
    # usually irregularly sampled, so that makes FT harder


    for i in range(npts):
        folded = fold(time,folds[i])
        order = np.sort(folded)
        lc = y[order]                  # light curve ordered up
        lc2 = np.roll(lc,-1)           # pull the points back one spot
                                       # sl(i)=sum(abs(lc-lc2)) ;and there's
                                       # the string length!, Hooray for not
                                       # writing loops!
        f2 = np.roll(folded,-1)
        sl[i] = np.sum(np.abs(lc-lc2))

    min_sl = np.min(sl)
    # danger, can return a huge array if the user supplied range is larger
    #         than the actual range and it doesn't find a good period

    # Check if the minimum is a harmonic rather than the fundamental period
    
    nh = 10  # number of harmonics to check
    bf = np.max(folds[np.where(sl == np.min(min_sl))[0]])  # best_fit so far
    folds = bf/(np.arange(nh)+1)
    good, = np.where(folds >= np.min(rr))
    nh = len(good)
    folds = folds[good]
    sl = np.zeros(nh)
    for i in range(nh):
        folded = fold(time,folds[i])
        order = np.sort(folded)
        lc = y[order]            # light curve ordered up
        lc2 = np.roll(lc,-1)     # pull the points back one spot
                                 # sl(i)=sum(abs(lc-lc2)) ;and there's
                                 # the string length!, Hooray for not
                                 # writing loops!
        f2 = np.roll(folded,-1)
        sl[i] = np.sum(np.abs(lc-lc2))
        min_sl = np.min(sl)


    bestind, = np.where(sl == np.min(min_sl))
    length = sl[bestind]

    return bestind


