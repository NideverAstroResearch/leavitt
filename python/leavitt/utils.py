#!/usr/bin/env python
#
# DLNPYUTILS.PY - Utility functions.
#

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@noao.edu>'
__version__ = '20180823'  # yyyymmdd

import re
import logging
import os
import sys
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table, Column
from astropy import modeling
from astropy.convolution import Gaussian1DKernel, convolve
from glob import glob
from scipy.signal import medfilt
from scipy.ndimage.filters import median_filter,gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.linalg import svd
#from astropy.utils.exceptions import AstropyWarning
#import socket
#from scipy.signal import convolve2d
#from scipy.ndimage.filters import convolve
import astropy.stats
import matplotlib.pyplot as plt


# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

def datadir():
    """ Return the data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

# Size, number of elements
def size(a=None):
    """Returns the number of elements"""
    if a is None: return 0
    return np.array(a,ndmin=1).size

# Median Absolute Deviation
def mad(data, axis=None, func=None, ignore_nan=True):
    """ Calculate the median absolute deviation."""
    if type(data) is not np.ndarray: raise ValueError("data must be a numpy array")    
    return 1.4826 * astropy.stats.median_absolute_deviation(data,axis=axis,func=func,ignore_nan=ignore_nan)

def minmax(a):
    """ Return a 2-element array of minimum and maximum."""
    if type(a) is not np.ndarray: raise ValueError("a must be a numpy array")
    return np.array([np.min(a),np.max(a)])

def stat(a=None,silent=False):
    """ Returns basic statistics on an array."""
    if a is None: raise ValueError("a must be input")
    if type(a) is not np.ndarray: raise ValueError("a must be a numpy array")
    #  This is what stat returns:
    #  info[0]: Number of Elements
    #  info[1]: Minimum
    #  info[2]: Maximum
    #  info[3]: Range
    #  info[4]: Mean
    #  info[5]: Median
    #  info[6]: Standard Deviation
    #  info[7]: Standard Error
    #  info[8]: Root Mean Square (R.M.S.)
    #  info[9]: MAD estimate of St.Dev.
    info = np.zeros(10,float)
    info[0] = len(a)
    info[1] = np.min(a)
    info[2] = np.max(a)
    info[3] = info[2]-info[1]
    info[4] = np.mean(a)
    if info[0] > 1:
        info[5] = np.median(a)
        info[6] = np.std(a,ddof=1)               # std. dev.
        info[7] = info[6]/np.sqrt(info[0])       # std. err.
        info[8] = np.sqrt(np.sum(a**2)/info[0])  # RMS
        info[9] = mad(a)                         # MAD
    if silent is not True:
        print('----------------------')
        print('elements = %d' % info[0])
        print('minimum  = %f' % info[1])
        print('maximum  = %f' % info[2])
        print('range    = %f' % info[3])
        print('mean     = %f' % info[4])
        print('median   = %f' % info[5])
        print('st. dev. = %f' % info[6])
        print('st. err. = %f' % info[7])
        print('r.m.s.   = %f' % info[8])
        print('mad s.d. = %f' % info[9])
        print('----------------------')
    return

def where(statement,comp=False):
    """ Wrapper around numpy.where() to be more like IDL"""
    # If comp=True then the complement will be returned
    gd, = np.where(statement)
    ngd = len(gd)
    if comp:
        bd, = np.where(~statement)
        nbd = len(bd)
        return gd,ngd,bd,nbd
    else:
        return gd,ngd

def strlen(lst=None):
    """ Calculate the string lengths of a string array."""
    if lst is None: raise ValueError("lst must be input")
    n = size(lst)
    out = np.zeros(n,int)
    for i,a in enumerate(np.array(lst,ndmin=1)):
        out[i] = len(a)
    if n==1: out=int(out)
    return out


def strip(lst=None,chars=None):
    """ Strip on a scalar or list."""
    if lst is None: raise ValueError("lst must be input")
    if type(lst) is str: return lst.strip(chars)
    return [o.strip(chars) for o in np.array(lst,ndmin=1)]


def strjoin(a=None,b=None,c=None,sep=None):
    """ Join two string lists/arrays or scalars"""
    if (a is None) | (b is None): raise ValueError("a and b must be input")
    na = size(a)
    nb = size(b)
    nc = size(c)
    if sep is None: sep=''
    n = np.max([na,nb,nc])
    len1 = strlen(a)
    t1 = type(a)
    len2 = strlen(b)
    t2 = type(b)
    if nc>0:
        len3 = strlen(c)
        t3 = type(c)
    else:
        len3 = 0
        t3 = t2
    nlen = np.max(len1)+np.max(len2)+np.max(len3)+len(sep)
    out = np.zeros(n,(np.str,nlen))
    for i in range(n):
        if na>1:
            a1 = a[i]
        else:
            a1 = np.array(a,ndmin=1)[0]
        arr = tuple(a1)
        if nb>1:
            b1 = b[i]
        else:
            b1 = np.array(b,ndmin=1)[0]
        arr += tuple(b1)
        if nc>0:
            if nc>1:
                c1 = c[i]
            else:
                c1 = np.array(c,ndmin=1)[0]
            arr += tuple(c1)
        out[i] = sep.join(arr)
    if (n==1) & (t1 is str) & (t2 is str) & (t3 is str): return out[0]  # scalar
    if (t1 is list) | (t2 is list) | (t3 is list): return list(out)
    return out


def strsplit(lst=None,delim=None,asarray=False):
    """ Split a string array."""
    if (lst is None): raise ValueError("lst must be input")
    if size(lst)==1:
        out = lst.split(delim)
    else:
        out = [l.split(delim) for l in lst]
    if asarray is True:
        nlst = np.array(lst).size
        nel = [len(o) for o in out]
        nlen = np.max(strlen(lst))
        outarr = np.zeros((nlst,np.max(nel)),(np.str,nlen))
        for i in range(nlst):
            temp = np.array(out[i])
            ntemp = len(temp)
            outarr[i,0:ntemp] = temp
        return outarr
    else:
        return out

def pathjoin(indir=None,name=None):
    """ Join two or more pathname components, inserting '/' as needed
    Same as os.path.join but also works on arrays/lists."""
    if indir is None: raise ValueError("must input indir")
    if name is None: raise ValueError("must input name")
    nindir = size(indir)
    nname = size(name)
    n = np.max([nindir,nname])
    len1 = strlen(indir)
    len2 = strlen(name)
    nlen = np.max(len1)+np.max(len2)+1
    out = np.zeros(n,(np.str,nlen))
    for i in range(n):
        if nindir>1:
            indir1 = indir[i]
        else:
            indir1 = np.array(indir,ndmin=1)[0]
        if indir1[-1] != '/': indir1+='/'
        if nname>1:
            name1 = nname[i]
        else:
            name1 = np.array(name,ndmin=1)[0]
        out[i] = indir1+name1
    if (n==1) & (type(indir) is str) & (type(name) is str): return out[0]  # scalar
    if (type(indir) is list) | (type(name) is list): return list(out)
    return out

def first_el(lst):
    """ Return the first element"""
    if lst is None: return None
    if size(lst)>1: return lst[0]
    if (size(lst)==1) & (type(lst) is list): return lst.pop() 
    if (size(lst)==1) & (type(lst) is np.ndarray): return lst.item()
    return lst
        

# Standard grep function that works on string list
def grep(lines=None,expr=None,index=False):
    """
    Similar to the standard unit "grep" but run on a list of strings.
    Returns a list of the matching lines unless index=True is set,
    then it returns the indices.
    Parameters
    ----------
    lines : list
          The list of string lines to check.
    expr  : str
          Scalar string expression to search for.
    index : bool, optional
          If this is ``True`` then the indices of matching lines will be
          returned instead of the actual lines.  index is ``False`` by default.

    Returns
    -------
    out : list
        The list of matching lines or indices.

    Examples
    --------
    Search for a string and return the matching lines:
    .. code-block:: python
        mlines = grep(lines,"hello")
    Search for a string and return the indices of the matching lines:
    .. code-block:: python
        index = grep(lines,"hello",index=True)
    """
    if lines is None: raise ValueError("lines must be input")
    if expr is None: raise ValueError("expr must be input")
    out = []
    cnt = 0
    for l in np.array(lines,ndmin=1):
        m = re.search(expr,l)
        if m != None:
            if index is False:
                out.append(l)
            else:
                out.append(cnt)
        cnt = cnt+1
    return out

# Create an empty file
def touch(fname):
    open(fname, 'a').close()


# Read in all lines of files
def readlines(fil=None,raw=False):
    """
    Read in all lines of a file.
    
    Parameters
    ----------
    file : str
         The name of the file to load.
    raw : bool, optional, default is false
         Do not trim \n off the ends of the lines.

    Returns
    -------
    lines : list
          The list of lines from the file

    Examples
    --------
    .. code-block:: python
       lines = readlines("file.txt")
    """
    if fil is None: raise ValueError("File not input")
    f = open(fil,'r')
    lines = f.readlines()
    f.close()
    # Strip newline off
    if raw is False: lines = [l.rstrip('\n') for l in lines]
    return lines


# Write all lines to file
def writelines(filename=None,lines=None,overwrite=True,raw=False):
    """
    Write a list of lines to a file.
    
    Parameters
    ----------
    filename : str
        The filename to write the lines to.
    lines : list
         The list of lines to write to a file.
    overwrite : bool, optional, default is True
        If the output file already exists, then overwrite it.
    raw : bool, optional, default is False
        Do not modify the lines. Write out as is.

    Returns
    -------
    Nothing is returned.  The lines are written to `fil`.

    Examples
    --------
    .. code-block:: python
       writelines("file.txt",lines)
    """
    # Not enough inputs
    if lines is None: raise ValueError("No lines input")
    if filename is None: raise ValueError("No file name input")
    # Check if the file exists already
    if os.path.exists(filename):
        if overwrite is True:
            os.remove(filename)
        else:
            print(filename+" already exists and overwrite=False")
            return
    # Modify the input as needed
    if raw is False:
        # List, make sure it ends with \n
        if type(lines) is list:
            for i,l in enumerate(lines):
                if l.endswith('\n') is False:
                    lines[i] += '\n'
            # Make sure final element does not end in \n
            n = size(lines)
            if n>1:
                if lines[-1].endswith('\n'):
                    lines[-1] = lines[-1][0:-1]
            else:
                if lines.endswith('\n'):
                    lines = lines[0:-1]
    # Convert string to list
    if (type(lines) is str) | (type(lines) is np.str_): lines=list(lines)
    # Convert numpy array and numbers to list of strings
    if type(lines) is not list:
        if hasattr(lines,'__iter__'):
            lines = [str(l)+'\n' for l in lines]
            # Make sure final element does not end in \n        
            if lines[-1].endswith('\n'): lines[-1] = lines[-1][0:-1]        
        else:
            lines = str(lines)
    # Write the file
    f = open(filename,'w')
    f.writelines(lines)
    f.close()


# Remove indices from a list
def remove_indices(lst=None,index=None):
    """
    This will remove elements from a list given their indices.
    Use numpy.delete() for numpy arrays instead.
    Parameters
    ----------
    lst : list
          The list from which to remove elements.
    index : list or array
          The list or array of indices to remove.

    Returns
    -------
    newlst : list
           The new list with indices removed.

    Examples
    --------
    Remove indices 1 and 5 from array `arr`.
    .. code-block:: python
        index = [1,5]
        arr  = range(10)
        arr2 = remove_indices(arr,index)
        print(arr)
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    if lst is None: raise ValueError("list must be input")
    if index is None: raise ValueError("index must be input")
    newlst = []
    for i in range(len(lst)):
       if i not in index: newlst.append(lst[i])
    if type(lst) is np.ndarray: newlst = np.array(newlst)
    return newlst


# Little function used by numlines
def blocks(files, size=65536):
    """
    This is a small utility function used by numlines()
    """
    while True:
        b = files.read(size)
        if not b: break
        yield b


# Read number of lines in a file
def numlines(fil=None):
    """
    This function quickly counts the number of lines in a file.
    Parameters
    ----------
    fil : str
          The filename to check the number of lines.
    Returns
    -------
    nlines : int
           The number of lines in `fil`.

    Examples
    --------

    .. code-block:: python
        n = numlines("file.txt")
    """
    if fil is None: raise ValueError("file must be input")
    with open(fil, "r") as f:
        return (sum(bl.count("\n") for bl in blocks(f)))

    # Could also use this
    #count=0
    #for line in open(fil): count += 1


# Set up basic logging to screen
def basiclogger(name=None):
    """
    This sets up a basic logger that writes just to the screen.
    """
    if name is None: name = "log"
    logger = logging.getLogger(name)
    # Only add a handler if none exists
    #  the logger might already have been created
    if len(logger.handlers)==0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)-2s %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# Delete files
def remove(files=None,allow=True):
    """ Delete a list of files."""
    if files is None: raise ValueError("No files input")
    nfiles = np.array(files).size
    for f in np.array(files,ndmin=1):
        if os.path.exists(f):
            os.remove(f)
        else:
            if allow is False: raise Exception(f+" does not exist")

# Do files exist
def exists(files=None):
    """ Check if a list of files exist."""
    if files is None: raise ValueError("No files input")
    nfiles = np.array(files).size
    out = np.zeros(nfiles,np.bool)+False
    for i,f in enumerate(np.array(files,ndmin=1)):
        if os.path.exists(f): out[i] = True
    return out

def lt(x,limit):
    """Takes the lesser of x or limit"""
    # np.minimum() also does this
    if np.array(x).size>1:
        out = [i if (i<limit) else limit for i in x]
    else:
        out = x if (x<limit) else limit
    if type(x) is np.ndarray: return np.array(out)
    return out
    
def gt(x,limit):
    """Takes the greater of x or limit"""
    # np.maximum() also does this
    if np.array(x).size>1:
        out = [i if (i>limit) else limit for i in x]
    else:
        out = x if (x>limit) else limit
    if type(x) is np.ndarray: return np.array(out)
    return out        

def limit(x,llimit,ulimit):
    """Require x to be within upper and lower limits"""
    return lt(gt(x,llimit),ulimit)

def valrange(array):
    if size(array)==1:
        return 0.0
    else:
        return np.max(array)-np.min(array)

def signs(inp):
    """ Return the sign of input.  Return +1.0 for 0.0"""
    s = np.sign(inp)
    bad,nbad = where(s== 0)
    if nbad>0:
        if size(s)>1:
            s[bad] = 1
        else:
            s = 1.0
    return s

def scale(arr,oldrange,newrange):
    """
    This function maps an array or image onto a new
    scale given two points on the old scale and
    the corresponding points on the new scale.
    The array is converted to double type.
    It's similar to BYTSCL.PRO except that you
    can set the bottom value as well.
    The ranges can be increasing or decreasing.
    INPUTS:
    arr      The array of values to be scaled
    oldrange Two-element array specifiying The original range which
               will be scaled to newrange.
    newrange Two-element array specifiying The new range which
               the oldrange will be scaled to.
    OUTPUTS:
    narr     The new scaled array
    USAGE:
    arr2 = scale(arr,[0,1],[150,2000])
    By D.Nidever   March 2007
    """

    if len(newrange)!=2:
        raise ValueError("newrange must be a 2-element array or list")
    if len(oldrange)!=2:
        raise ValueError("oldrange must be a 2-element array or list")
    
    # Does it flip around
    signchange = 1.0
    if signs(oldrange[1]-oldrange[0]) != signs(newrange[1]-newrange[0]):
        signchange = -1.0 
    # Scale
    narr = valrange(newrange) * signchange*(np.float64(arr)-oldrange[0])/val.range(oldrange) + newrange[0]
    return narr
    
def scale_vector(vector, minrange, maxrange):
    """ Scale a vector to minrange and maxrange. """

    # Make sure we are working with floating point numbers.
    minRange = np.float64( minrange )
    maxRange = np.float64( maxrange )

    # Make sure we have a valid range.
    if (maxRange == minrange):
        raise ValueError("Range max and min are coincidental")
        return vector*0+minrange

    vectormin = np.float64(np.min(vector))
    vectormax = np.float64(np.max(vector))
    
    # Calculate the scaling factors.
    scaleFactor = [((minrange * vectormax)-(maxrange * vectormin)) /
                   (vectormax - vectormin), (maxrange - minrange) / (vectormax - vectormin)]

    # Return the scaled vector.
    return vector * scaleFactor[1] + scaleFactor[0]
    

def quadratic_bisector(x,y):
    """ Calculate the axis of symmetric or bisector of parabola"""
    #https://www.azdhs.gov/documents/preparedness/state-laboratory/lab-licensure-certification/technical-resources/
    #    calibration-training/12-quadratic-least-squares-regression-calib.pdf
    #quadratic regression statistical equation
    n = len(x)
    if n<3:
        return None
    Sxx = np.sum(x**2) - np.sum(x)**2/n
    Sxy = np.sum(x*y) - np.sum(x)*np.sum(y)/n
    Sxx2 = np.sum(x**3) - np.sum(x)*np.sum(x**2)/n
    Sx2y = np.sum(x**2 * y) - np.sum(x**2)*np.sum(y)/n
    Sx2x2 = np.sum(x**4) - np.sum(x**2)**2/n
    #a = ( S(x^2*y)*S(xx)-S(xy)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    #b = ( S(xy)*S(x^2x^2) - S(x^2y)*S(xx^2) ) / ( S(xx)*S(x^2x^2) - S(xx^2)^2 )
    denom = Sxx*Sx2x2 - Sxx2**2
    if denom==0:
        return np.nan
    a = ( Sx2y*Sxx - Sxy*Sxx2 ) / denom
    b = ( Sxy*Sx2x2 - Sx2y*Sxx2 ) / denom
    if a==0:
        return np.nan
    return -b/(2*a)

def wtmean(x,sigma,error=False,reweight=False,magnitude=False):
    """ Calculate weighted mean and error"""
    n = len(x)
    wt = 1/sigma**2
    # Magnitudes
    if magnitude:
        fmn = np.sum( 2.5118864**x * wt) / np.sum(wt)
        xmn = 2.50*np.log10(fmn)
    else:
        xmn = np.sum(wt*x)/np.sum(wt)
    # Reweight the points based on the residuals
    #  using formula similar to the one given by
    #  Stetson (1996) pg.4
    if reweight:
        if magnitude:
            resid = x-xmn
            wt2 = wt/(1+np.abs(resid)**2/np.mean(sigma))            
            fmn2 = np.sum( 2.5118864**x * wt2) / np.sum(wt2)
            xmn = 2.50*np.log10(fmn2)
        else:
            resid = x-xmn
            wt2 = wt/(1+np.abs(resid)**2/np.mean(sigma))
            xmn = np.sum(wt2*x)/np.sum(wt2)
    # Include uncertainty
    if error:
        if magnitude:
            xerr = np.sqrt(1.0/np.sum(wt))
        else:
            xerr = np.sqrt( np.sum( ((x-xmn)**2)*wt)*n / ((n-1)*np.sum(wt))) / np.sqrt(n)
        return xmn,xerr
    else:
        return xmn

def wtslope(x,y,sigma,error=False,reweight=False):
    """ Calculate weighted slope and error"""
    n = len(x)
    wt = 1/sigma**2
    totwt = np.sum(wt)
    mnx = np.sum(wt*x)/totwt
    mny = np.sum(wt*y)/totwt
    wtx =  (np.sum(wt*x*y)/totwt-mnx*mny)/(np.sum(wt*x**2)/totwt-mnx**2)
    # Reweight the points based on the residuals
    #  using formula similar to the one given by
    #  Stetson (1996) pg.4
    if reweight:
        resid = y-wtx*x
        resid -= np.mean(resid)
        wt2 = wt/(1+np.abs(resid)**2/np.mean(sigma))
        totwt2 = np.sum(wt2)
        mnx2 = np.sum(wt2*x)/totwt2
        mny2 = np.sum(wt2*y)/totwt2
        wtx =  (np.sum(wt2*x*y)/totwt2-mnx2*mny2)/(np.sum(wt2*x**2)/totwt2-mnx2**2)
    if error:
        wtxerr = 1.0/np.sqrt( np.sum(wt*x**2)-mnx**2 * np.sum(wt))
        return wtx, wtxerr
    else:
        return wtx

def robust_slope(x,y,sigma,limits=None,npt=15,reweight=False):
    """ Calculate robust weighted slope"""
    # Maybe add sigma outlier rejection in the future
    n = len(x)
    if n==2:
        return wtslope(x,y,sigma,error=True,reweight=reweight)
    # Calculate weighted pmx/pmxerr
    wt_slp,wt_slperr = wtslope(x,y,sigma,error=True,reweight=reweight)
    wt_y, wt_yerr = wtmean(y,sigma,error=True,reweight=reweight)
    # Unweighted slope
    uwt_slp = wtslope(x,y,sigma*0+1,reweight=reweight)
    # Calculate robust loss metric for range of slope values
    #   chisq = Sum( abs(y-(x*slp-mean(x*slp)))/sigma )
    if limits is None:
        limits = np.array([np.min([0.5*wt_slp,0.5*uwt_slp]), np.max([1.5*wt_slp,1.5*uwt_slp])])
    slp_step = (np.max(limits)-np.min(limits))/(npt-1)
    slp_arr = np.arange(npt)*slp_step + np.min(limits)
    # Vectorize it
    resid = np.outer(y,np.ones(npt))-np.outer(x,np.ones(npt))*np.outer(np.ones(n),slp_arr)
    mnresid = np.mean(resid,axis=0)
    resid -= np.outer(np.ones(n),mnresid)    # remove the mean
    chisq = np.sum( np.abs(resid) / np.outer(sigma,np.ones(npt)) ,axis=0)
    bestind = np.argmin(chisq)
    best_slp = slp_arr[bestind]
    # Get parabola bisector
    lo = np.maximum(0,bestind-2)
    hi = np.maximum(bestind+2,n)
    quad_slp = quadratic_bisector(slp_arr[lo:hi],chisq[lo:hi])
    # Problem with parabola bisector, use best point instead                                                                                                                 
    if np.isnan(quad_slp) | (np.abs(quad_slp-best_slp)> slp_step):
        best_slp = best_slp
    else:
        best_slp = quad_slp
    return best_slp, wt_slperr

def wtmedian(val,wt):
    """Weighted median can be computed by sorting the set of numbers and finding the
    smallest numbers which sums to half the weight of total weight."""
    # https://en.wikipedia.org/wiki/Weighted_median
    si = np.argsort(val.flatten())
    totwt = np.cumsum(np.abs(wt).flatten()[si])
    ind = totwt.searchsorted(totwt.max()*0.5)
    return val.flatten()[si[ind-1]]

def gaussian(x, amp, cen, sig, const=0.0, slp=0.0):
    """1-D gaussian: gaussian(x, amp, cen, sig)"""
    #return (amp / (np.sqrt(2*np.pi) * sig)) * np.exp(-(x-cen)**2 / (2*sig**2)) + const
    return amp * np.exp(-(x-cen)**2 / (2*sig**2)) + const + slp*(x-cen)

def gaussbin(x, amp, cen, sig, const=0, slp=0.0, dx=1.0):
    """1-D gaussian with pixel binning
    
    This function returns a binned Gaussian
    par = [height, center, sigma]
    
    Parameters
    ----------
    x : array
       The array of X-values.
    amp : float
       The Gaussian height/amplitude.
    cen : float
       The central position of the Gaussian.
    sig : float
       The Gaussian sigma.
    const : float, optional, default=0.0
       A constant offset.
    slp : float, optional, default=0.0
       A linear slope around cen.
    dx : float, optional, default=1.0
      The width of each "pixel" (scalar).
    
    Returns
    -------
    geval : array
          The binned Gaussian in the pixel
    """

    xcen = np.array(x)-cen             # relative to the center
    x1cen = xcen - 0.5*dx  # left side of bin
    x2cen = xcen + 0.5*dx  # right side of bin

    t1cen = x1cen/(np.sqrt(2.0)*sig)  # scale to a unitless Gaussian
    t2cen = x2cen/(np.sqrt(2.0)*sig)

    # For each value we need to calculate two integrals
    #  one on the left side and one on the right side

    # Evaluate each point
    #   ERF = 2/sqrt(pi) * Integral(t=0-z) exp(-t^2) dt
    #   negative for negative z
    geval_lower = erf(t1cen)
    geval_upper = erf(t2cen)

    geval = amp*np.sqrt(2.0)*sig * np.sqrt(np.pi)/2.0 * ( geval_upper - geval_lower )
    geval += const + slp(x-cen)   # add constant offset and slope

    return geval

def gaussfit(x,y,initpar,sigma=None, bounds=None, binned=False):
    """Fit 1-D Gaussian to X/Y data"""
    #gmodel = Model(gaussian)
    #result = gmodel.fit(y, x=x, amp=initpar[0], cen=initpar[1], sig=initpar[2], const=initpar[3])
    #return result
    func = gaussian
    if binned is True: func=gaussbin
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)


def voigt(x, height, cen, sigma, gamma, const=0.0, slp=0.0):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian sigma.
    """

    maxy = np.real(wofz((1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
    return (height/maxy) * np.real(wofz(((x-cen) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi) + const + slp*(x-cen)

def voigtfit(x,y,initpar=None,sigma=None,bounds=(-np.inf,np.inf)):
    """Fit a Voigt profile to data."""
    if initpar is None:
        initpar = [np.max(y),x[np.argmax(y)],1.0,1.0,np.median(y),0.0]
    func = voigt
    return curve_fit(func, x, y, p0=initpar, sigma=sigma, bounds=bounds)

def voigtarea(pars):
    """ Compute area of Voigt profile"""
    sig = np.maximum(pars[2],pars[3])
    x = np.linspace(-20*sig,20*sig,1000)+pars[1]
    dx = x[1]-x[0]
    v = voigt(x,np.abs(pars[0]),pars[1],pars[2],pars[3])
    varea = np.sum(v*dx)
    return varea

def poly(x,coef,*args):
    """ Evaluate a polynomial function of a variable."""
    # p(x) = p[0] * x**deg + ... + p[deg]
    y = np.array(x).copy()*0.0
    # concatenate coefficients
    if len(args)>0:
        coef = np.hstack((coef,np.array(args)))
    n = len(coef)
    for i in range(n):
        y += coef[i]*x**(n-1-i)
    return y

def poly_resid(coef,x,y,sigma=1.0):
    sig = sigma
    if sigma is None: sig=1.0
    return (poly(x,coef)-y)/sig

def poly_fit(x,y,nord,robust=False,sigma=None,initpar=None,bounds=(-np.inf,np.inf),error=False,max_nfev=None):
    if initpar is None: initpar = np.zeros(nord+1)
    # Normal polynomial fitting
    #if sigma is None: sigma=np.zeros(len(x))+1
    #coef, cov = curve_fit(poly, x, y, p0=initpar, sigma=sigma, bounds=bounds)
    #perr = np.sqrt(np.diag(cov))
    #return coef, perr

    #weights = None
    #if sigma is not None: weights=1/sigma**2
    #if error:
    #    if len(x)>nord+3:
    #        coef, cov = np.polyfit(x,y,nord,w=weights,cov='unscaled')
    #        perr = np.sqrt(np.diag(cov))
    #    else:
    #        coef = np.polyfit(x,y,nord,w=weights)
    #        perr = coef.copy()*0.0
    #
    #    return coef, perr
    #else:
    #    coef = np.polyfit(x,y,nord,w=weights)
    #    return coef

    loss = 'linear'
    if robust: loss='soft_l1'
    if sigma is None: sigma=np.zeros(len(x))+1
    res = least_squares(poly_resid, initpar, loss=loss, f_scale=0.1, args=(x,y,sigma), max_nfev=max_nfev)
    if res.success is False:
        import pdb; pdb.set_trace()
        raise Exception("Problem with least squares polynomial fitting. Status="+str(res.status))
        return initpar+np.nan
    coef = res.x
    # Calculate the covariance matrix
    #  this is how scipy.optimize.curve_fit computes the covariance matrix
    #  https://github.com/scipy/scipy/blob/2526df72e5d4ca8bad6e2f4b3cbdfbc33e805865/scipy/optimize/minpack.py#L739
    if error:
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        # Compute errors on the parameters
        perr = np.sqrt(np.diag(pcov))
        return coef, perr
    else:
        return coef

# Derivative or slope of an array
def slope(array):
    """Derivative or slope of an array: slp = slope(array)"""
    n = len(array)
    return array[1:n]-array[0:n-1]

# Gaussian filter
def gsmooth(data,fwhm,mask=None,boundary='extend',fill=0.0,truncate=4.0,squared=False):
    # astropy.convolve automatically ignores NaNs
    # Create kernel
    xsize = np.ceil(fwhm/2.35*truncate*2)
    if xsize % 2 == 0: xsize+=1   # must be odd
    g = Gaussian1DKernel(stddev=fwhm/2.35,x_size=xsize)
    if squared is False:
        return convolve(data, g.array, mask=mask, boundary=boundary, fill_value=fill)
        #return gaussian_filter1d(data,fwhm/2.35,axis=axis,mode=mode,cval=cval,truncate=truncate)
    else:
        return convolve(data, g.array**2, mask=mask, boundary=boundary, fill_value=fill, normalize_kernel=False)
        #return gaussian_filter1d(data,fwhm/2.35,axis=axis,mode=mode,cval=cval,truncate=truncate)**2
    
# Rebin data
def rebin(arr, new_shape):
    if arr.ndim>2:
        raise Exception("Maximum 2D arrays")
    if arr.ndim==0:
        raise Exception("Must be an array")
    if arr.ndim==2:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        return arr.reshape(shape).mean(-1).mean(1)
    if arr.ndim==1:
        shape = (np.array(new_shape,ndmin=1)[0], arr.shape[0] // np.array(new_shape,ndmin=1)[0])
        return arr.reshape(shape).mean(-1)

def roi_cut(xcut,ycut,x,y):
    """
    Use cuts in a 2D plane to select points from arrays.
    Parameters
    ----------
    xcut : numpy array
         Array of x-values for the cut
    ycut : numpy array
         Array of y-values for the cut
    x : numpy array or list
         Array of x-values that should be cut
    y : numpy array or list
         Array of y-values that should be cut
    Returns
    -------
    ind : numpy array
       The indices of values OUTSIDE the cut
    cutind : 
       The indices of values INSIDE the cut
    Example
    -------
    .. code-block:: python
        ind, cutind = roi_cut(xcut,ycut,x,y)
    """

    from matplotlib.path import Path

    tupVerts = list(zip(xcut,ycut))

    points = np.vstack((x,y)).T
    
    p = Path(tupVerts) # make a polygon
    inside = p.contains_points(points)

    ind, = np.where(~inside)
    cutind, = np.where(inside)

    return ind, cutind


def create_index(arr):
    """
    Create an index of array values like reverse indices.
    arr[index['index'][index['lo'][2]:index['hi'][2]+1]]
    """
    
    narr = size(arr)
    if narr==0:
        raise ValueError('arr has no elements')
    si = np.argsort(arr)
    sarr = np.array(arr)[si]
    brklo, = np.where(sarr != np.roll(sarr,1))
    nbrk = len(brklo)
    if nbrk>0:
        brkhi = np.hstack((brklo[1:nbrk]-1,narr-1))
        num = brkhi-brklo+1
        index = {'index':np.atleast_1d(si),'value':np.atleast_1d(sarr[brklo]),
                 'num':np.atleast_1d(num),'lo':np.atleast_1d(brklo),'hi':np.atleast_1d(brkhi)}
    else:
        index = {'index':np.atleast_1d(si),'value':np.atleast_1d(arr[0]),
                 'num':np.atleast_1d(narr),'lo':np.atleast_1d(0),'hi':np.atleast_1d(narr-1)}

    return index


def match(a,b,epsilon=0):
    """
    Routine to match values in two vectors.
    
    CALLING SEQUENCE:
        match, a, b, suba, subb, [ COUNT =, /SORT, EPSILON =  ]
  
    INPUTS:
        a,b - two vectors to match elements, numeric or string data types
    
    OUTPUTS:
      suba - subscripts of elements in vector a with a match
                  in vector b
      subb - subscripts of the positions of the elements in
                  vector b with matchs in vector a.
  
          suba and subb are ordered such that a[suba] equals b[subb]
          suba and subb are set to !NULL if there are no matches (or set to -1
                if prior to IDL Version 8.0)
  
    OPTIONAL INPUT KEYWORD:
          /SORT - By default, MATCH uses two different algorithm: (1) the
                  /REVERSE_INDICES keyword to HISTOGRAM is used for integer data,
                  while (2) a sorting algorithm is used for non-integer data.  The
                  histogram algorithm is usually faster, except when the input
                  vectors are sparse and contain very large numbers, possibly
                  causing memory problems.   Use the /SORT keyword to always use
                  the sort algorithm.
          epsilon - if values are within epsilon, they are considered equal. Used only
                  only for non-integer matching.  Note that input vectors should
                  be unique to within epsilon to provide one-to-one mapping.
                  Default=0.
   
    OPTIONAL KEYWORD OUTPUT:
          COUNT - set to the number of matches, integer scalar
   
    SIDE EFFECTS:
          The obsolete system variable !ERR is set to the number of matches;
          however, the use !ERR is deprecated in favor of the COUNT keyword
   
    RESTRICTIONS:
          The vectors a and b should not have duplicate values within them.
          You can use rem_dup function to remove duplicate values
          in a vector
   
    EXAMPLE:
          If a = [3,5,7,9,11]   & b = [5,6,7,8,9,10]
          then
                  IDL> match, a, b, suba, subb, COUNT = count
   
          will give suba = [1,2,3], subb = [0,2,4],  COUNT = 3
          and       a[suba] = b[subb] = [5,7,9]
   
   
    METHOD:
          For non-integer data types, the two input vectors are combined and
          sorted and the consecutive equal elements are identified.   For integer
          data types, the /REVERSE_INDICES keyword to HISTOGRAM of each array
          is used to identify where the two arrays have elements in common.
    HISTORY:
         D. Lindler  Mar. 1986.
         Fixed "indgen" call for very large arrays   W. Landsman  Sep 1991
         Added COUNT keyword    W. Landsman   Sep. 1992
         Fixed case where single element array supplied   W. Landsman Aug 95
         Use a HISTOGRAM algorithm for integer vector inputs for improved
               performance                W. Landsman         March 2000
         Work again for strings           W. Landsman         April 2000
         Use size(/type)                  W. Landsman         December 2002
         Work for scalar integer input    W. Landsman         June 2003
         Assume since V5.4, use COMPLEMENT to WHERE() W. Landsman Apr 2006
         Added epsilon keyword            Kim Tolbert         March 14, 2008
         Fix bug with Histogram method with all negative values W. Landsman/
         R. Gutermuth, return !NULL for no matches  November 2017
         Added epsilon test in na=1||nb=1 section (missed that when added
               epsilon in 2008)           Kim Tolbert         July 10, 2018
  
    """

    #da = size(a,/type) & db =size(b,/type)
    #if keyword_set(sort) then hist = 0b else $
    #  hist = (( da LE 3 ) || (da GE 12)) &&  ((db LE 3) || (db GE 12 ))

    na = size(a)             # number of elements in a
    nb = size(b)             # number of elements in b
    
    # Check for a single element array
    if (na==1) | (nb==1):
        if (nb>1):
            if epsilon==0.0:
                subb, = np.where(b==a)
                nw = len(subb)
            else:
                subb, = np.where(np.abs(b-a) < epsilon)
                nw = len(subb)
            if (nw>0):
                suba = np.zeros(nw,int)
            else:
                suba = np.array([])
        else:
            if epsilon==0.0:
                suba, = np.where(a==b)
                nw = len(suba)
            else:
                suba, = np.where(np.abs(a-b) < epsilon)
                nw = len(suba)
            if (nw>0):
                subb, = np.zeros(nw,int)
            else:
                subb = np.array([])
        count = nw
        return suba,subb

    # Conver to numpy.chararray if either of them are strings
    a1 = first_el(a)
    b1 = first_el(b)
    if isinstance(a,np.chararray) | isinstance(a1,str) | isinstance(a1,np.string_) | \
       isinstance(b,np.chararray) | isinstance(b1,str) | isinstance(b1,np.string_):
        atemp = np.char.array(a)
        btemp = np.char.array(b)
        # Use the dtype with the largest number of characters
        if atemp.dtype > btemp.dtype:
            dtype = atemp.dtype
        else:
            dtype = btemp.dtype
        c = np.zeros(na+nb,dtype=dtype)
        c[0:na] = atemp
        c[na:] = btemp
        c = np.char.array(c)  # convert to np.chararray, removes trailing strings
        del atemp, btemp
        #c = np.hstack((np.char.array(a),np.char.array(b)))           # combined list of a and b
    else:
        c = np.hstack((np.array(a),np.array(b)))                     # combined list of a and b
    ind = np.hstack((np.arange(na),np.arange(nb)))               # combined list of indices
    vec = np.hstack((np.zeros(na,bool),np.zeros(nb,bool)+True))  # flag of which vector in  combined
    #list   False - a   True - b

    # sort combined list
    sub = np.argsort(c)
    c = c[sub]
    ind = ind[sub]
    vec = vec[sub]

    # find duplicates in sorted combined list
    n = na + nb                            #t otal elements in c
    if epsilon == 0.0:
      firstdup, = np.where( (c == np.roll(c,-1)) & (vec != np.roll(vec,-1)) )
      count = len(firstdup)
    else:
      firstdup, = np.where( (np.abs(c - np.roll(c,-1)) < epsilon) & (vec != np.roll(vec,-1)) )
      count = len(firstdup)

    if count==0:               # any found?
      suba = np.array([])
      subb = np.array([])
      return suba,subb

    dup = np.zeros( count*2, int )        # both duplicate values
    even = np.arange( len(firstdup))*2     # Changed to LINDGEN 6-Sep-1991
    dup[even] = firstdup
    dup[even+1] = firstdup+1
    ind = ind[dup]                         # indices of duplicates
    vec = vec[dup]                         # vector id of duplicates
    vone, = np.where(vec)
    vzero, = np.where(~vec)
    subb = ind[vone]                       # b subscripts
    suba = ind[vzero]


    # # Integer calculation using histogram.
    # else:
    #
    #     minab = min(a, MAX=maxa) > min(b, MAX=maxb) #Only need intersection of ranges
    #     maxab = maxa < maxb
    #
    #     #If either set is empty, or their ranges don't intersect:
    #     #  result = NULL (which is denoted by integer = -1)
    #     !ERR = -1
    #     if !VERSION.RELEASE GE '8.0' then begin
    #        suba = !NULL
    #        subb = !NULL
    #     endif else begin
    #        suba = -1
    #        subb = -1
    #     endelse
    #     COUNT = 0L
    #     if maxab lt minab then return       #No overlap
    #
    #     ha = histogram([a], MIN=minab, MAX=maxab, reverse_indices=reva)
    #     hb = histogram([b], MIN=minab, MAX=maxab, reverse_indices=revb)
    #
    #     r = where((ha ne 0) and (hb ne 0), count)
    #
    #     if count gt 0 then begin
    #        suba = reva[reva[r]]
    #        subb = revb[revb[r]]
    
    return suba, subb

# Interpolation with extrapolation
def interp(x,y,xout,kind='cubic',bounds_error=False,assume_sorted=True,extrapolate=True,exporder=2,fill_value=np.nan):
    yout = interp1d(x,y,kind=kind,bounds_error=bounds_error,fill_value=(fill_value,fill_value),assume_sorted=assume_sorted)(xout)
    # Need to extrapolate
    if ((np.min(xout)<np.min(x)) | (np.max(xout)>np.max(x))) & (extrapolate is True):
        si = np.argsort(x)
        npix = len(x)
        nfit = np.min([10,npix])
        # At the beginning
        if (np.min(xout)<np.min(x)):
            coef1 = poly_fit(x[0:nfit], y[0:nfit], exporder)
            bd1, nbd1 = where(xout < np.min(x))
            yout[bd1] = poly(xout[bd1],coef1)
        # At the end
        if (np.max(xout)>np.max(x)):
            coef2 = poly_fit(x[npix-nfit:], y[npix-nfit:], exporder)
            bd2, nbd2 = where(xout > np.max(x))
            yout[bd2] = poly(xout[bd2],coef2)     
    return yout

def concatenate(a,b=None):
    # Concatenate two or more numpy structured arrays
    # Can input two numpy structured arrays or a list of them
    if (b is None and type(a) is not list) | (b is not None and type(a) is list):
        raise Exception('Must input two numpy structured arrays or a list of them')
        return
    if type(a) is not list: a=list(a)
    if b is not None: a.append(b)

    # Get dtypes for all of the numpy structured arrays
    ncat = size(a)
    dtypearr = []
    ncols = []
    nrows = []
    for a1 in a:
        dtype1 = a1.dtype
        dtypearr.append(dtype1)
        ncols.append(len(dtype1.names))
        nrows.append(len(a1))
    ncols = np.array(ncols)
    nrows = np.array(nrows)
    # Ncols not the same
    if np.min(ncols) != np.max(ncols):
        raise Exception('Number of columns are not the same: min='+str(np.min(ncols))+' max='+str(np.max(ncols)))
    # Checking column names
    colnames = np.zeros((ncat,ncols[0]),dtype=(np.str,100))
    for i in range(ncat):
        colnames[i,:] = a[i].dtype.names
        if i>0:
            if list(colnames[0,:]) != list(colnames[i,:]):
                raise Exception('Column names are not the same')

    # Make the final dtype
    #   make sure string columns are same length
    dtype_list = []
    for f in a[0].dtype.names:
        if a[0].dtype[f].char == 'S':
            isize = []
            for d1 in dtypearr:
                isize.append(d1[f].itemsize)
            maxsize = np.max(isize)
            dtype_list.append((f,'S'+str(maxsize)))
        else:
            dtype_list.append((f,a[0].dtype[f].str))
    dtype = np.dtype(dtype_list)

    # Create the final structure and load the data
    nlstr = np.sum(nrows)
    lstr = np.zeros(nlstr,dtype=dtype)
    count = 0
    for i in range(ncat):
        a1 = a[i]
        n = len(a1)
        lstr[count:count+n] = a[i]
        count += n
    return lstr

def addcatcols(cat,dt):
    """ Add new columns to an existing numpty structured array catalog."""
    ncat = len(cat)
    odt = cat.dtype

    # Concatenate the dtypes
    dtype_list = []
    for f in cat.dtype.names:
        dtype_list.append((f,cat.dtype[f]))
    #    dtype_list.append((f,cat.dtype[f].str))
    for f in dt.names:
        dtype_list.append((f,dt[f]))
    #    dtype_list.append((f,dt[f].str))
    newdtype = np.dtype(dtype_list)    
    
    # Create the final structure and load the data
    new = np.zeros(ncat,dtype=newdtype)
    for n in cat.dtype.names: new[n] = cat[n]

    return new    
    

def onclick(event):
    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #      ('double' if event.dblclick else 'single', event.button,
    #       event.x, event.y, event.xdata, event.ydata))

    #global ix, iy
    if event.xdata is None:
        global cid
        fig.canvas.mpl_disconnect(cid)
        print('Done.  Coordinates are in global "coords" list')
        return
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(ix, iy))

    global coords
    try:
        dum = len(coords)
    except:
        coords = []
    coords.append((ix, iy))

    return

def clicker():
    # im=fits.getdata('F1-00507803_23.fits.fz') 
    # fig = plt.figure()
    # plt.imshow(im)
    # clicker()
    # then the coordinates will be the global "coords" list
    print('Click outside the plot to end')
    global cid
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    coords = []


def add_elements(cat,num=10000):
    """ Add more elements to a catalog"""
    ncat = len(cat)
    old = cat.copy()
    num = gt(num,ncat)
    cat = np.zeros(ncat+num,dtype=old.dtype)
    cat[0:ncat] = old
    del old
    return cat 


def ellipsecoords(pars,npoints=100):
    """ Create coordinates of an ellipse."""
    # [x,y,asemi,bsemi,theta]
    # copied from ellipsecoords.pro
    xc = pars[0]
    yc = pars[1]
    asemi = pars[2]
    bsemi = pars[3]
    pos_ang = pars[4]
    phi = 2*np.pi*(np.arange(npoints)/(npoints-1))   # Divide circle into Npoints
    ang = np.deg2rad(pos_ang)                             # Position angle in radians
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    x =  asemi*np.cos(phi)                              # Parameterized equation of ellipse
    y =  bsemi*np.sin(phi)
    xprime = xc + x*cosang - y*sinang               # Rotate to desired position angle
    yprime = yc + x*sinang + y*cosang
    return xprime, yprime

def closest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)

def phase_fold(mjd, period, mjd0=None, centeredzero=False):
    '''
    Given a series of dates and a period, it phase folds them into 
    a number of cycles.
    
    Parameters
    ----------
    mjd: array-like
        Array of dates to phase fold in Mean Julian Date (MJD).
    period: Quantity 
        Period to use for the phase fold, in days.
    mjd0: float, optional
        Initial day to use for the phase fold. Default is the earliest 
        date in mjd.
    centeredzero: boolean, optional
        Whether the phase goes from 0 to 1 or -0.5 to 0.5. Default 
        is False (0 to 1).
    
    Returns
    -------
    phase: array-like
        Converted mjd to phase values.
    '''
    
    if mjd0==None:
        mjd0 = min(mjd)
    
    temp = mjd - mjd0
    nonfold_phase = temp/period
    cycle = np.array(nonfold_phase, dtype=int)
    phase = nonfold_phase - cycle 
    
    if centeredzero:
        phase -= 0.5
    
    return phase

def plot_periodogram(frequency, power, filename='periodogram.png', units='days'):
    '''
    Plots a periodogram based on the power and frequency given.
    
    Parameters
    ----------
    frequency: array-like
        Frequency as 1/period, where the period is measured in days. 
    power: array-like
        Corresponding power for each frequency. Must be the same shape as frequency.
    filename: str, optional
        Name of the file where to store the plot.
    units: str, optional
        Units used for the periodogram plot. Default is days.
        
    Returns
    -------
    Saves a file to disk.
    '''
    
    periods = 1/frequency
    
    fig = plt.figure()
    ax = fig.add_subplot()
        
    ax.plot(periods, power, lw=1, color='k')
    ax.set_ylabel('Power')
    ax.set_xlabel(f'Period ({units})')
    ax.set_xscale('log')
    
    fig.savefig(filename,bbox_inches = 'tight')
    
    
def plot_phased_lightcurve(phase, mags, mags_errs=None, filters=None, filename='timecurve.png'):
    '''
    Plot a phase folded light curve. If multiple filters are present, plots them with different colors.
    
    Parameters
    ----------
    phase: array-like
        Phases for each exposures.
    mags: array-like
        Magnitude of object in each exposure. Must have same shape as phase.
    mags_errs: array-like
        Associated errors for magnitudes. Must have the same shape as phase and mags.
    filters: string, array-like, optional
        Unique filters present in the data. If not given, all will be assumed to be the same.
    filename: string
        Path to save the plot.
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    
    if filters is not None:
        for fltr in np.unique(filters):
            sel = filters==fltr
            ax.scatter(phase[sel], mags[sel], label=fltr, s=3)
            if mags_errs is not None:
                ax.errorbar(phase[sel], mags[sel], yerr=mags_errs[sel], 
                            fmt='none', capsize=0, 
                            elinewidth=1.5, 
                            ecolor='gray', 
                            alpha=0.7)
        ax.legend(markerscale=2)
    else:
        ax.scatter(phase, mags, label=fltr, s=3)
        if mags_errs is not None:
            ax.errorbar(phase, mags, yerr=mags_errs, 
                        fmt='none', capsize=0, 
                        elinewidth=1.5, 
                        ecolor='gray', 
                        alpha=0.7)

    
    ax.set_xlabel('Phase')
    ax.set_ylabel('Magnitude')
    fig.savefig(filename,bbox_inches = 'tight')
    
def most_frequent(array, return_counts = False):
    '''
    Returns the most frequent element in an array.
    
    Parameters
    ----------
    array: array-like
        Input array.
    return_counts: bool, optional
        Whether to return the number of counts of the most frequent element. Default is false.
        
    Returns
    -------
    most_frequent
        Most frequent element in array.
    counts: float
        
    '''
    
    unique, counts = np.unique(array, return_counts=True)
    index = np.argmax(counts)
    if return_counts:
        return unique[index], counts[index]
    else:
        return unique[index]
