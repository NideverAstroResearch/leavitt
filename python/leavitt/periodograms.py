import numpy as np
import numba
from astropy.timeseries import LombScargle, LombScargleMultiband

# Use the compiled Cython/C extension when available (built at the repo root).
# Falls back to the Numba JIT version transparently.
_cythonc = True
try:
    import psearch_pyc
except ImportError:
    _cythonc = False


def _ctheta_slave(parray, mag, tobs):
    """
    Compute the Lafler-Kinman string-length statistic over an array of trial periods.

    For each trial period the magnitudes are sorted by phase and the sum of
    squared differences between consecutive phase-sorted values is computed,
    normalised by the total variance.  Lower theta indicates a better period
    (smoother phase-folded light curve).

    This function is the pure-Python/Numba-compatible implementation; use
    ``lk_periodogram`` rather than calling it directly.

    Parameters
    ----------
    parray : ndarray of float64
        Trial periods in days.
    mag : ndarray of float64
        Magnitudes, co-aligned with ``tobs``.
    tobs : ndarray of float64
        Observation times in days, co-aligned with ``mag``.

    Returns
    -------
    theta : ndarray of float64
        String-length statistic at each trial period (same length as ``parray``).

    References
    ----------
    Lafler, J., & Kinman, T. D. 1965, ApJS, 11, 216
    Saha, A., & Vivas, A. K. 2017, AJ, 154, 231
    """
    t0 = np.min(tobs)
    tt = tobs - t0
    theta = np.zeros_like(parray)
    mmplus = np.zeros_like(mag)
    avm = np.sum(mag) / len(mag)
    denom = np.sum((mag - avm) ** 2)
    for k in range(len(parray)):
        phi = tt / parray[k]
        phi -= phi.astype(np.int64)
        ss = np.argsort(phi)
        mm = mag[ss]
        mmplus[:-1] = mm[1:]
        mmplus[-1] = mm[0]
        theta[k] = np.sum((mmplus - mm) ** 2) / denom
    return theta


# Numba-compiled version; compiled on first call.
_ctheta_slave_jit = numba.jit(nopython=True)(_ctheta_slave)


def frequency_array(minimum_frequency, maximum_frequency, nbins=100):
    """
    Build a linearly spaced frequency grid.

    Parameters
    ----------
    minimum_frequency : float
        Lower bound of the frequency range (1/day).
    maximum_frequency : float
        Upper bound of the frequency range (1/day).
    nbins : int, optional
        Number of frequency bins.  Default is 100.

    Returns
    -------
    farray : ndarray
        Array of ``nbins`` evenly spaced frequencies from ``minimum_frequency``
        to ``maximum_frequency`` inclusive.
    """
    return np.linspace(minimum_frequency, maximum_frequency, nbins)


def ls_periodogram(time, mag, mag_err, minimum_frequency, maximum_frequency,
                   method='flexible', normalization='standard'):
    """
    Compute a single-band Lomb-Scargle periodogram.

    Wraps ``astropy.timeseries.LombScargle.autopower``.

    Parameters
    ----------
    time : array-like
        Observation times (days).
    mag : array-like
        Magnitudes, co-aligned with ``time``.
    mag_err : array-like
        Magnitude uncertainties, co-aligned with ``time``.
    minimum_frequency : float
        Lower bound of the frequency search range (1/day).
    maximum_frequency : float
        Upper bound of the frequency search range (1/day).
    method : str, optional
        Lomb-Scargle algorithm passed to ``autopower``.  Default ``'flexible'``.
    normalization : str, optional
        Periodogram normalisation passed to ``autopower``.  Default
        ``'standard'``.

    Returns
    -------
    frequency : ndarray
        Frequencies at which the periodogram was evaluated (1/day).
    power : ndarray
        Lomb-Scargle power at each frequency.
    """
    frequency, power = LombScargle(time, mag, dy=mag_err).autopower(
        method=method,
        normalization=normalization,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return frequency, power


def ls_mb_periodogram(time, mag, band, mag_err, minimum_frequency, maximum_frequency,
                      method='flexible', normalization='standard'):
    """
    Compute a multi-band Lomb-Scargle periodogram.

    Fits a shared period across all photometric bands simultaneously, using
    ``astropy.timeseries.LombScargleMultiband.autopower``.

    Parameters
    ----------
    time : array-like
        Observation times (days).
    mag : array-like
        Magnitudes, co-aligned with ``time``.
    band : array-like
        Filter/band label for each observation, co-aligned with ``time``.
    mag_err : array-like
        Magnitude uncertainties, co-aligned with ``time``.
    minimum_frequency : float
        Lower bound of the frequency search range (1/day).
    maximum_frequency : float
        Upper bound of the frequency search range (1/day).
    method : str, optional
        Algorithm passed to ``autopower``.  Default ``'flexible'``.
    normalization : str, optional
        Periodogram normalisation passed to ``autopower``.  Default
        ``'standard'``.

    Returns
    -------
    frequency : ndarray
        Frequencies at which the periodogram was evaluated (1/day).
    power : ndarray
        Multi-band Lomb-Scargle power at each frequency.
    """
    frequency, power = LombScargleMultiband(time, mag, band, dy=mag_err).autopower(
        method=method,
        normalization=normalization,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    return frequency, power


def lk_periodogram(time, mag, minimum_frequency, maximum_frequency, nbins=100):
    """
    Compute a Lafler-Kinman string-length periodogram.

    For each trial period the data are phase-folded and the sum of squared
    differences between consecutive phase-sorted magnitudes is computed.  The
    result (theta) is normalised by the total magnitude variance, so theta ~ 1
    for a non-periodic source and theta < 1 at a true period.

    Uses the Cython/C extension (``psearch_pyc``) when available; falls back
    to a Numba JIT-compiled Python implementation.

    Parameters
    ----------
    time : array-like
        Observation times (days).
    mag : array-like
        Magnitudes, co-aligned with ``time``.
    minimum_frequency : float
        Lower bound of the frequency search range (1/day).
    maximum_frequency : float
        Upper bound of the frequency search range (1/day).
    nbins : int, optional
        Number of trial periods/frequencies to evaluate.  Default is 100.

    Returns
    -------
    frequency : ndarray
        Frequencies evaluated (1/day), linearly spaced between
        ``minimum_frequency`` and ``maximum_frequency``.
    theta : ndarray
        Lafler-Kinman statistic at each frequency.  Lower values indicate
        a stronger period detection.

    References
    ----------
    Lafler, J., & Kinman, T. D. 1965, ApJS, 11, 216
    Saha, A., & Vivas, A. K. 2017, AJ, 154, 231
    """
    farray = frequency_array(minimum_frequency, maximum_frequency, nbins)
    parray = (1.0 / farray).astype(np.float64)
    tobs = np.asarray(time, dtype=np.float64)
    mag  = np.asarray(mag,  dtype=np.float64)

    if _cythonc:
        theta = psearch_pyc.ctheta_slave(parray, mag, tobs)
    else:
        theta = _ctheta_slave_jit(parray, mag, tobs)

    return farray, theta
