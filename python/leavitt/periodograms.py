import warnings
import numpy as np
import numba
import astropy.units as u
from astropy.timeseries import LombScargle, LombScargleMultiband

# Use the compiled Cython/C extension when available (built at the repo root).
# Falls back to the Numba JIT version transparently.
_cythonc = True
try:
    import psearch_pyc
except ImportError:
    _cythonc = False
    warnings.warn(
        "psearch_pyc (Cython/C extension) not found or failed to import — "
        "lk_periodogram will use the Numba JIT fallback, which is slower.",
        ImportWarning,
        stacklevel=2,
    )


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
    Build a linearly spaced frequency grid with a fixed number of bins.

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


def adaptive_frequency_array(time, minimum_frequency, maximum_frequency, dphi=0.02):
    """
    Build a frequency grid with fixed phase resolution (Saha & Vivas 2017).

    The step size is ``dphi / tspan``, so consecutive trial periods never
    alias by more than ``dphi`` in phase, regardless of dataset length.
    The number of bins is derived from the data time baseline rather than
    fixed in advance.

    Parameters
    ----------
    time : array-like
        Observation times (days).
    minimum_frequency : float
        Lower bound of the frequency range (1/day).
    maximum_frequency : float
        Upper bound of the frequency range (1/day).
    dphi : float, optional
        Maximum phase change between adjacent trial periods.  Default is 0.02.

    Returns
    -------
    farray : ndarray
        Linearly spaced frequencies with step ``dphi / tspan``.
    """
    tspan = np.max(time) - np.min(time)
    min_f = minimum_frequency.to(u.day**-1).value if hasattr(minimum_frequency, 'unit') else float(minimum_frequency)
    max_f = maximum_frequency.to(u.day**-1).value if hasattr(maximum_frequency, 'unit') else float(maximum_frequency)
    deltafreq = dphi / tspan
    nfreq = int((max_f - min_f) / deltafreq)
    return min_f + np.arange(nfreq) * deltafreq


def ls_periodogram(time, mag, mag_err, minimum_frequency, maximum_frequency,
                   frequency=None, method='auto', normalization='standard'):
    """
    Compute a single-band Lomb-Scargle periodogram.

    Wraps ``astropy.timeseries.LombScargle``.  If ``frequency`` is provided
    the periodogram is evaluated at those exact frequencies via ``.power()``;
    otherwise ``.autopower()`` generates the frequency grid from
    ``minimum_frequency`` and ``maximum_frequency``.

    Parameters
    ----------
    time : array-like
        Observation times (days).
    mag : array-like
        Magnitudes, co-aligned with ``time``.
    mag_err : array-like
        Magnitude uncertainties, co-aligned with ``time``.
    minimum_frequency : float
        Lower bound of the frequency search range (1/day).  Used only when
        ``frequency`` is None.
    maximum_frequency : float
        Upper bound of the frequency search range (1/day).  Used only when
        ``frequency`` is None.
    frequency : array-like, optional
        Explicit frequency grid (1/day).  If given, ``minimum_frequency``
        and ``maximum_frequency`` are ignored.
    method : str, optional
        Lomb-Scargle algorithm.  Default ``'auto'``.
    normalization : str, optional
        Periodogram normalisation.  Default ``'standard'``.

    Returns
    -------
    frequency : ndarray
        Frequencies at which the periodogram was evaluated (1/day).
    power : ndarray
        Lomb-Scargle power at each frequency.
    """
    ls = LombScargle(time, mag, dy=mag_err)
    if frequency is None:
        frequency, power = ls.autopower(
            method=method,
            normalization=normalization,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
    else:
        power = ls.power(frequency, method=method, normalization=normalization)
    return frequency, power


def ls_mb_periodogram(time, mag, band, mag_err, minimum_frequency, maximum_frequency,
                      method='fast', normalization='standard'):
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
    frequency, power = LombScargleMultiband(
        np.asarray(time, dtype=np.float64),
        np.asarray(mag, dtype=np.float64),
        np.asarray(band),
        dy=np.asarray(mag_err, dtype=np.float64),
    ).autopower(
        method=method,
        normalization=normalization,
        minimum_frequency=float(minimum_frequency),
        maximum_frequency=float(maximum_frequency),
    )
    return frequency, power


def lk_periodogram(time, mag, minimum_frequency, maximum_frequency, dphi=0.02):
    """
    Compute a Lafler-Kinman string-length periodogram.

    For each trial period the data are phase-folded and the sum of squared
    differences between consecutive phase-sorted magnitudes is computed.  The
    result (theta) is normalised by the total magnitude variance, so theta ~ 1
    for a non-periodic source and theta < 1 at a true period.

    Uses the Cython/C extension (``psearch_pyc``) when available; falls back
    to a Numba JIT-compiled Python implementation.

    The frequency grid is built with ``adaptive_frequency_array`` so that its
    resolution scales with the data time baseline (Saha & Vivas 2017).

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
    dphi : float, optional
        Phase resolution parameter passed to ``adaptive_frequency_array``.
        Default is 0.02.

    Returns
    -------
    frequency : ndarray
        Frequencies evaluated (1/day).
    theta : ndarray
        Lafler-Kinman statistic at each frequency.  Lower values indicate
        a stronger period detection.

    References
    ----------
    Lafler, J., & Kinman, T. D. 1965, ApJS, 11, 216
    Saha, A., & Vivas, A. K. 2017, AJ, 154, 231
    """
    tobs = np.asarray(time, dtype=np.float64)
    mag  = np.asarray(mag,  dtype=np.float64)
    farray = adaptive_frequency_array(tobs, minimum_frequency, maximum_frequency, dphi)
    parray = (1.0 / farray).astype(np.float64)

    if _cythonc:
        theta = psearch_pyc.ctheta_slave(parray, mag, tobs)
    else:
        theta = _ctheta_slave_jit(parray, mag, tobs)

    return farray, theta


def psi_periodogram(time, mag, band, mag_err, minimum_frequency, maximum_frequency,
                    dphi=0.02, ls_method='auto', n_thresh=1, random_state=None):
    """
    Compute the Psi (Ψ) hybrid periodogram (Saha & Vivas 2017).

    For each photometric band, computes a single-band Lomb-Scargle periodogram
    and the Lafler-Kinman string-length statistic on the same adaptive frequency
    grid, then sums the per-band Psi values:

        Ψ_band = 2 · LS_band / θ_LK_band
        Ψ = Σ Ψ_band

    Processing each band independently prevents inter-band magnitude offsets
    from contaminating the LK phase-coherence test.

    A noise threshold is estimated via Monte Carlo: for each band and each of
    ``n_thresh`` iterations, Psi is computed on (1) pure Gaussian noise scaled
    by the measurement errors and (2) randomly scrambled magnitudes.  Each
    noise Psi is normalised to the same total power as the real per-band Psi
    and accumulated with a running maximum, mirroring the approach of
    Saha & Vivas (2017).  The per-band thresholds are summed across bands.

    Parameters
    ----------
    time : array-like
        Observation times in days (plain floats, e.g. MJD).
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
    dphi : float, optional
        Phase resolution parameter passed to ``adaptive_frequency_array``.
        Default is 0.02.
    ls_method : str, optional
        Algorithm passed to ``ls_periodogram`` for each band.  Default
        is ``'auto'``.
    n_thresh : int, optional
        Number of Monte Carlo iterations used to estimate the noise threshold.
        Set to 0 to skip threshold computation (``thresh`` will be all zeros).
        Default is 1.
    random_state : int or None, optional
        Seed for the random number generator used in threshold estimation.
        Default is None (non-reproducible).

    Returns
    -------
    frequency : ndarray
        Frequencies evaluated (1/day).
    psi : ndarray
        Psi statistic at each frequency.  Higher values indicate a
        stronger period detection.
    thresh : ndarray
        Noise threshold at each frequency.  Peaks where ``psi > thresh``
        are statistically significant.  All zeros when ``n_thresh=0``.

    References
    ----------
    Saha, A., & Vivas, A. K. 2017, AJ, 154, 231
    """
    tobs = np.asarray(time, dtype=np.float64)
    mag_arr = np.asarray(mag, dtype=np.float64)
    band_arr = np.asarray(band)
    magerr_arr = np.asarray(mag_err, dtype=np.float64)

    farray = adaptive_frequency_array(tobs, minimum_frequency, maximum_frequency, dphi)
    farray_plain = np.asarray(farray, dtype=np.float64)
    parray = 1.0 / farray_plain

    rng = np.random.default_rng(random_state)
    psi = np.zeros(len(farray_plain))
    thresh = np.zeros(len(farray_plain))

    for b in np.unique(band_arr):
        mask = band_arr == b
        if np.count_nonzero(mask) < 2:
            continue
        t_b = tobs[mask]
        m_b = mag_arr[mask]
        e_b = magerr_arr[mask]
        n_b = len(m_b)

        _, fy_b = ls_periodogram(
            t_b, m_b, e_b, minimum_frequency, maximum_frequency,
            frequency=farray_plain, method=ls_method,
        )

        if _cythonc:
            theta_b = psearch_pyc.ctheta_slave(parray, m_b, t_b)
        else:
            theta_b = _ctheta_slave_jit(parray, m_b, t_b)

        psi_b = 2.0 * fy_b / theta_b
        psi += psi_b

        if n_thresh > 0:
            psi_b_sum = np.sum(psi_b)
            conf1_b = np.zeros(len(farray_plain))
            conf2_b = np.zeros(len(farray_plain))

            for _ in range(n_thresh):
                # conf1: pure Gaussian noise scaled by measurement errors
                er = e_b * rng.standard_normal(n_b)
                _, fe_b = ls_periodogram(
                    t_b, er, e_b, minimum_frequency, maximum_frequency,
                    frequency=farray_plain, method=ls_method,
                )
                if _cythonc:
                    thetaerr_b = psearch_pyc.ctheta_slave(parray, er, t_b)
                else:
                    thetaerr_b = _ctheta_slave_jit(parray, er, t_b)
                conf1a = 2.0 * fe_b / thetaerr_b
                conf1a *= psi_b_sum / np.sum(conf1a)
                conf1_b = np.maximum(conf1_b, conf1a)

                # conf2: scrambled magnitudes at original timestamps
                zr = m_b[rng.permutation(n_b)]
                _, fz_b = ls_periodogram(
                    t_b, zr, e_b, minimum_frequency, maximum_frequency,
                    frequency=farray_plain, method=ls_method,
                )
                if _cythonc:
                    thetaz_b = psearch_pyc.ctheta_slave(parray, zr, t_b)
                else:
                    thetaz_b = _ctheta_slave_jit(parray, zr, t_b)
                conf2a = 2.0 * fz_b / thetaz_b
                conf2a *= psi_b_sum / np.sum(conf2a)
                conf2_b = np.maximum(conf2_b, conf2a)

            thresh += conf1_b + conf2_b

    return farray, psi, thresh
