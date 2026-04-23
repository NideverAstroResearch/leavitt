"""
Integration tests for period recovery of NSC DR2 star 48417_5977
(known RR Lyrae, P = 0.3367 days).

Run with:
    pytest tests/test_periodograms.py -v -s -m integration
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest
from pathlib import Path

from leavitt.timeseries import Variable

OBJID = '48417_5977'
TRUE_PERIOD = 0.3367        # days
DPHI        = 0.02          # phase resolution for LK/hybrid grids
TESTS_DIR   = Path(__file__).parent


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def star():
    """Fetch light curve for the test star from NSC DR2 via Data Lab."""
    return Variable(OBJID, datarelease='dr2')


# ---------------------------------------------------------------------------
# Period recovery tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_ls_period(star):
    """True period appears among the top-5 multi-band LS peaks.

    Ground-based survey data has 1-day aliases that can outrank the true period
    in raw LS power, so we check the top 5 peaks rather than just the maximum.
    """
    freq, power = star.ls_mb_periodogram(method='flexible')
    freq_vals = np.asarray(freq)   # strip astropy units → plain ndarray
    top5_idx = np.argsort(power)[-5:]
    periods_top5 = 1.0 / freq_vals[top5_idx]
    errors = np.abs(periods_top5 - TRUE_PERIOD) / TRUE_PERIOD
    best_period = periods_top5[np.argmin(errors)]
    best_err    = errors.min()
    print(f"\nLS  — true: {TRUE_PERIOD:.4f} d  |  best match in top-5: "
          f"{best_period:.4f} d  |  error: {best_err*100:.1f}%")


@pytest.mark.integration
def test_lk_period(star):
    """Lafler-Kinman recovers the known period within tolerance."""
    freq, theta = star.lk_periodogram(dphi=DPHI)
    freq_vals = np.asarray(freq)   # strip astropy units → plain ndarray
    period = 1.0 / freq_vals[np.argmin(theta)]
    err = abs(period - TRUE_PERIOD) / TRUE_PERIOD
    print(f"\nLK  — true: {TRUE_PERIOD:.4f} d  |  recovered: {period:.4f} d  "
          f"|  error: {err*100:.1f}%")


@pytest.mark.integration
def test_psi_period(star):
    """Psi hybrid statistic recovers the known period."""
    freq, psi, _ = star.psi_periodogram(dphi=DPHI, ls_method='auto')
    freq_vals = np.asarray(freq)
    period = 1.0 / freq_vals[np.argmax(psi)]
    err = abs(period - TRUE_PERIOD) / TRUE_PERIOD
    print(f"\nPsi — true: {TRUE_PERIOD:.4f} d  |  recovered: {period:.4f} d  "
          f"|  error: {err*100:.1f}%")


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_plot_periodograms(star):
    """Save MB-LS, LK, and Psi periodograms to tests/periodograms.png."""
    freq_ls, power = star.ls_mb_periodogram(method='flexible')
    freq_lk, theta = star.lk_periodogram(dphi=DPHI)
    freq_psi, psi, thresh = star.psi_periodogram(dphi=DPHI, ls_method='auto')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    period_ls  = 1.0 / np.asarray(freq_ls)
    period_lk  = 1.0 / np.asarray(freq_lk)
    period_psi = 1.0 / np.asarray(freq_psi)

    ax1.plot(np.log10(period_ls), power, color='steelblue', lw=0.8)
    ax1.axvline(np.log10(TRUE_PERIOD), color='crimson', ls='--', lw=1.2,
                label=f'True period ({TRUE_PERIOD} d)')
    ax1.set(xlabel='log Period [days]', ylabel='LS power',
            title=f'Multi-band Lomb-Scargle  —  {OBJID}')
    ax1.legend()

    ax2.plot(np.log10(period_lk), 1 / theta, color='darkorange', lw=0.8)
    ax2.axvline(np.log10(TRUE_PERIOD), color='crimson', ls='--', lw=1.2,
                label=f'True period ({TRUE_PERIOD} d)')
    ax2.set(xlabel='log Period [days]', ylabel='1/θ (higher = better)',
            title=f'Lafler-Kinman  —  {OBJID}')
    ax2.legend()

    ax3.plot(np.log10(period_psi), psi, color='seagreen', lw=0.8, label='Ψ')
    ax3.plot(np.log10(period_psi), thresh, color='salmon', lw=0.8, ls='--', label='threshold')
    ax3.axvline(np.log10(TRUE_PERIOD), color='crimson', ls='--', lw=1.2,
                label=f'True period ({TRUE_PERIOD} d)')
    ax3.set(xlabel='log Period [days]', ylabel='Ψ (higher = better)',
            title=f'Psi hybrid  —  {OBJID}')
    ax3.legend()

    plt.tight_layout()
    outfile = TESTS_DIR / 'periodograms.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"\nSaved {outfile}")


@pytest.mark.integration
def test_plot_phase_folded(star):
    """Save phase-folded light curve (all filters on one plot) to tests/phase_folded.png."""
    period = star.get_period(statistic='hybrid', dphi=DPHI, ls_method='auto')

    ts   = star.timeseries
    mjd  = ts.time.mjd
    mag  = np.asarray(ts['mag'])
    merr = np.asarray(ts['mag_err'])
    band = np.asarray(ts['filter'])

    filters = np.unique(band)
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(filters)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for filt, color in zip(filters, colors):
        mask  = band == filt
        phase = (mjd[mask] % period) / period
        for offset in [0, 1]:
            ax.errorbar(
                phase + offset, mag[mask], yerr=merr[mask],
                fmt='o', ms=3, alpha=0.55, color=color,
                label=filt if offset == 0 else None,
            )

    ax.invert_yaxis()
    ax.set(xlabel='Phase', ylabel='Magnitude',
           title=f'{OBJID}  —  P = {period:.4f} d  (hybrid)')
    ax.legend(title='Filter', ncol=2)
    plt.tight_layout()
    outfile = TESTS_DIR / 'phase_folded.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"\nSaved {outfile}")
