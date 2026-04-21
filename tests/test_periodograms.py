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
PERIOD_TOL  = 0.05          # 5 % fractional tolerance
LK_NBINS    = 1000          # frequency resolution for LK search
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
    """Lomb-Scargle recovers the known period within tolerance."""
    freq, power = star.ls_periodogram()
    period = 1.0 / freq[np.argmax(power)]
    err = abs(period - TRUE_PERIOD) / TRUE_PERIOD
    print(f"\nLS  — true: {TRUE_PERIOD:.4f} d  |  recovered: {period:.4f} d  "
          f"|  error: {err*100:.1f}%")
    assert err < PERIOD_TOL, (
        f"LS period {period:.4f} d deviates {err*100:.1f}% from "
        f"true period {TRUE_PERIOD:.4f} d (tolerance {PERIOD_TOL*100:.0f}%)"
    )


@pytest.mark.integration
def test_lk_period(star):
    """Lafler-Kinman recovers the known period within tolerance."""
    freq, theta = star.lk_periodogram(nbins=LK_NBINS)
    period = 1.0 / freq[np.argmin(theta)]
    err = abs(period - TRUE_PERIOD) / TRUE_PERIOD
    print(f"\nLK  — true: {TRUE_PERIOD:.4f} d  |  recovered: {period:.4f} d  "
          f"|  error: {err*100:.1f}%")
    assert err < PERIOD_TOL, (
        f"LK period {period:.4f} d deviates {err*100:.1f}% from "
        f"true period {TRUE_PERIOD:.4f} d (tolerance {PERIOD_TOL*100:.0f}%)"
    )


# ---------------------------------------------------------------------------
# Plot tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_plot_periodograms(star):
    """Save LS and LK periodograms to tests/periodograms.png."""
    freq_ls, power = star.ls_periodogram()
    freq_lk, theta = star.lk_periodogram(nbins=LK_NBINS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.plot(1 / freq_ls, power, color='steelblue', lw=0.8)
    ax1.axvline(TRUE_PERIOD, color='crimson', ls='--', lw=1.2,
                label=f'True period ({TRUE_PERIOD} d)')
    ax1.set(xlabel='Period [days]', ylabel='LS power',
            title=f'Lomb-Scargle  —  {OBJID}')
    ax1.legend()

    ax2.plot(1 / freq_lk, 1 / theta, color='darkorange', lw=0.8)
    ax2.axvline(TRUE_PERIOD, color='crimson', ls='--', lw=1.2,
                label=f'True period ({TRUE_PERIOD} d)')
    ax2.set(xlabel='Period [days]', ylabel='1/θ (higher = better)',
            title=f'Lafler-Kinman  —  {OBJID}')
    ax2.legend()

    plt.tight_layout()
    outfile = TESTS_DIR / 'periodograms.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"\nSaved {outfile}")


@pytest.mark.integration
def test_plot_phase_folded(star):
    """Save phase-folded light curve (all filters on one plot) to tests/phase_folded.png."""
    freq_ls, power = star.ls_periodogram()
    period = 1.0 / freq_ls[np.argmax(power)]

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
           title=f'{OBJID}  —  P = {period:.4f} d  (LS)')
    ax.legend(title='Filter', ncol=2)
    plt.tight_layout()
    outfile = TESTS_DIR / 'phase_folded.png'
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"\nSaved {outfile}")
