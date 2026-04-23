import numpy as np
from astropy.timeseries import TimeSeries
from astropy.time import Time
from astropy.stats import sigma_clip as astropy_sigma_clip
import astropy.units as u
from leavitt.query import *
from leavitt.utils import *
from leavitt import utils
from leavitt import periodograms

# Data Lab
from dl import authClient as ac, queryClient as qc



class Variable:
    """
    This class gives the core functionality to look for
    and analyse variability inside of the NSC. It operates 
    on the basis of stars being objects.
    
    Parameters
    ----------
    objid: str
        Unique object ID inside of the NSC catalog.
    period: float, optional
        Period of the variable star, if known.
    variclass: str, optional
        Variability class (e.g. RRLab, Classical Cepheid, etc.). No functionality implemented for it at the moment.
    timeseries: TimeSeries object, optional
        Object with data including times and magnitudes. Additional data is also possible using the Astropy TimeSeries functionality.
        If not given, the data will be retrieved automatically based on the given Object ID and data release.
    datarelease: str, optional
        Data release from which the data comes from, or where it should be taken from. Default is DR2.
    max_magerr: float, optional
        Maximum magnitude error allowed. Observations with mag_err > max_magerr are excluded before
        any periodogram calculation. Default is 0.2.
    """
    
    def __init__(self, objid, period=None, variclass=None, timeseries=None, datarelease="dr2", catalog="NSC", sigma=3.0, max_magerr=0.2):
        """ 
        Initialize the Star object. If data for the time series is not given, then
        get it from Datalab.
        """
        
        accepted_variclass = ['Cepheid', 'RRLyrae', 'RRLab', 'RRLc', 'Mira', 'LPV']
        
        self.objid = objid
        self.period = period
        self.variclass = variclass
        self.catalog = catalog
        self.datarelease = datarelease
        self.sigma = sigma
        self.max_magerr = max_magerr
        if timeseries is None:
            self.timeseries = self.get_timeseries_data()
        else:
            self.timeseries = timeseries
        

    def get_timeseries_data(self, datarelease=None, datalab_token=None):
        """
        For a given Object ID, return time series data.
        
        Parameters
        ----------
        objid : str, optional
            ID of the star. Keep in mind it does not carry over different data releases.
        datarelease: str, optional
            Data release from which to get the data from. Default is the class default, currently "dr2".
        """
        
        if datarelease==None: datarelease = self.datarelease

        if self.catalog =="NSC":
            if datarelease.lower()=='dr2' or datarelease.lower()=='dr1':
                mag_name = 'mag_auto'
                magerr_name = 'magerr_auto'
            else:
                mag_name = 'mag'
                magerr_name = 'magerr'
            
            query = f"""SELECT m.mjd,m.{mag_name},m.{magerr_name},m.filter,e.exptime 
                    FROM nsc_dr2.meas AS m JOIN nsc_dr2.exposure as e ON m.exposure=e.exposure
                    WHERE m.objectid='{self.objid}'"""
            # f"SELECT mjd,{mag_name},{magerr_name},filter FROM nsc_{datarelease}.meas WHERE objectid='{self.objid}'"
            table_res = qc.query(query,fmt='table')
            timeseries_obj = TimeSeries(data={'mag':table_res[mag_name],'mag_err':table_res[magerr_name],'filter':table_res['filter'],'exptime':table_res['exptime']},time=Time(table_res['mjd'], format='mjd'))
            
        elif self.catalog=="Gaia":
            latest_gaia_release = "DR3"
            if datarelease=="dr3" or datarelease=="dr1" or datarelease=="DR1":
                datarelease = latest_gaia_release
            elif datarelease=="dr2":
                datarelease = "DR2"
                
            table_res = gaialc(int(self.objid),release=datarelease)
            table_res = table_res[table_res['rejected_by_photometry']==False]
            
            if datarelease=="DR3":
                mag_g_name = 'g_transit_mag'
                mag_bp_name = 'bp_mag'
                mag_rp_name = 'rp_mag'
                g_flux_error_name = 'g_transit_flux_over_error'
                bp_flux_error_name = 'bp_flux_over_error'
                rp_flux_error_name = 'rp_flux_over_error'
                g_time_name = 'g_transit_time'
                bp_time_name = 'bp_obs_time'
                rp_time_name = 'rp_obs_time'
                # Mags
                g_mask = ~table_res[mag_g_name].mask
                g_mag = table_res[mag_g_name][g_mask]
                bp_mask = ~table_res[mag_bp_name].mask
                bp_mag = table_res[mag_bp_name][bp_mask]
                rp_mask = ~table_res[mag_rp_name].mask
                rp_mag = table_res[mag_rp_name][rp_mask]
                mags = np.hstack([g_mag,bp_mag,rp_mag])
                #print(mags)
                # Mag_errs
                g_mag_err = 1.0857362047581294/table_res[g_flux_error_name][g_mask]
                bp_mag_err = 1.0857362047581294/table_res[bp_flux_error_name][bp_mask]
                rp_mag_err = 1.0857362047581294/table_res[rp_flux_error_name][rp_mask]
                mag_err = np.hstack([g_mag_err,bp_mag_err,rp_mag_err])
                # Bands
                gs = np.array(['G' for x in range(len(g_mag))],dtype=str)
                bps = np.array(['BP' for x in range(len(bp_mag))],dtype=str)
                rps = np.array(['RP' for x in range(len(rp_mag))],dtype=str)
                filters = np.hstack([gs,bps,rps])
                # Times
                #print(table_res)
                times = np.hstack([table_res[g_time_name][g_mask],table_res[bp_time_name][bp_mask],table_res[rp_time_name][rp_mask]])
                #print(times)
                
                
            elif datarelease=="DR2":
                mag_name = 'mag'
                flux_error_name = 'flux_over_error'
                time_name = 'time'
                filter_name = 'band'
                filters = table_res[filter_name]
                mags = table_res[mag_name]
                mag_err = 1.0857362047581294/table_res[flux_error_name]
                times = table_res[time_name]
            else:
                raise ValueError(datarelease+" not supported")
                
            
            timeseries_obj = TimeSeries(data={'mag':mags,'mag_err':mag_err,'filter':filters}, 
                                        time=Time(times+2455197.5, format='jd', scale='tcb'))
        
        return timeseries_obj
         
    def issp(self):
        """
        Function to determine whether a variable has or could have
        a short or long period. Mostly for internal use, to establish
        period search ranges.
        
        It will use the actual period value if available, if not the variability class
        is used. If nothing is available, a short period is assumed.
        
        Returns
        -------
        shortperiod: bool
            True if the variable has a short period, False if not.
        """
        
        if self.period is not None:
            period_days = self.period.to(u.day).value if hasattr(self.period, 'unit') else self.period
            if period_days <= 10:
                return True
            else:
                return False
            
        if self.variclass in ['Cepheid', 'RRLyrae', 'RRLab', 'RRLc']:
            return True
        elif self.variclass=='Mira' or self.variclass=='LPV':
            return False
        else:
            return True
        
    def franges(self):
        '''
        Determine the frequency search range from the data time baseline.

        The minimum frequency is set to 2/tspan so that at least two full
        cycles are required for period detection, consistent with the adaptive
        frequency grid convention (Saha & Vivas 2017).  The maximum frequency
        depends on whether the star is classified as short- or long-period.

        Returns
        -------
        minimum_frequency : float
            Minimum frequency in 1/day.
        maximum_frequency : float
            Maximum frequency in 1/day.
        '''
        tspan = np.max(self.timeseries.time.mjd) - np.min(self.timeseries.time.mjd)
        minimum_frequency = 2.0 / tspan

        if self.issp():
            maximum_frequency = 1.0 / 0.04   # 25 cycles/day
        else:
            maximum_frequency = 1.0 / 50.0   # 0.02 cycles/day

        return minimum_frequency, maximum_frequency
        
    
    def _clean_timeseries(self):
        """Return a sigma-clipped view of the timeseries.

        Clipping is applied per band using ``self.sigma``.  If ``self.sigma``
        is None, the full timeseries is returned unchanged.
        """
        ts = self.timeseries
        mag_err = np.asarray(ts['mag_err'])
        keep = (mag_err >= 0.0) & (mag_err <= self.max_magerr)
        if self.sigma is not None:
            for band in np.unique(ts['filter']):
                idx = np.where(np.asarray(ts['filter']) == band)[0]
                clipped = astropy_sigma_clip(np.asarray(ts['mag'][idx]), sigma=self.sigma)
                keep[idx[clipped.mask]] = False
        return ts[keep]

    def frequency_array(self, nbins=100, minimum_frequency=None, maximum_frequency=None):
        """Return a linear frequency array spanning the star's expected frequency range."""
        if minimum_frequency is None or maximum_frequency is None:
            minimum_frequency, maximum_frequency = self.franges()
        return periodograms.frequency_array(minimum_frequency, maximum_frequency, nbins)
        
        
    def ls_mb_periodogram(self, method='fast', normalization='standard', minimum_frequency=None, maximum_frequency=None):
        """
        Calculates a multi-band Lomb-Scargle periodogram
        based on the data stored in timeseries.

        Returns
        -------
        frequency : ndarray
        power : ndarray
        """
        if minimum_frequency is None or maximum_frequency is None:
            minimum_frequency, maximum_frequency = self.franges()
        ts = self._clean_timeseries()
        return periodograms.ls_mb_periodogram(
            ts.time.mjd, ts['mag'],
            ts['filter'], ts['mag_err'],
            minimum_frequency, maximum_frequency,
            method=method, normalization=normalization,
        )
    
    def ls_periodogram(self, band=None, method='auto', normalization='standard', minimum_frequency=None, maximum_frequency=None):
        """
        Calculates a Lomb-Scargle periodogram for a single band,
        based on the data stored in timeseries.

        Returns
        -------
        frequency : ndarray
        power : ndarray
        """
        if minimum_frequency is None or maximum_frequency is None:
            minimum_frequency, maximum_frequency = self.franges()
        ts = self._clean_timeseries()
        if band is None:
            band = utils.most_frequent(ts['filter'])
        sel = ts[np.asarray(ts['filter']) == band]
        return periodograms.ls_periodogram(
            sel.time.mjd, sel['mag'], sel['mag_err'],
            minimum_frequency, maximum_frequency,
            method=method, normalization=normalization,
        )

    
    def psi_periodogram(self, minimum_frequency=None, maximum_frequency=None,
                        dphi=0.02, ls_method='auto', n_thresh=1, random_state=None):
        """
        Calculates the Psi hybrid periodogram (Saha & Vivas 2017) combining
        single-band Lomb-Scargle and Lafler-Kinman per band on a shared frequency grid.

        Returns
        -------
        frequency : ndarray
        psi : ndarray
            Psi statistic at each frequency (higher = better period).
        thresh : ndarray
            Noise threshold at each frequency. All zeros when ``n_thresh=0``.
        """
        if minimum_frequency is None or maximum_frequency is None:
            minimum_frequency, maximum_frequency = self.franges()
        ts = self._clean_timeseries()
        return periodograms.psi_periodogram(
            ts.time.mjd, ts['mag'],
            ts['filter'], ts['mag_err'],
            minimum_frequency, maximum_frequency,
            dphi=dphi, ls_method=ls_method, n_thresh=n_thresh, random_state=random_state,
        )

    def lk_periodogram(self, minimum_frequency=None, maximum_frequency=None, dphi=0.02):
        """
        Calculates a Lafler-Kinman periodogram based on the data stored in timeseries.

        Returns
        -------
        frequency : ndarray
        theta : ndarray
            LK statistic at each frequency (lower = better period).
        """
        if minimum_frequency is None or maximum_frequency is None:
            minimum_frequency, maximum_frequency = self.franges()
        ts = self._clean_timeseries()
        return periodograms.lk_periodogram(
            ts.time.mjd, ts['mag'],
            minimum_frequency, maximum_frequency, dphi=dphi,
        )
    
    
    def get_folded_ts(self, period=None):
        """
        Folds time series data according to the period of the star.
        
        Returns
        -------
        phase : ndarray
            
        """
        
        if period==None: period = self.period
        
        try:
            phase = utils.phase_fold(self.timeseries['time'], period)
        except:
            print('The star has no calculated period.')
            return None
            
        return phase
    
    def get_period(self, statistic='hybrid', dphi=0.02, ls_method='auto',
                   n_thresh=1, random_state=None,
                   minimum_frequency=None, maximum_frequency=None):
        """
        Compute a periodogram with the chosen statistic and return the
        best-fit period, also setting ``self.period``.

        Parameters
        ----------
        statistic : {'hybrid', 'ls', 'ls_mb', 'lk'}, optional
            Periodogram method to use.  Default is ``'hybrid'`` (Psi).
        dphi : float, optional
            Phase resolution for LK and hybrid frequency grids.  Default is 0.02.
        ls_method : str, optional
            LS algorithm passed to each per-band Lomb-Scargle call.  Default is ``'auto'``.
        n_thresh : int, optional
            Monte Carlo iterations for the Psi noise threshold.  Only used when
            ``statistic='hybrid'``.  Set to 0 to skip.  Default is 1.
        random_state : int or None, optional
            Random seed for threshold estimation.  Default is None.
        minimum_frequency : float, optional
        maximum_frequency : float, optional

        Returns
        -------
        period : float
            Most likely period in days.
        """
        fkw = dict(minimum_frequency=minimum_frequency,
                   maximum_frequency=maximum_frequency)

        if statistic == 'ls':
            freq, stat = self.ls_periodogram(**fkw)
            period = 1.0 / np.asarray(freq)[np.argmax(stat)]
        elif statistic == 'ls_mb':
            freq, stat = self.ls_mb_periodogram(**fkw)
            period = 1.0 / np.asarray(freq)[np.argmax(stat)]
        elif statistic == 'lk':
            freq, stat = self.lk_periodogram(dphi=dphi, **fkw)
            period = 1.0 / np.asarray(freq)[np.argmin(stat)]
        elif statistic == 'hybrid':
            freq, stat, _ = self.psi_periodogram(
                dphi=dphi, ls_method=ls_method,
                n_thresh=n_thresh, random_state=random_state, **fkw,
            )
            period = 1.0 / np.asarray(freq)[np.argmax(stat)]
        else:
            raise ValueError(
                f"Unknown statistic '{statistic}'. "
                "Choose from 'ls', 'ls_mb', 'lk', 'hybrid'."
            )

        self.period = period
        return period
