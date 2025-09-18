import os
import numpy as np
from astropy.table import Table
from astroquery.gaia import Gaia

def gaialc(source_id,release='dr3'):
    """ Get Gaia lightcurve """
    # only does one object at a time
    retrieval_type = 'EPOCH_PHOTOMETRY'
    data_structure = 'INDIVIDUAL'
    data_release   = 'Gaia '+str(release).upper()
    if isinstance(source_id,list):
        ids = source_id
    elif isinstance(source_id,np.ndarray):
        ids = list(source_id)
    else:
        ids = [source_id]
    datalink = Gaia.load_data(ids=ids,
                              data_release=data_release,
                              retrieval_type=retrieval_type,
                              data_structure=data_structure)
    if len(datalink)==0:
        return []
    tab = datalink[list(datalink.keys())[0]][0]
    tab = Table(tab.array)
    return tab
