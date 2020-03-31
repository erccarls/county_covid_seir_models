import os
from pyseir.load_data import load_county_metadata
from pyseir import OUTPUT_DIR
import pandas as pd
from datetime import datetime


def get_t0(fips):
    """
    Given FIPS return a datetime object with t0(C=N) cases.

    Parameters
    ----------
    fips

    Returns
    -------
    : datetime
    """
    county_metadata = load_county_metadata().set_index('fips')
    state = county_metadata.loc[fips]['state']
    fit_results = os.path.join(OUTPUT_DIR, state, f'summary__{state}_imputed_start_times.pkl')
    return datetime.fromtimestamp(pd.read_pickle(fit_results).set_index('fips').loc[fips]['t0_date'].timestamp())
