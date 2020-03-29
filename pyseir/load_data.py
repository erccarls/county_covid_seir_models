import os
import pandas as pd
import numpy as np
import urllib.request
import io
import zipfile
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def load_zip_get_file(url, file, decoder='utf-8'):
    """
    Load a zipfile from a URL and extract a single file.  Note that this is
    not ideal and may fail for large files since the files must fit in memory.

    Parameters
    ----------
    url: str
        URL to read from.
    file: str
        Filename to pull out of the zipfile.
    decoder: str
        Usually None for raw bytes or 'utf-8', or 'latin1'

    Returns
    -------
    file_buffer: io.BytesIO or io.StringIO
        The file buffer for the requested file if decoder is None else return
        a decoded StringIO.
    """
    remotezip = urllib.request.urlopen(url)
    zipinmemory = io.BytesIO(remotezip.read())
    zf = zipfile.ZipFile(zipinmemory)
    byte_string = zf.read(file)
    if decoder:
        string = byte_string.decode(decoder)
        return io.StringIO(string)
    else:
        return io.BytesIO(byte_string)


def cache_county_case_data():
    """
    Cache county covid case data in #PYSEIR_HOME/data.
    """
    print('Downloading covid case data')
    county_fips_map = pd.read_csv(os.path.join(DATA_DIR, 'county_state_fips.csv'), dtype='str', low_memory=False)
    case_data = pd.read_csv('https://coronadatascraper.com/timeseries-tidy.csv', low_memory=False)

    fips_merged = case_data.merge(county_fips_map, left_on=('county', 'state'), right_on=('COUNTYNAME', 'STATE'))\
              [['STCOUNTYFP', 'county', 'state', 'population', 'lat', 'long', 'date', 'type', 'value']]

    fips_merged.columns = [col.lower() for col in fips_merged.columns]
    fips_merged.to_pickle(os.path.join(DATA_DIR, 'covid_case_timeseries.pkl'))


def cache_county_metadata():
    """
    Cache 2019 census data including age distribution by state/county FIPS.

    # TODO Add pop density
    """
    print('Downloading county level population data')
    county_summary = pd.read_csv(
        'https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv',
        sep=',', encoding="ISO-8859-1", dtype='str', low_memory=False)

    df = county_summary[county_summary.YEAR == '11'][['STATE', 'COUNTY', 'CTYNAME', 'AGEGRP', 'TOT_POP']]
    df[['AGEGRP', 'TOT_POP']] = df[['AGEGRP', 'TOT_POP']].astype(int)
    list_agg = df.sort_values(['STATE', 'COUNTY', 'CTYNAME', 'AGEGRP']) \
        .groupby(['STATE', 'COUNTY', 'CTYNAME'])['TOT_POP'] \
        .apply(np.array) \
        .reset_index()
    list_agg['TOTAL'] = list_agg['TOT_POP'].apply(lambda x: x[0])
    list_agg['AGE_DISTRIBUTION'] = list_agg['TOT_POP'].apply(lambda x: x[1:])
    list_agg.drop('TOT_POP', axis=1)

    age_bins = list(range(0, 86, 5))
    age_bins += [120]
    list_agg['AGE_BIN_EDGES'] = [np.array(age_bins) for _ in
                                 range(len(list_agg))]

    list_agg.insert(0, 'stcountyfp', list_agg['STATE'] + list_agg['COUNTY'])
    list_agg = list_agg.drop(['STATE', 'COUNTY', 'TOT_POP'], axis=1)
    list_agg.columns = [col.lower() for col in list_agg.columns]
    list_agg = list_agg.rename(
        mapper={'ctyname': 'county_name', 'total': 'total_population'}, axis=1)
    list_agg.to_pickle(os.path.join(DATA_DIR, 'covid_county_metadata.pkl'))


def cache_hospital_beds():
    """
    Pulled from "Definitive"
    See: https://services7.arcgis.com/LXCny1HyhQCUSueu/arcgis/rest/services/Definitive_Healthcare_Hospitals_Beds_Hospitals_Only/FeatureServer/0
    """
    print('Downloading ICU capacity data.')
    url = 'http://opendata.arcgis.com/datasets/f3f76281647f4fbb8a0d20ef13b650ca_0.geojson'
    tmp_file = urllib.request.urlretrieve(url)[0]

    with open(tmp_file) as f:
        vals = json.load(f)
    df = pd.DataFrame([val['properties'] for val in vals['features']])
    df.rename({'FIPS': 'stcountyfp'}, axis=1)
    df.columns = [col.lower() for col in df.columns]
    df.drop(['objectid', 'state_fips', 'cnty_fips'], axis=1)
    df.to_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


def load_county_case_data():
    """
    Return county level case data. The following columns:

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'covid_case_timeseries.pkl'))


def load_county_metadata():
    """
    Return county level metadata such as age distributions, populations etc..

    Returns
    -------
    : pd.DataFrame

    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'covid_county_metadata.pkl'))


def load_hospital_data():
    """
    Return hospital level data. Note that this must be aggregated by stcountyfp
    to obtain county level estimates.

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_county_case_data()
    cache_county_metadata()
    cache_hospital_beds()


if __name__ == '__main__':
    cache_all_data()
