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
    # Previous datasets from coronadatascraper
    # county_fips_map = pd.read_csv(os.path.join(DATA_DIR, 'county_state_fips.csv'), dtype='str', low_memory=False)
    # case_data = pd.read_csv('https://coronadatascraper.com/timeseries-tidy.csv', low_memory=False)
    #
    # fips_merged = case_data.merge(county_fips_map, left_on=('county', 'state'), right_on=('COUNTYNAME', 'STATE'))\
    #           [['STCOUNTYFP', 'county', 'state', 'population', 'lat', 'long', 'date', 'type', 'value']]
    #
    # fips_merged.columns = [col.lower() for col in fips_merged.columns]

    # NYT dataset
    county_case_data = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', dtype='str')
    county_case_data['date'] = pd.to_datetime(county_case_data['date'])
    county_case_data[['cases', 'deaths']] = county_case_data[['cases', 'deaths']].astype(int)
    county_case_data = county_case_data[county_case_data['fips'].notnull()]
    county_case_data.to_pickle(os.path.join(DATA_DIR, 'covid_case_timeseries.pkl'))


# def cache_county_metadata():
#     """
#     Cache 2019 census data including age distribution by state/county FIPS.
#
#     # TODO Add pop density
#     """
#     print('Downloading county level population data')
#     county_summary = pd.read_csv(
#         'https://www2.census.gov/programs-surveys/popest/datasets/2010-2018/counties/asrh/cc-est2018-alldata.csv',
#         sep=',', encoding="ISO-8859-1", dtype='str', low_memory=False)
#
#     df = county_summary[county_summary.YEAR == '11'][['STATE', 'COUNTY', 'CTYNAME', 'AGEGRP', 'TOT_POP']]
#     df[['AGEGRP', 'TOT_POP']] = df[['AGEGRP', 'TOT_POP']].astype(int)
#     list_agg = df.sort_values(['STATE', 'COUNTY', 'CTYNAME', 'AGEGRP']) \
#         .groupby(['STATE', 'COUNTY', 'CTYNAME'])['TOT_POP'] \
#         .apply(np.array) \
#         .reset_index()
#     list_agg['TOTAL'] = list_agg['TOT_POP'].apply(lambda x: x[0])
#     list_agg['AGE_DISTRIBUTION'] = list_agg['TOT_POP'].apply(lambda x: x[1:])
#     list_agg.drop('TOT_POP', axis=1)
#
#     age_bins = list(range(0, 86, 5))
#     age_bins += [120]
#     list_agg['AGE_BIN_EDGES'] = [np.array(age_bins) for _ in
#                                  range(len(list_agg))]
#
#     list_agg.insert(0, 'fips', list_agg['STATE'] + list_agg['COUNTY'])
#     list_agg = list_agg.drop(['COUNTY', 'TOT_POP'], axis=1)
#     list_agg.columns = [col.lower() for col in list_agg.columns]
#     list_agg = list_agg.rename(
#         mapper={'ctyname': 'county_name', 'total': 'total_population'}, axis=1)
#     list_agg.to_pickle(os.path.join(DATA_DIR, 'covid_county_metadata.pkl'))


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
    df.columns = [col.lower() for col in df.columns]
    df = df.drop(['objectid', 'state_fips', 'cnty_fips'], axis=1)
    df.to_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


def cache_mobility_data():
    """
    Pulled from https://github.com/descarteslabs/DL-COVID-19
    """
    print('Downloading mobility data.')
    url = 'https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv'

    dtypes_mapping = {
        'country_code': str,
        'admin_level': int,
        'admin1': str,
        'admin2': str,
        'fips': str,
        'samples': int,
        'm50': float,
        'm50_index': float}

    df = pd.read_csv(filepath_or_buffer=url, parse_dates=['date'], dtype=dtypes_mapping)
    df__m50 = df.query('admin_level == 2')[['fips', 'date', 'm50']]
    df__m50_index = df.query('admin_level == 2')[['fips', 'date', 'm50_index']]
    df__m50__final = df__m50.groupby('fips').agg(list).reset_index()
    df__m50_index__final = df__m50_index.groupby('fips').agg(list).reset_index()
    df__m50__final['m50'] = df__m50__final['m50'].apply(lambda x: np.array(x))
    df__m50_index__final['m50_index'] = df__m50_index__final['m50_index'].apply(lambda x: np.array(x))

    df__m50__final.to_pickle(os.path.join(DATA_DIR, 'mobility_data__m50.pkl'))
    df__m50_index__final.to_pickle(os.path.join(DATA_DIR, 'mobility_data__m50_index.pkl'))


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
    # return pd.read_pickle(os.path.join(DATA_DIR, 'covid_county_metadata.pkl'))
    return pd.read_json(os.path.join(DATA_DIR, 'county_metadata.json'), dtype={'fips': 'str'})


def load_hospital_data():
    """
    Return hospital level data. Note that this must be aggregated by stcountyfp
    to obtain county level estimates.

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


def load_mobility_data_m50():
    """
    Return mobility data without normalization

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'mobility_data__m50.pkl'))




# Ensembles need to access this 1e6 times and it makes 10ms simulations -> 100 ms otherwise.
in_memory_cache = None
def load_mobility_data_m50_index():
    """
    Return mobility data with normalization: per
    https://github.com/descarteslabs/DL-COVID-19 normal m50 is defined during
    2020-02-17 to 2020-03-07.

    Returns
    -------
    : pd.DataFrame
    """
    global in_memory_cache
    if in_memory_cache is not None:
        return in_memory_cache
    else:
        in_memory_cache = pd.read_pickle(os.path.join(DATA_DIR, 'mobility_data__m50_index.pkl')).set_index('fips')

    return in_memory_cache.copy()


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_county_case_data()
    # cache_county_metadata()
    cache_hospital_beds()
    cache_mobility_data()


if __name__ == '__main__':
    cache_all_data()
