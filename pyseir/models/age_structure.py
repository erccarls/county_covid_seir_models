import re
import os
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from string import Template
from pyseir import load_data

THIS_FILE_PATH = os.path.dirname(os.path.realpath('__file__'))

def get_age_distribution(fips):
    """
    Extract age groups and corresponding age distribution for given fips.

    Parameters
    ----------

    """

    county_metadata = load_data.load_county_metadata()
    age_bin_edges = county_metadata.set_index('fips').loc[fips]['age_bin_edges']
    age_distribution = county_metadata.set_index('fips').loc[fips]['age_distribution']
    age_dist = pd.DataFrame({'age_bin_edges': age_bin_edges[:-1],
                             'age_distribution': age_distribution})
    return age_dist

def extract_contact_matrix(config, age_dist):
    """
    Extract contact matrices from polymod service data.

    Parameters
    ----------
    config : dict
        Used to extract contact rate matrix using R 'socialmixr' package.
        Example of config:
            fips : '06075'
            contact_matrices_r_script_path: '~/covid/county_covid_seir_models/pyseir/models/contact_matrices.r'
            r_substitution:
               country: 'United Kingdom',
               num_sample: 5,
               weight_by_dayofweek: 'TRUE',
               matrices: '$matrices',
               matrix: '$matrix'
    age_dist :

    """
    age_distribution = age_dist['age_distribution']
    age_bin_edges = age_dist['age_bin_edges']

    config['r_substitution']['age_bin_edges'] = ','.join([str(n) for n in age_bin_edges])
    config['r_substitution']['age_distribution'] = ','.join([str(n) for n in age_distribution])

    r_script_path = config['contact_matrices_r_script_path']
    r_script = Template(open(r_script_path, 'r').read()).substitute(config['r_substitution'])

    pandas2ri.activate()
    ro = robjects
    ro.r(r_script)
    contact_matrix = ro.r('mr')
    contact_matrix = ro.conversion.rpy2py(contact_matrix)
    extract_age_bin_edges = lambda x: int(re.sub("[b'[)+]", '', str(x)).split(',')[0])
    age_bin_edges = sorted([extract_age_bin_edges(col) for col in contact_matrix.columns])

    return age_bin_edges, contact_matrix


def get_age_group_size_for_bin_edges(age_dist, age_bin_edges):
    """
    Get size of age groups in the socialmatrix query results from county metadata.
    Combine age groups if they fall in the same bin (those in oldest group), i.e. 75+ includes 75~80, 80~85 and 85+.

    Parameters
    ----------
    age_dist : pd.DataFrame
        Contains:
        - age_bin_edges: lower age limits from county metadata
        - age_distribution: population size fall between two sequential age bin edges from county metadata
    age_bin_edges : list
        Bin edges identified by polymod survey data, can be same or fewer than age_bin_edges in county metadata.

    Returns
    -------

    """
    age_group_size = np.zeros(age_dist.shape[0])
    age_group_idx = None
    for n in age_dist.index:
        if age_dist.loc[n]['age_bin_edges'] in age_bin_edges:
            age_group_size[n] += age_dist.loc[n]['age_distribution']
            age_group_idx = n
        else:
            age_group_size[age_group_idx] += age_dist.loc[n]['age_distribution']
    age_group_size = age_group_size[age_group_size > 0]
    return age_group_size

def get_age_distribution_and_contact_matrix(config):
    """

    """
    age_dist = get_age_distribution(config['fips'])
    age_bin_edges, contact_matrix = extract_contact_matrix(config, age_dist)
    age_group_sizes = get_age_group_size_for_bin_edges(age_dist, age_bin_edges)
    return age_bin_edges, age_group_sizes, contact_matrix

