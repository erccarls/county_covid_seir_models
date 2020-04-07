import pandas as pd
import numpy as np
import urllib
import zipfile
import io


odds_dict = dict(
    asthma=5.4,
    afib=5.4,
    cancer=10,
    copd=5.4,
    chronic_kidney=6,
    diabetes=2.85,
    hiv_aids=20,
    heart_failure=21.4,
    hepatitis=5,
    hyperlipidemia=3.05,
    hypertension=3.05,
    ischemic_heart_disease=21.4,
    stroke=3
)
disease_list = list(odds_dict.keys())
odds_dict['assumed_mean_age'] = 1.14


def get_cms_data():
    """
    Retrieves a zip file of county level pre-existing condition rates from CMS
    and prepares it for use. Condition Rates are reported as percentages.

    :return: pandas.DataFrame
        A dataframe containing condition rates broken down by age
    """

    tmp_file = urllib.request.urlretrieve(
        'https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Chronic-Conditions/Downloads/CC_Prev_State_County_Age.zip')[
        0]

    with open(tmp_file, 'rb') as f:
        zip_data = zipfile.ZipFile(f)
        excel_bytes = io.BytesIO(zip_data.read(
            'County_Table_Chronic_Conditions_Prevalence_by_Age_2017.xlsx'))

        disease_old = pd.read_excel(excel_bytes,
                                    sheet_name='Beneficiaries 65 Years and Over',
                                    skiprows=5, na_values=['* '],)
        disease_old['assumed_mean_age'] = 70
        disease_young = pd.read_excel(excel_bytes,
                                      sheet_name='Beneficiaries Less than 65 Year',
                                      skiprows=5, na_values=['* '])
        disease_young['assumed_mean_age'] = 40

        # fills NA with national averages when missing data
        disease_young = disease_young.fillna(dict(disease_young.iloc[0]))
        disease_old = disease_old.fillna(dict(disease_young.iloc[0]))
        df = pd.concat([disease_old, disease_young])

    col_names = ['State', 'County', 'FIPS_code', 'Alcohol Abuse',
       "Alzheimer's Disease/Dementia", 'Arthritis', 'Asthma',
       'Atrial Fibrillation', 'Autism Spectrum Disorders', 'Cancer',
       'Chronic Kidney Disease', 'COPD', 'Depression ', 'Diabetes',
       'Drug Abuse/Substance Abuse', 'HIV/AIDS', 'Heart Failure',
       'Hepatitis (Chronic Viral B & C)', 'Hyperlipidemia', 'Hypertension',
       'Ischemic Heart Disease', 'Osteoporosis',
       'Schizophrenia/Other Psychotic Disorders', 'Stroke']
    need_cols = ['State', 'County', 'FIPS_code',
       'Asthma',
       'Atrial Fibrillation', 'Cancer',
       'COPD', 'Chronic Kidney Disease', 'Diabetes',
       'HIV/AIDS', 'Heart Failure',
       'Hepatitis (Chronic Viral B & C)', 'Hyperlipidemia', 'Hypertension',
       'Ischemic Heart Disease',
       'Stroke', 'assumed_mean_age']

    df.columns = col_names + ['assumed_mean_age']
    df = df[need_cols]

    for col in ['State', 'County']:
        df[col] = df[col].apply(lambda x: x.strip())

    df.columns = ['state', 'county', 'fips'] + disease_list + ['assumed_mean_age']

    return df


def get_overall_risk(odds_dict=odds_dict):
    """
    Calculates log odds covid risk from CMS data and appends that information to
    a dataframe.

    :param odds_dict: dict
        Keys are the column names for the CMS df of any columns that model
        parameters, values are their coefficients.

    :return: pandas.DataFrame
        The data with a column, "log_odds_risk", that represents log odds of the
        COVID severity based on modeled demographics and prevalences.
    """

    df = get_cms_data()
    df['log_odds_risk'] = np.dot(df[list(odds_dict.keys())],
                                 np.log(list(odds_dict.values())))

    return df


if __name__ == '__main__':

    df = get_overall_risk()
    df.to_csv('county_level_covid_risk.csv', index=False)


