import numpy as np
from pyseir import load_data


class ParameterEnsembleGenerator:

    def __init__(self, fips, N_samples, t_list, ventilators_per_icu_bed=.75,
                 I_initial=5, infected_to_case_count_ratio=1,
                 suppression_policy=None):
        """
        Generate ensembles of parameters for SEIR modeling.

        Parameters
        ----------
        fips: str
            County fips code.
        N_samples: int
            Integer number of samples to generate.
        t_list: array-like
            Array of times to integrate against.
        ventilators_per_icu_bed: float
            Number of ventilators rto assume per ICU bed available.
        I_initial: int
            Initial infected case count to consider.
        infected_to_case_count_ratio: float
            Multiplier on the ratio of tested cases vs untested cases at the
            time of the simulation start. Note that asymptomatic cases are
            already modeled.
        suppression_policy: callable(t): pyseir.model.suppression_policy
            Suppression policy to apply.
        """
        self.fips = fips
        self.N_samples = N_samples
        self.ventilators_per_icu_bed = ventilators_per_icu_bed
        self.I_initial = I_initial
        self.infected_to_case_count_ratio = infected_to_case_count_ratio
        self.suppression_policy = suppression_policy
        self.t_list = t_list
        county_metadata = load_data.load_county_metadata()
        hospital_bed_data = load_data.load_hospital_data()

        # TODO: Some counties do not have hospitals. Need to figure out what to do here.
        hospital_bed_data = hospital_bed_data[
            ['fips',
             'num_licensed_beds',
             'num_staffed_beds',
             'num_icu_beds',
             'bed_utilization',
             'potential_increase_in_bed_capac']].groupby('fips').sum()
        self.county_metadata_merged = county_metadata.merge(hospital_bed_data, on='fips', how='left').set_index('fips').loc[fips].to_dict()

    def sample_seir_parameters(self, override_params=None):
        """
        Generate N_samples of parameter values from the priors listed below.

        Parameters
        ----------
        override_params: dict()
            Individual parameters can be overridden here.

        Returns
        -------
        : list(dict)
            List of parameter sets to feed to the simulations.
        """
        override_params = override_params or dict()

        parameter_sets = []
        for _ in range(self.N_samples):

            hospitalization_rate_general = np.random.uniform(low=.05, high=0.2)
            fraction_asymptomatic = np.random.uniform(0.4, .6)
            # https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
            parameter_sets.append(dict(
                t_list=self.t_list,
                N=self.county_metadata_merged['total_population'],
                A_initial=fraction_asymptomatic * self.I_initial / (1 - fraction_asymptomatic), # assume no asymptomatic cases are tested.
                I_initial=self.I_initial,
                R_initial=0,
                E_initial=0,
                D_initial=0,
                HGen_initial=0,
                HICU_initial=0,
                HICUVent_initial=0,
                suppression_policy=self.suppression_policy,
                R0=np.random.uniform(low=3, high=4.5),            # Imperial College
                hospitalization_rate_general=hospitalization_rate_general,
                hospitalization_rate_icu=np.random.normal(loc=.25, scale=0.05) * hospitalization_rate_general,
                fraction_icu_requiring_ventilator=np.random.uniform(low=0.75, high=0.9),
                sigma=1 / np.random.normal(loc=5.1, scale=0.86),  # Imperial college
                kappa=1,
                gamma=fraction_asymptomatic,
                symptoms_to_hospital_days=np.random.normal(loc=5, scale=1),
                symptoms_to_mortality_days=np.random.normal(loc=18.8, scale=.45), # Imperial College
                hospitalization_length_of_stay_general=np.random.normal(loc=7, scale=2),
                hospitalization_length_of_stay_icu=np.random.normal(loc=16, scale=3),
                hospitalization_length_of_stay_icu_and_ventilator=np.random.normal(loc=17, scale=3),
                mortality_rate=np.random.normal(loc=0.0075, scale=0.0025),
                mortality_rate_no_ICU_beds=0.85,
                mortality_rate_no_ventilator=1.0,
                mortality_rate_no_general_beds=0.6,

                beds_general=self.county_metadata_merged.get('num_licensed_beds', 0)
                             - self.county_metadata_merged.get('bed_utilization', 0)
                             + self.county_metadata_merged.get('potential_increase_in_bed_capac', 0),
                beds_ICU=self.county_metadata_merged.get('num_icu_beds', 0),
                ventilators=self.county_metadata_merged.get('num_icu_beds', 0) * self.ventilators_per_icu_bed,
            ))


        for parameter_set in parameter_sets:
            parameter_set.update(override_params)

        return parameter_sets
