import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class SEIRModel:

    def __init__(self,
                 N,
                 t_list,
                 suppression_policy,
                 I_initial=1,
                 R_initial=0,
                 E_initial=0,
                 HGen_initial=0,
                 HICU_initial=0,
                 HICUVent_initial=0,
                 D_initial=0,
                 R0=2.4,
                 alpha=1 / 4.58,
                 gamma=1 / 2.09,
                 hospitalization_rate_general=0.11,
                 hospitalization_rate_icu=0.04,
                 mortality_rate=0.005,
                 symptoms_to_hospital_days=5,
                 symptoms_to_mortality_days=13,
                 hospitalization_length_of_stay_general=8,
                 hospitalization_length_of_stay_icu=8,
                 hospitalization_length_of_stay_icu_and_ventilator=12,
                 fraction_icu_requiring_ventilator=0.53,
                 beds_general=30,
                 beds_ICU=15,
                 ventilators=10):
        """
        This class implements a SEIR-like compartmental epidemic model
        consisting of SEIR states plus death, and hospitalizations.

        In the diff eq modeling, these parameters are assumed exponentially
        distributed and modeling occurs in the thermodynamic limit, i.e. we do
        not perform monte carlo for individual cases.

        Model Refs:
         - https://arxiv.org/pdf/2002.06563.pdf

        Need more details on hospitalization parameters...

        Imperial college has more pessimistic numbers.
        1. https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf

        UW tends to have more optimistic numbers
        2. http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf

        Parameters
        ----------
        N: int
            Total population
        t_list: array-like
            Array of timesteps. Usually these are spaced daily.
        suppression_policy: callable
            Suppression_policy(t) should return a scalar in [0, 1] which
            represents the contact rate reduction from social distancing.
        I_initial:
            Initial infections.
        R_initial:
            Initial recovered.
        E_initial:
            Initial exposed
        HGen_initial: int
            Initial number of General hospital admissions.
        HICU_initial: int
            Initial number of ICU cases.
        HICUVent_initial: int
            Initial number of ICU cases.
        D_initial: int
            Initial number of deaths
        n_days: int
            Number of days to simulate.
        R0: float
            Basic Reproduction number
        alpha: float
            Latent decay scale is defined as 1 / incubation period.
        gamma: float
            Infection decay scale.
        hospitalization_rate_general: float
            Fraction of infected that are hospitalized generally (not in ICU)
            TODO: Make this age dependent
        hospitalization_rate_icu: float
            Fraction of infected that are hospitalized in the ICU
            TODO: Make this age dependent
        hospitalization_length_of_stay_icu_and_ventilator: float
            Mean LOS for those requiring ventilators
        fraction_icu_requiring_ventilator: float
            Of the ICU cases, which require ventilators.
        mortality_rate: float
            Fraction of infected that die.
            TODO: Make this age dependent
        beds_general: int
            General (non-ICU) hospital beds available.
        beds_ICU: int
            ICU beds available
        ventilators: int
            Ventilators available.
        symptoms_to_hospital_days: float
            Mean number of days elapsing between infection and
            hospital admission.
        symptoms_to_mortality_days: float
            Mean number of days for an infected individual to die.
        hospitalization_length_of_stay_general: float
            Mean number of days for a hospitalized individual to be discharged.
        hospitalization_length_of_stay_icu
            Mean number of days for a ICU hospitalized individual to be
            discharged.
        """
        self.N = N
        self.suppression_policy = suppression_policy
        self.I_initial = I_initial
        self.R_initial = R_initial
        self.E_initial = E_initial
        self.D_initial = D_initial

        self.HGen_initial = HGen_initial
        self.HICU_initial = HICU_initial
        self.HICUVent_initial = HICUVent_initial

        self.S_initial = self.N - self.I_initial - self.R_initial - self.E_initial \
                         - self.D_initial - self.HGen_initial - self.HICU_initial \
                         - self.HICUVent_initial

        # Epidemiological Parameters
        self.R0 = R0        # Reproduction Number
        self.alpha = alpha  # Latent Period = 1 / incubation
        self.gamma = gamma  # Infection decay scale = 1 / t_infectious

        # These need to be made age dependent
        self.beta = self.R0 * self.gamma  # Contact number
        self.mortality_rate = mortality_rate
        self.symptoms_to_hospital_delay = symptoms_to_hospital_days
        self.symptoms_to_mortality = symptoms_to_mortality_days

        # Hospitalization Parameters
        # https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf
        # Page 16
        self.hospitalization_rate_general = hospitalization_rate_general
        self.hospitalization_rate_icu = hospitalization_rate_icu
        self.hospitalization_length_of_stay_general = hospitalization_length_of_stay_general
        self.hospitalization_length_of_stay_icu = hospitalization_length_of_stay_icu
        self.hospitalization_length_of_stay_icu_and_ventilator = hospitalization_length_of_stay_icu_and_ventilator

        # http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
        # = 0.53
        self.fraction_icu_requiring_ventilator = fraction_icu_requiring_ventilator

        # Capacity
        self.beds_general = beds_general
        self.beds_ICU = beds_ICU
        self.ventilators = ventilators

        # List of times to integrate.
        self.t_list = t_list
        self.results = None

    def _time_step(self, y, t):
        """
        One integral moment.
        """
        S, E, I, R, HNonICU, HICU, HICUVent, D = y

        # TODO: County-by-county affinity matrix terms can be used to describe
        # transmission network effects.
        #  For those living in county i, the interacting county j exposure is given
        #  by A term dE_i/dt += N_i * Sum_j [ beta_j * mix_ij * I_j * S_i + beta_i *
        #  mix_ji * I_j * S_i ] mix_ij can be proxied by Census-based commuting
        #  matrices as workplace interactions are the dominant term. See:
        #  https://www.census.gov/topics/employment/commuting/guidance/flows.html
        #
        # TODO: Age-based contact mixing affinities.
        #    It is important to track demographics themselves as they impact
        #    hospitalization and mortality rates. Additionally, exposure rates vary
        #    by age, described by matrices linked below which need to be extracted
        #    from R for the US.
        #    https://cran.r-project.org/web/packages/socialmixr/vignettes/introduction.html
        #    For an infected age PMF vector I, and a contact matrix gamma dE_i/dT =
        #    S_i (*) gamma_ij I^j / N - gamma * E_i   # Someone should double check
        #    this

        dSdt = - self.beta * self.suppression_policy(t) * S * (I + 10) / self.N  # Fraction Susceptible. 0.1 here is to simulate other infected coming into the community.
        dEdt = self.beta * self.suppression_policy(t) * S * (I + 10) / self.N - self.alpha * E  # Fraction exposed
        dIdt = self.alpha * E - self.gamma * I  # Fraction that are Infected

        # Fraction that recover
        dRdt = self.gamma * I - \
               I * ( self.mortality_rate
                    + self.hospitalization_rate_icu
                    + self.hospitalization_rate_general) \
               + HNonICU / self.hospitalization_length_of_stay_general \
               + HICU / self.hospitalization_length_of_stay_icu

        dHNonICU_dt = I * self.hospitalization_rate_general - HNonICU / self.hospitalization_length_of_stay_general
        dHICU_dt = I * self.hospitalization_rate_icu \
                   - HICU * (1 - self.fraction_icu_requiring_ventilator) / self.hospitalization_length_of_stay_icu \
                   - HICUVent * self.fraction_icu_requiring_ventilator / self.hospitalization_length_of_stay_icu_and_ventilator
        dHICUVent_dt = I * self.hospitalization_rate_icu * self.fraction_icu_requiring_ventilator - HICUVent / self.hospitalization_length_of_stay_icu_and_ventilator

        # TODO Modify this based on increased mortality if beds saturated
        # TODO Age dep mortality. Recent estimate fo relative distribution Fig 3 here:
        #      http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
        dDdt = self.mortality_rate * I  # Fraction that die.
        return dSdt, dEdt, dIdt, dRdt, dHNonICU_dt, dHICU_dt, dHICUVent_dt, dDdt

    def run(self):
        """
        Integrate the ODE numerically.

        Returns
        -------
        results: dict
        {
            't_list': self.t_list,
            'S': S,
            'E': E,
            'I': I,
            'R': R,
            'HNonICU': HNonICU,
            'HICU': HICU,
            'HICUVent': HICUVent,
            'D': D
        }
        """
        # Initial conditions vector
        y0 = self.S_initial, self.E_initial, self.R_initial, self.I_initial, \
             self.HGen_initial, self.HICU_initial, self.HICUVent_initial, self.D_initial

        # Integrate the SIR equations over the time grid, t.
        result_time_series = odeint(self._time_step, y0, self.t_list)
        S, E, I, R, HNonICU, HICU, HICUVent, D = result_time_series.T

        self.results = {
            't_list': self.t_list,
            'S': S,
            'E': E,
            'I': I,
            'R': R,
            'HNonICU': HNonICU,
            'HICU': HICU,
            'HICUVent': HICUVent,
            'D': D
        }

    def plot_results(self):
        """
        Generate a summary plot for the simulation.
        """
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        plt.figure(facecolor='w', figsize=(16, 16))
        plt.subplot(221)
        plt.plot(self.t_list, self.results['S'], alpha=1, lw=2, label='Susceptible')
        plt.plot(self.t_list, self.results['E'], alpha=.5, lw=2, label='Exposed')
        plt.plot(self.t_list, self.results['I'], alpha=.5, lw=2, label='Infected')
        plt.plot(self.t_list, self.results['R'], alpha=1, lw=2, label='Recovered & Immune', linestyle='--')
        plt.plot(self.t_list, self.results['D'], alpha=1, c='k', lw=4, label='Dead', linestyle='-')
        plt.xlabel('Time [days]', fontsize=12)
        plt.yscale('log')
        plt.ylim(1, self.N * 1.1)
        plt.grid(True, which='both', alpha=.35)
        plt.legend(framealpha=.5)
        plt.xlim(0, self.t_list.max())

        plt.subplot(222)
        plt.plot(self.t_list, self.results['HNonICU'], alpha=1, lw=2, c='steelblue', label='General Beds Required', linestyle='-')
        plt.hlines(self.beds_ICU, self.t_list[0], self.t_list[-1], 'steelblue', alpha=1, lw=2, label='ICU Bed Capacity', linestyle='--')

        plt.plot(self.t_list, self.results['HICU'], alpha=1, lw=2, c='firebrick', label='ICU Beds Required', linestyle='-')
        plt.hlines(self.beds_general, self.t_list[0], self.t_list[-1], 'firebrick', alpha=1, lw=2, label='General Bed Capacity', linestyle='--')

        plt.plot(self.t_list, self.results['HICUVent'], alpha=1, lw=2, c='seagreen', label='Ventilators Required', linestyle='-')
        plt.hlines(self.ventilators, self.t_list[0], self.t_list[-1], 'seagreen', alpha=1, lw=2, label='Ventilator Capacity', linestyle='--')

        plt.xlabel('Time [days]', fontsize=12)
        plt.ylabel('')
        plt.yscale('log')
        plt.ylim(1, self.N * 1.1)
        plt.grid(True, which='both', alpha=.35)
        plt.legend(framealpha=.5)
        plt.xlim(0, self.t_list.max())

        # Reproduction numbers
        plt.subplot(223)
        plt.plot(self.t_list, self.suppression_policy(self.t_list), c='steelblue')
        plt.ylabel('Contact Rate Reduction')
        plt.xlabel('Time [days]', fontsize=12)
