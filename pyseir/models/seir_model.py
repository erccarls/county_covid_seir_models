import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SEIRModel:

    def __init__(self,
                 N,
                 t_list,
                 suppression_policy,
                 A_initial=50,
                 I_initial=50,
                 R_initial=0,
                 E_initial=0,
                 HGen_initial=0,
                 HICU_initial=0,
                 HICUVent_initial=0,
                 D_initial=0,
                 R0=2.4,
                 sigma=1 / 5.2,
                 kappa=1,
                 delta_A=1 / 14,
                 gamma=0.5,
                 hospitalization_rate_general=0.11,
                 hospitalization_rate_icu=0.04,
                 mortality_rate=0.0052,
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
         - https://arxiv.org/pdf/2003.10047.pdf  # We mostly follow this notation.
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
        A_initial: int
            Initial asymptomatic
        I_initial: int
            Initial infections.
        R_initial: int
            Initial recovered.
        E_initial: int
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
        kappa: float
            Fractional contact rate for those with symptoms since they should be
            isolated vs asymptomatic who are less isolated. A value 1 implies
            the same rate. A value 0 implies symptomatic people never infect
            others.
        sigma: float
            Latent decay scale is defined as 1 / incubation period.
            1 / 4.8: https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Global-Impact-26-03-2020.pdf
            1 / 5.2 [3, 8]: https://arxiv.org/pdf/2003.10047.pdf
        delta_A: float
            Inverse infectious period (asymptomatic and symptomatic)
            1/14: https://arxiv.org/pdf/2003.10047.pdf
        gamma: float
            Clinical outbreak rate (fraction of infected that show symptoms)
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
            0.0052: https://arxiv.org/abs/2003.10720
            TODO: Make this age dependent
            TODO: This is modeled below as P(mortality | symptoms) which is higher than the overall mortality rate by factor 2.
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
            Hospitalization to death Needs to be added to time to
                15.16 [0, 42] - https://arxiv.org/pdf/2003.10047.pdf
        hospitalization_length_of_stay_general: float
            Mean number of days for a hospitalized individual to be discharged.
        hospitalization_length_of_stay_icu
            Mean number of days for a ICU hospitalized individual to be
            discharged.
        """
        self.N = N
        self.suppression_policy = suppression_policy
        self.I_initial = I_initial
        self.A_initial = A_initial
        self.R_initial = R_initial
        self.E_initial = E_initial
        self.D_initial = D_initial

        self.HGen_initial = HGen_initial
        self.HICU_initial = HICU_initial
        self.HICUVent_initial = HICUVent_initial

        self.S_initial = self.N - self.A_initial - self.I_initial - self.R_initial - self.E_initial \
                         - self.D_initial - self.HGen_initial - self.HICU_initial \
                         - self.HICUVent_initial

        # Epidemiological Parameters
        self.R0 = R0              # Reproduction Number
        self.sigma = sigma        # Latent Period = 1 / incubation
        self.gamma = gamma        # Clinical outbreak rate
        self.delta_A = delta_A    # Infectious period
        self.kappa = kappa        # Discount fraction due to isolation of symptomatic cases.

        # These need to be made age dependent
        # R0 =  beta = Contact rate * incubation period
        self.beta = self.R0 * self.sigma

        self.mortality_rate = mortality_rate
        self.symptoms_to_hospital_days = symptoms_to_hospital_days
        self.symptoms_to_mortality_days = symptoms_to_mortality_days

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
        S, E, A, I, R, HNonICU, HICU, HICUVent, D = y

        # TODO: County-by-county affinity matrix terms can be used to describe
        # transmission network effects. ( also known as Multi-Region SEIR)
        # https://arxiv.org/pdf/2003.09875.pdf
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

        # Effective contact rate * those that get exposed * those susceptible.
        number_exposed = self.beta * self.suppression_policy(t) * S * (self.kappa * I + A) / self.N
        dSdt = - number_exposed

        exposed_and_symptomatic = self.gamma * self.sigma * E           # latent period moving to infection = 1 / incubation
        exposed_and_asymptomatic = (1 - self.gamma) * self.sigma * E    # latent period moving to asymptomatic but infected) = 1 / incubation
        dEdt = number_exposed - exposed_and_symptomatic - exposed_and_asymptomatic

        asymptomatic_and_recovered = self.delta_A * A
        dAdt = exposed_and_asymptomatic - asymptomatic_and_recovered

        # Fraction that didn't die or go to hospital
        infected_and_recovered_no_hospital = self.delta_A * I * (1 - self.mortality_rate - self.hospitalization_rate_general - self.hospitalization_rate_icu)
        infected_and_in_hospital_general = I * self.hospitalization_rate_general / self.symptoms_to_hospital_days
        infected_and_in_hospital_icu = I * self.hospitalization_rate_icu / self.symptoms_to_hospital_days
        infected_and_dead = I * self.mortality_rate / self.symptoms_to_mortality_days

        dIdt = exposed_and_symptomatic - infected_and_recovered_no_hospital - infected_and_in_hospital_general - infected_and_in_hospital_icu - infected_and_dead

        recovered_after_hospital_general = HNonICU / self.hospitalization_length_of_stay_general
        recovered_after_hospital_icu = HICU * ((1 - self.fraction_icu_requiring_ventilator)/ self.hospitalization_length_of_stay_icu
                                               + self.fraction_icu_requiring_ventilator / self.hospitalization_length_of_stay_icu_and_ventilator)

        dHNonICU_dt = infected_and_in_hospital_general - recovered_after_hospital_general
        dHICU_dt = infected_and_in_hospital_icu - recovered_after_hospital_icu

        # This compartment is for tracking ventillator count. The beds are accounted for in the ICU cases.
        dHICUVent_dt = infected_and_in_hospital_icu * self.fraction_icu_requiring_ventilator \
                       - HICUVent / self.hospitalization_length_of_stay_icu_and_ventilator

        # Fraction that recover
        dRdt = (asymptomatic_and_recovered
                + infected_and_recovered_no_hospital
                + recovered_after_hospital_general
                + recovered_after_hospital_icu)

        # TODO Modify this based on increased mortality if beds saturated
        # TODO Age dep mortality. Recent estimate fo relative distribution Fig 3 here:
        #      http://www.healthdata.org/sites/default/files/files/research_articles/2020/covid_paper_MEDRXIV-2020-043752v1-Murray.pdf
        dDdt = infected_and_dead  # Fraction that die.
        return dSdt, dEdt, dAdt, dIdt, dRdt, dHNonICU_dt, dHICU_dt, dHICUVent_dt, dDdt

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
        y0 = self.S_initial, self.E_initial, self.A_initial, self.I_initial, self.R_initial,\
             self.HGen_initial, self.HICU_initial, self.HICUVent_initial, self.D_initial

        # Integrate the SIR equations over the time grid, t.
        result_time_series = odeint(self._time_step, y0, self.t_list)
        S, E, A, I, R, HGen, HICU, HICUVent, D = result_time_series.T

        self.results = {
            't_list': self.t_list,
            'S': S,
            'E': E,
            'A': A,
            'I': I,
            'R': R,
            'HGen': HGen,
            'HICU': HICU,
            'HVent': HICUVent,
            'D': D
        }

    def plot_results(self, y_scale='log'):
        """
        Generate a summary plot for the simulation.

        Parameters
        ----------
        y_scale: str
            Matplotlib scale to use on y-axis. Typically 'log' or 'linear'
        """
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        plt.figure(facecolor='w', figsize=(20, 6))
        plt.subplot(131)
        plt.plot(self.t_list, self.results['S'], alpha=1, lw=2, label='Susceptible')
        plt.plot(self.t_list, self.results['E'], alpha=.5, lw=2, label='Exposed')
        plt.plot(self.t_list, self.results['A'], alpha=.5, lw=2, label='Asymptomatic')
        plt.plot(self.t_list, self.results['I'], alpha=.5, lw=2, label='Infected')
        plt.plot(self.t_list, self.results['R'], alpha=1, lw=2, label='Recovered & Immune', linestyle='--')

        # This is debugging and should be constant.
        # TODO: we must be missing a small conservation term above.
        plt.plot(self.t_list,  self.results['S'] + self.results['E'] + self.results['A']
                             + self.results['I'] + self.results['R'] + self.results['D']
                             + self.results['HGen'] + self.results['HICU'], label='Total')

        plt.xlabel('Time [days]', fontsize=12)
        plt.yscale(y_scale)
        # plt.ylim(1, plt.ylim(1))
        plt.grid(True, which='both', alpha=.35)
        plt.legend(framealpha=.5)
        plt.xlim(0, self.t_list.max())
        plt.ylim(1, self.N * 1.1)

        plt.subplot(132)
        plt.plot(self.t_list, self.results['HGen'], alpha=1, lw=2, c='steelblue', label='General Beds Required', linestyle='-')
        plt.hlines(self.beds_ICU, self.t_list[0], self.t_list[-1], 'steelblue', alpha=1, lw=2, label='ICU Bed Capacity', linestyle='--')

        plt.plot(self.t_list, self.results['HICU'], alpha=1, lw=2, c='firebrick', label='ICU Beds Required', linestyle='-')
        plt.hlines(self.beds_general, self.t_list[0], self.t_list[-1], 'firebrick', alpha=1, lw=2, label='General Bed Capacity', linestyle='--')

        plt.plot(self.t_list, self.results['HVent'], alpha=1, lw=2, c='seagreen', label='Ventilators Required', linestyle='-')
        plt.hlines(self.ventilators, self.t_list[0], self.t_list[-1], 'seagreen', alpha=1, lw=2, label='Ventilator Capacity', linestyle='--')

        plt.plot(self.t_list, self.results['D'], alpha=1, c='k', lw=4, label='Dead', linestyle='-')

        plt.xlabel('Time [days]', fontsize=12)
        plt.ylabel('')
        plt.yscale(y_scale)
        plt.ylim(1, plt.ylim()[1])
        plt.grid(True, which='both', alpha=.35)
        plt.legend(framealpha=.5)
        plt.xlim(0, self.t_list.max())

        # Reproduction numbers
        plt.subplot(133)
        plt.plot(self.t_list, [self.suppression_policy(t) for t in self.t_list], c='steelblue')
        plt.ylabel('Contact Rate Reduction')
        plt.xlabel('Time [days]', fontsize=12)
