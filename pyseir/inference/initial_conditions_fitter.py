from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import numpy as np
from pyseir import load_data
import iminuit


class InitialConditionsFitter:

    def __init__(self, fips, t0_case_count=5, start_days_before_t0=5,
                 start_days_after_t0=1000):
        """
        Fit an exponential model to observations assuming a binomial error on
        observations. Identify t0 at the threshold specified.

        # TODO: Can we incorporate the death rate into the fit to also infer
        #       actual cases?

        Parameters
        ----------
        fips: str
            County fips code
        t0_case_count: int
            Case count to infer the start date of.
        start_days_before_t0: int
            After we find the time of t0_case_count, filter observations
            occurring more than this many days before.
        start_days_after_t0: int
            After we find the time of t0_case_count, filter observations
            occurring more than this many days after.
        """
        self.t0_case_count = t0_case_count
        self.start_days_before_t0 = start_days_before_t0
        self.start_days_after_t0 = start_days_after_t0



        # Load case data
        case_data = load_data.load_county_case_data()
        self.cases = case_data[case_data['fips'] == fips]
        self.t = (self.cases.date - self.cases.date.min()).dt.days.values
        self.data_start_date = self.cases.date.min()

        self.y = self.cases.cases.values

        self.fit_predictions = None
        self.t0 = None
        self.t0_date = None
        self.model_params = None

    @staticmethod
    def exponential_model(norm, t0, scale, t):
        """
        Simple exponential model.

        Parameters
        ----------
        norm
        t0
        scale
        t

        Returns
        -------

        """
        return np.exp(norm * np.exp((t - t0) / scale))

    @staticmethod
    def reduced_chi2(y_pred, y):
        """
        Calculate reduced chi^2.

        Parameters
        ----------
        y_pred
        y

        Returns
        -------
        reduced_chi2: float
        """
        chi2 = (y_pred[y > 0] - y[y > 0]) ** 2 / y[y > 0]
        return np.sum(chi2) / (len(chi2) - 1)

    def exponential_loss(self, norm, t0, scale):
        """Return the reduced chi2 for an exponential fit to teh data"""
        y_pred = self.exponential_model(norm, t0, scale, self.t)
        return self.reduced_chi2(y_pred, self.y)

    def fit_county_initial_conditions(self, t, y):
        x0 = dict(norm=1, t0=5, scale=20, error_norm=.01, error_t0=.1, error_scale=.01)
        m = iminuit.Minuit(self.exponential_loss, **x0, errordef=0.5)
        fit = m.migrad()
        return {val['name']: val['value'] for val in fit.params}

    def fit(self):
        model_params = self.fit_county_initial_conditions(self.t, self.y)
        fit_predictions = self.exponential_model(**model_params, t=self.t)

        # Filter out data a few days before this and re-fit.
        t0_idx = np.argmin(np.abs(fit_predictions - self.t0_case_count))

        filter_start = max(0, t0_idx - self.start_days_before_t0)
        filter_end = min(len(self.t), t0_idx + self.start_days_after_t0)
        t_filtered = self.t[filter_start: filter_end]
        y_filtered = self.y[filter_start: filter_end]

        self.model_params = self.fit_county_initial_conditions(t_filtered, y_filtered)
        self.fit_predictions = self.exponential_model(**self.model_params, t=self.t)
        self.t0_idx = np.argmin(np.abs(self.fit_predictions - self.t0_case_count))
        self.t0 = self.t[self.t0_idx]
        self.t0_date = self.data_start_date + timedelta(days=int(self.t0))

    def plot_fit(self):
        plt.figure(figsize=(10, 7))
        plt.errorbar(self.t - self.t0, self.cases.cases, yerr=np.sqrt(self.cases.cases), marker='o', label='Cases')
        plt.plot(self.t - self.t0, self.fit_predictions, label='Best Fit with Filters')
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.ylabel('Count')
        plt.xlabel(f'Time Since {self.t0_case_count} cases predicted')
        plt.legend()


if __name__ == '__main__':

    fitter = InitialConditionsFitter(
        fips='06075',  # SF County
        t0_case_count=5,
        start_days_before_t0=5,
        start_days_after_t0=1000
    )
    fitter.fit()
    fitter.plot_fit()
