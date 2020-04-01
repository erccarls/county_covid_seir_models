import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyseir import load_data
import iminuit
import scipy
from sklearn.model_selection import ParameterGrid


class ModelFitter:
    def __init__(self,
                 model,
                 model_params,
                 compartments,
                 agg_function,
                 observations,
                 observation_t_list,
                 sample_quantile_range,
                 parameter_sample_n,
                 priors,
                 likelihood_method,
                 error_augmentation,
                 projection_start_time,
                 projection_end_time):
        """
        Parameters
        ----------
        model : SEIRModel
        model_params : dict
            Default model parameters
        compartments :
            Name of compartments to generate predictions (i.e. 'I' to fit to the observed cases)
        agg_function : callable
            Function to aggregate compartments to make it comparable to observations
        observations : np.array
            Observations.
        observation_t_list : np.array
            Observation time list


        """
        self.model = model
        self.model_params = model_params
        self.compartments = compartments
        self.agg_function = agg_function    # function to aggregate compartments to make it comparable to observations
        self.observations = observations
        self.observation_t_list = observation_t_list
        self.sample_quantile_range = sample_quantile_range
        self.parameter_sample_n = parameter_sample_n
        self.priors = priors
        self.likelihood_method = likelihood_method
        self.error_augmentation = error_augmentation

        self.fit_summary = None

        # parameters for prediction
        self.projection_start_time = projection_start_time
        self.projection_end_time = projection_end_time
        # increase in standard deviation as function of t

    def sample_from_parameter_prior(self):
        """
        Sample by quantiles.
        """

        parameter_samples = {}
        for param in self.priors:
            parameter_samples[param] = self.priors[param].ppf(np.linspace(self.sample_quantile_range[0],
                                                                          self.sample_quantile_range[1],
                                                                          self.parameter_sample_n))
            parameter_samples[param] = parameter_samples[param][parameter_samples[param]>0]
        return parameter_samples

    @staticmethod
    def log_likelihood(predictions, observations, degree_of_freedom, scenario='log_likelihood'):
        """
        Currently assume Gaussian approximation for poisson cases.
        """

        if scenario == 'chi_2':
            chi_square = (predictions - observations) ** 2 / observations
            likelihood = scipy.stats.chi_2.sf(chi_square - chi_square.min(), df=degree_of_freedom)
        elif scenario == 'log_likelihood':
            vpmf = np.vectorize(lambda x, y: scipy.stats.poisson(mu=x).pmf(round(y)))
            likelihood = vpmf(observations, predictions)
        return np.log(sum(likelihood))

    def run_model(self, model_params, parameter_sample, prediction_t_list):
        model_params.update(parameter_sample)
        model = self.model(**model_params)
        model.run()
        predictions = None
        for c in self.compartments:
            if predictions is None:
                predictions = model.results[c]
            predictions += model.results[c]
        predictions = self.agg_function(predictions)
        f = scipy.interpolate.interp1d(model.t_list, predictions)
        predictions = f(prediction_t_list)
        return model, predictions


    def fit(self):
        """
        Generate posterior estimates of the model parameters.
        """
        parameter_samples = self.sample_from_parameter_prior()
        param_grid = ParameterGrid(parameter_samples)
        likelihoods = list()
        results = dict()
        for p in self.priors:
            results[p] = list()
            results[p + '_prior'] = list()
        for param in param_grid:
            for p in param:
                results[p].append(param[p])
                results[p + '_prior'].append(self.priors[p].pdf(param[p]))
            model, predictions = self.run_model(self.model_params, param, self.observation_t_list)
            likelihood = self.log_likelihood(predictions, self.observations,
                                             degree_of_freedom=len(self.observations) - 1,
                                             scenario=self.likelihood_method)
            likelihoods.append(likelihood)
        results['likelihood'] = likelihoods
        results = pd.DataFrame(results)
        parameter_posteriors = self.parameter_posterior_likelihood_profile(results)

        self.fit_summary = {'parameter_posteriors': parameter_posteriors,
                            'likelihoods_and_priors': results}

        return self.fit_summary


    def parameter_posterior_likelihood_profile(self, results):
        """
        Calculate posterior likelihood profile for each parameter.
        """
        param_posteriors = {}
        for param in self.priors:
            # marginalize through all other parameter priors SUM(P(y | p1, p2, p3) * P(p2) * P(p3)) over p2 and p2 for
            # p1.
            cols_to_marginalize = [col for col in results.columns if '_prior' in col and 'sigma' not in col]
            results['marginal_likelihood_%s' % param] = results[cols_to_marginalize + ['likelihood']].product(axis=1)
            marginal_likelihoods = results.groupby(param)['marginal_likelihood_%s' % param].sum().reset_index()
            posterior_likelihoods = marginal_likelihoods.merge(results[[param, param + '_prior']],
                                                               on=param).drop_duplicates()

            # posterior probability P(p | y) = P(y | p) * P(p) / P(y), P(y) is omitted because it is a constant.
            posterior_likelihoods['posterior'] = \
                posterior_likelihoods[param + '_prior'] * posterior_likelihoods['marginal_likelihood_%s' % param]
            param_posteriors[param] = posterior_likelihoods[[param, 'posterior']]

        return param_posteriors

    def plot_parameter_pdf(self):
        for param in self.priors:
            steps = self.priors[param].ppf(np.linspace(0.01, 0.99, 1000))
            pdf = self.priors[param].pdf(steps)
            ax, fig = plt.subplots()
            axes = ax.get_axes()
            axes[0].plot(steps, pdf, label='prior pdf', color='r')
            axes[0].legend()

            ax2 = axes[0].twinx()
            self.fit_summary['parameter_posteriors'][param].plot(param, 'posterior', kind='line',
                                             ax=ax2, color='g', label='posterior pdf')
            ax2.legend()

            plt.title(param)
            plt.xlabel(param)
            plt.ylabel('likelihood profile')


    def _error_scale(self, t_list, projection_start_time):
        """
        Estimate increment in error rate as function of forward time steps (drift forecasts)
        """
        # time steps
        hs = np.maximum(t_list, projection_start_time) - projection_start_time
        T = t_list.max()
        error_scale = (hs * (1 + hs/T) + 1) ** (1/2)

        return error_scale

    def calculate_stats(self, values, likelihoods):
        sorted_idx = np.argsort(values)
        values = values[sorted_idx]
        likelihoods = likelihoods[sorted_idx]
        cdf = np.cumsum(likelihoods)
        cdf /= cdf.max()
        f = scipy.interpolate.interp1d(cdf, values)

        stat = {}
        stat['ci95_lower'] = f(0.025)
        stat['ci95_upper'] = f(0.975)
        stat['ci68_lower'] = f(0.16)
        stat['ci68_upper'] = f(0.84)
        stat['median'] = f(0.5)
        stat['mean'] = (values * likelihoods).sum() / likelihoods.sum()

        return stat

    def rescale_ci(self, stat):
        # rescale based on forecasting error scale
        stat['ci95_lower_rescale'] = stat['median'] - (stat['median'] - stat['ci95_lower']) * stat['error_scale']
        stat['ci95_upper_rescale'] = stat['median'] + (stat['ci95_upper'] - stat['median']) * stat['error_scale']
        stat['ci68_lower_rescale'] = stat['median'] - (stat['median'] - stat['ci68_lower']) * stat['error_scale']
        stat['ci68_upper_rescale'] = stat['median'] + (stat['ci68_upper'] - stat['median']) * stat['error_scale']

        return stat


    def projection_stats(self, projections, error_scale):

        projection_stats = []
        for col in projections:
            if 'pred' in col:
                t = int(col.split('_')[1])
                projection_stat = self.calculate_stats(projections[col], projections['likelihood'])
                projection_stat['time'] = t
                projection_stats.append(projection_stat)

        projection_stats = pd.DataFrame(projection_stats)
        projection_stats['error_scale'] = error_scale
        projection_stats = self.rescale_ci(projection_stats)
        return projection_stats

    def run_projection(self):
        self.model_params.update({'t_list': range(0, self.projection_end_time)})
        projections = {}

        for p in self.priors:
            projections[p] = []

        for t in self.model_params['t_list']:
            projections['pred_%d' % t] = list()
        parameter_samples = self.sample_from_parameter_prior()
        param_grid = ParameterGrid(parameter_samples)
        for param in param_grid:
            for p in param:
                projections[p].append(param[p])
                _, predictions = self.run_model(self.model_params,
                                              param,
                                              list(range(0, self.projection_end_time)))

            for t in self.model_params['t_list']:
                projections['pred_%d' % t].append(predictions[list(self.model_params['t_list']).index(t)])

        projections = pd.DataFrame(projections)
        for param in self.priors.keys():
            projections = projections.merge(
                self.fit_summary['parameter_posteriors'][param].rename(columns={'posterior': param + '_posterior'}),
                on=param)

        # combine parameter posterior probability
        projections['likelihood'] = \
            projections[[col for col in projections.columns if 'posterior' in col]].product(axis=1)

        if self.error_augmentation:
            error_scale = self._error_scale(np.array(self.model_params['t_list']), self.projection_start_time)
        else:
            error_scale = {t: 1 for t in self.model_params['t_list']}

        projection_stats = self.projection_stats(projections, error_scale)

        return projection_stats

    def plot_projection(self, projection_stats, forecasting_rescale=True):
        if forecasting_rescale:
            suffix = '_rescale'
        else:
            suffix = ''
        plt.plot(projection_stats['time'], projection_stats['mean'], linewidth=2, label='mean')  # mean curve.
        plt.plot(projection_stats['time'], projection_stats['median'], linewidth=2, label='median')
        # median curve.
        '''plt.fill_between(projection_stats['time'],
                         projection_stats['ci95_lower%s' % suffix],
                         projection_stats['ci95_upper%s' % suffix],
                         facecolor='b', alpha=0.05, label='95% CI')
        plt.fill_between(projection_stats['time'],
                         projection_stats['ci68_lower%s' % suffix],
                         projection_stats['ci68_upper%s' % suffix],
                         facecolor='b', alpha=.1, label='68% CI')'''
        plt.plot(projection_stats['time'], projection_stats['ci95_lower%s' % suffix], color='r',
                 linestyle='--')
        plt.plot(projection_stats['time'], projection_stats['ci95_upper%s' % suffix], color='r',
                 linestyle='--', label = '95% CI')
        plt.scatter(self.observation_t_list, self.observations, label='observations')
        plt.yscale('log')
        plt.xlabel('time (days)')
        plt.legend()

