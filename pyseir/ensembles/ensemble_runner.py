import datetime
import logging
import os
import inspect
import numpy as np
import json
import copy
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.models.suppression_policies import generate_empirical_distancing_policy
from pyseir.reports.pdf_report import PDFReport
from pyseir import OUTPUT_DIR
from pyseir import load_data
from pyseir.inference import fit_results
from pyseir.reports.names import compartment_to_name_map


class EnsembleRunner:
    """
    The EnsembleRunner executes a collection of N_samples simulations based on
    priors defined in the ParameterEnsembleGenerator.

    Parameters
    ----------
    fips: str
        County fips code
    n_years: int
        Number of years to simulate
    n_samples: int
        Ensemble size to run for each suppression policy.
    suppression_policy: list(float or str)
        List of suppression policies to apply.
    output_percentiles: list
        List of output percentiles desired. These will be computed for each
        compartment.
    """
    def __init__(self, fips, n_years=2, n_samples=250,
                 suppression_policy=(0.35, 0.5, 0.75, 1),
                 skip_plots=False,
                 output_percentiles=(5, 25, 32, 50, 75, 68, 95)):

        self.fips = fips
        self.t_list = np.linspace(0, 365 * n_years, 365 * n_years)
        self.skip_plots = skip_plots

        self.county_metadata = load_data.load_county_metadata_by_fips(fips)
        self.output_percentiles = output_percentiles
        self.n_samples = n_samples
        self.n_years = n_years
        self.t0 = fit_results.load_t0(fips)
        self.date_generated = datetime.datetime.utcnow().isoformat()
        self.suppression_policy = suppression_policy

        self.summary = copy.deepcopy(self.__dict__)
        self.summary.pop('t_list')

        _county_case_data = load_data.load_county_case_data()
        self.county_case_data = _county_case_data[_county_case_data['fips'] == fips]

        self.all_outputs = {}
        self.output_file_report = os.path.join(OUTPUT_DIR, self.county_metadata['state'], 'reports',
            f"{self.county_metadata['state']}__{self.county_metadata['county']}__{self.fips}__ensemble_projections.pdf")
        self.output_file_data = os.path.join( OUTPUT_DIR, self.county_metadata['state'], 'data',
            f"{self.county_metadata['state']}__{self.county_metadata['county']}__{self.fips}__ensemble_projections.json")

        self.report = PDFReport(filename=self.output_file_report)

        self.report.write_text_page(self.summary,
                                    title=f'PySEIR COVID19 Estimates\n{self.county_metadata["county"]} County, {self.county_metadata["state"]}',
                                    page_heading=f'Generated {self.summary["date_generated"]}', body_fontsize=6, title_fontsize=12)

        self.report.write_text_page(inspect.getsource(ParameterEnsembleGenerator.sample_seir_parameters),
                                    title='PySEIR Model Ensemble Parameters')

    @staticmethod
    def _run_single_simulation(parameter_set):
        """
        Run a single simulation instance.

        Parameters
        ----------
        parameter_set: dict
            Params passed to the SEIR model

        Returns
        -------
        model: SEIRModel
            Executed model.
        """
        model = SEIRModel(**parameter_set)
        model.run()
        return model

    def run_ensemble(self):
        """
        Run an ensemble of models for each suppression policy nad generate the
        output report / results dataset.
        """
        for suppression_policy in self.summary['suppression_policy']:
            logging.info(f'Generating For Policy {suppression_policy}')

            parameter_ensemble = ParameterEnsembleGenerator(
                fips=self.fips,
                N_samples=self.n_samples,
                t_list=self.t_list,
                suppression_policy=generate_empirical_distancing_policy(
                    t_list=self.t_list,
                    fips=self.fips,
                    future_suppression=suppression_policy
                )).sample_seir_parameters()

            model_ensemble = list(map(self._run_single_simulation, parameter_ensemble))

            logging.info(f'Generating Report for suppression policy {suppression_policy}')
            self.generate_output(model_ensemble, suppression_policy)

        self.report.close()

        with open(self.output_file_data, 'w') as f:
            json.dump(self.all_outputs, f)

    def generate_output(self, model_ensemble, suppression_policy):
        """
        Generate a county level report.

        Parameters
        ----------
        model_ensemble: list(SEIRModel)
        suppression_policy: float()

        Returns
        -------

        """
        compartments = {key: [] for key in model_ensemble[0].results.keys() if key not in ('t_list')}

        for model in model_ensemble:
            for key in compartments:
                compartments[key].append(model.results[key])

        outputs = defaultdict(dict)
        outputs['t_list'] = model_ensemble[0].t_list.tolist()

        # ------------------------------------------
        # Calculate Confidence Intervals and Peaks
        # ------------------------------------------
        for compartment, value_stack in compartments.items():
            value_stack = np.vstack(value_stack)

            # Compute percentiles over the ensemble
            for percentile in self.output_percentiles:
                outputs[compartment]['ci_%i' % percentile] = np.percentile(value_stack, percentile, axis=0).tolist()

            # When is surge capacity reached?
            capacity_attr = {
                'HGen': 'beds_general',
                'HICU': 'beds_ICU',
                'HVent': 'ventilators'
            }
            if compartment in capacity_attr:
                outputs[compartment]['surge_start'] = []
                outputs[compartment]['surge_end'] = []
                for m in model_ensemble:
                    # Find the first t where overcapacity occurs

                    surge_start_idx = np.argwhere(m.results[compartment] > getattr(m, capacity_attr[compartment]))
                    outputs[compartment]['surge_start'].append(
                        outputs['t_list'][surge_start_idx[0][0]] if len(surge_start_idx) > 0 else float('NaN'))

                    # Reverse the t-list and capacity and do the same.
                    surge_end_idx = np.argwhere(m.results[compartment][::-1] > getattr(m, capacity_attr[compartment]))
                    outputs[compartment]['surge_end'].append(
                        outputs['t_list'][::-1][surge_end_idx[0][0]] if len(surge_end_idx) > 0 else float('NaN'))

            # Compute the peak times for each compartment by finding the arg
            # max, and selecting the corresponding time.
            peak_indices = value_stack.argmax(axis=1)

            outputs[compartment]['peak_times'] = [outputs['t_list'][peak_index] for peak_index in peak_indices]
            values_at_peak_index = [val[idx] for val, idx in zip(value_stack, peak_indices)]
            outputs[compartment]['peak_values'] = values_at_peak_index
            for percentile in self.output_percentiles:
                outputs[compartment]['peak_value_ci%i' % percentile] = np.percentile(values_at_peak_index, percentile).tolist()
                outputs[compartment]['peak_time_ci%i' % percentile] = np.percentile(outputs[compartment]['peak_times'], percentile).tolist()
            outputs[compartment]['peak_value_mean'] = np.mean(values_at_peak_index).tolist()

        outputs['HICU']['capacity'] = [m.beds_ICU for m in model_ensemble]
        outputs['HVent']['capacity'] = [m.ventilators for m in model_ensemble]
        outputs['HGen']['capacity'] = [m.beds_general for m in model_ensemble]

        self.all_outputs[f'suppression_policy__{suppression_policy}'] = outputs

        # TODO: Refactor... this plotting is ugly.
        if self.skip_plots:
            return

        # Add a sample model from the ensemble.
        fig = model_ensemble[0].plot_results(xlim=(0, 360))
        fig.suptitle(
            f'PySEIR COVID19 Estimates: {self.county_metadata["county"]} County, {self.county_metadata["state"]}. '
            f'SAMPLE OF MODEL ENSEMBLE', fontsize=16)
        self.report.add_figure(fig)

        # -----------------------------------
        # Plot each compartment distribution
        # -----------------------------------
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle(f'PySEIR COVID19 Estimates: {self.county_metadata["county"]} County, {self.county_metadata["state"]}. '
                     f'\nSupression Policy={suppression_policy} (1=No Suppression)' , fontsize=16)
        for i_plot, compartment in enumerate(compartments):
            plt.subplot(5, 5, i_plot + 1)
            plt.plot(outputs['t_list'], outputs[compartment]['ci_50'], color='steelblue',
                     linewidth=3, label=compartment_to_name_map[compartment])
            plt.fill_between(outputs['t_list'], outputs[compartment]['ci_32'], outputs[compartment]['ci_68'], alpha=.3, color='steelblue')
            plt.fill_between(outputs['t_list'], outputs[compartment]['ci_5'], outputs[compartment]['ci_95'], alpha=.3, color='steelblue')
            plt.yscale('log')
            plt.ylim(1e0)
            plt.xlim(0, 360)
            plt.grid(True, which='both', alpha=0.3)

            plt.xlabel('Days Since Case 0')
            if compartment == 'HICU':
                percentiles = np.percentile([m.beds_ICU for m in model_ensemble], (5, 32, 50, 68, 95))
                plt.hlines(percentiles[2], *plt.xlim(), label='ICU Capacity', color='darkseagreen')
                plt.hlines([percentiles[0], percentiles[4]], *plt.xlim(), color='darkseagreen', linestyles='-.', alpha=.4)
                plt.hlines([percentiles[1], percentiles[3]], *plt.xlim(), color='darkseagreen', linestyles='--', alpha=.2)
            elif compartment == 'HGen':
                percentiles = np.percentile([m.beds_general for m in model_ensemble], (5, 32, 50, 68, 95))
                plt.hlines(percentiles[2], *plt.xlim(), label='Bed Capacity', color='darkseagreen')
                plt.hlines([percentiles[0], percentiles[4]], *plt.xlim(), color='darkseagreen', linestyles='-.', alpha=.4)
                plt.hlines([percentiles[1], percentiles[3]], *plt.xlim(), color='darkseagreen', linestyles='--', alpha=.2)
            elif compartment == 'HVent':
                percentiles = np.percentile([m.ventilators for m in model_ensemble], (5, 32, 50, 68, 95))
                plt.hlines(percentiles[2], *plt.xlim(), label='Ventilator Capacity', color='darkseagreen')
                plt.hlines([percentiles[0], percentiles[4]], *plt.xlim(), color='darkseagreen', linestyles='-.', alpha=.4)
                plt.hlines([percentiles[1], percentiles[3]], *plt.xlim(), color='darkseagreen', linestyles='--', alpha=.2)

            # Plot data
            if compartment in ['D', 'total_deaths'] and len(self.county_case_data) > 0:
                plt.errorbar((self.county_case_data.date - self.summary['t0']).dt.days,
                         self.county_case_data.deaths, yerr=np.sqrt(self.county_case_data.deaths),
                         linestyle='-', label='Deaths Observed', marker='o', markersize=4)
            if compartment in ['I'] and len(self.county_case_data) > 0:
                plt.errorbar((self.county_case_data.date - self.summary['t0']).dt.days,
                         self.county_case_data.cases, yerr=np.sqrt(self.county_case_data.cases  ), linestyle='-',
                         label='Cases Observed', marker='o', markersize=4, color='firebrick')

            plt.legend()
            self._plot_dates(log=False)

        # -----------------------------
        # Plot peak Timing
        # -----------------------------
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + list('bgrcmyk')

        marker_cycle = ['o', 's', '+', 'd', 'o'] * 4


        plt.subplot(5, 5, len(compartments) + 1)

        for i, compartment in enumerate(['E', 'A', 'I', 'HGen', 'HICU', 'HVent', 'general_admissions_per_day',
                                         'icu_admissions_per_day', 'direct_deaths_per_day', 'total_deaths_per_day']):
            median = outputs[compartment]['peak_time_ci50']
            ci5, ci95 = outputs[compartment]['peak_time_ci5'], outputs[compartment]['peak_time_ci95']
            ci32, ci68 = outputs[compartment]['peak_time_ci32'], outputs[compartment]['peak_time_ci68']
            plt.scatter(median, i, label=compartment_to_name_map[compartment], c=color_cycle[i], marker=marker_cycle[i])
            plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i],)
            plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
        self._plot_dates(log=False)
        plt.legend(loc=(1.05, 0.0))
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('Peak Time After $t_0(C=5)$ [Days]')
        plt.yticks([])

        # -----------------------------
        # Plot peak capacity
        # -----------------------------
        plt.subplot(5, 5, len(compartments) + 3)
        for i, compartment in enumerate(['E', 'A', 'I', 'R', 'D', 'total_deaths',
                                         'direct_deaths_per_day', 'total_deaths_per_day', 'HGen', 'HICU', 'HVent',
                                         'HGen_cumulative', 'HICU_cumulative', 'HVent_cumulative',
                                         'general_admissions_per_day', 'icu_admissions_per_day']):
            median = outputs[compartment]['peak_value_ci50']
            ci5, ci95 = outputs[compartment]['peak_value_ci5'], outputs[compartment]['peak_value_ci95']
            ci32, ci68 = outputs[compartment]['peak_value_ci32'], outputs[compartment]['peak_value_ci68']
            plt.scatter(median, i, label=compartment_to_name_map[compartment], c=color_cycle[i], marker=marker_cycle[i])
            plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i])
            plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
            plt.xscale('log')

        plt.vlines(self.county_metadata['total_population'], *plt.ylim(), label='Entire Population', alpha=0.5, color='g')
        plt.vlines(self.county_metadata['total_population'] * 0.65, *plt.ylim(), label='Approx. Herd Immunity',
                   alpha=0.5, color='purple', linestyles='--', linewidths=2)
        plt.legend(loc=(1, -0.1))
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('Value at Peak')
        plt.yticks([])

        self.report.add_figure(fig)

    def _plot_dates(self, log=True):
        """
        Helper function to add date plots.

        Parameters
        ----------
        log: bool
            If True, shift y-positioning of labels based on a log scale.
        """
        low_limit = plt.ylim()[0]
        if log:
            upp_limit = 1 * np.log(plt.ylim()[1])
        else:
            upp_limit = 1 * plt.ylim()[1]

        for month in range(4, 11):
            dt = datetime.datetime(day=1, month=month, year=2020)
            offset = (dt - self.summary['t0']).days
            plt.vlines(offset, low_limit, upp_limit, color='firebrick', alpha=.4, linestyles=':')
            plt.text(offset, low_limit*1.3, dt.strftime('%B'), rotation=90, color='firebrick', alpha=0.6)



def _run_county(fips, ensemble_kwargs):
    """
    Execute the ensemble runner for a specific county.

    Parameters
    ----------
    fips: str
        County fips.
    ensemble_kwargs: dict
        Kwargs passed to the EnsembleRunner object.
    """
    runner = EnsembleRunner(fips=fips, **ensemble_kwargs)
    runner.run_ensemble()


def run_state(state, ensemble_kwargs):
    """
    Run the EnsembleRunner for each county in a state.

    Parameters
    ----------
    state: str
        State to run against.
    ensemble_kwargs: dict
        Kwargs passed to the EnsembleRunner object.
    """
    df = load_data.load_county_metadata()
    all_fips = df[df['state'].str.lower() == state.lower()].fips
    p = Pool()
    f = partial(_run_county, ensemble_kwargs=ensemble_kwargs)
    p.map(f, all_fips)
    p.close()
