import datetime
import os
import inspect
import numpy as np
import json
from collections import defaultdict
from functools import partial
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.models.suppression_policies import generate_triggered_suppression_model
from pyseir.reports.pdf_report_base import PDFReportBase
from pyseir import OUTPUT_DIR
from pyseir.load_data import load_county_metadata
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
    N_samples: int
        Ensemble size to run for each suppression policy.
    suppression_policy: list(float or str)
        List of suppression policies to apply.
    """
    output_percentiles = [5, 32, 50, 68, 95]

    def __init__(self, fips, n_years=3, N_samples=250,
                 suppression_policy=(0.35, 0.5, 0.70), skip_plots=False):
        self.t_list = np.linspace(0, 365 * n_years, 365 * n_years)
        self.skip_plots = skip_plots
        county_metadata = load_county_metadata().set_index('fips').loc[fips].to_dict()

        self.summary = {
            'date_generated': datetime.datetime.utcnow().isoformat(),
            'suppression_policy': suppression_policy,
            'fips': fips,
            'N_samples': N_samples,
            'n_years': n_years,
            't0': fit_results.get_t0(fips),
            'population': county_metadata['total_population'],
            'population_density': county_metadata['population_density'],
            'state': county_metadata['state'],
            'county': county_metadata['county']
        }

        self.all_outputs = {}
        self.output_file_prefix = os.path.join(
            OUTPUT_DIR,
            self.summary['state'],
            f"{self.summary['state']}__{self.summary['county']}__{self.summary['fips']}__ensemble_projections")
        self.report = PDFReportBase(filename=self.output_file_prefix + '.pdf')

        self.report.write_text_page(self.summary,
                                    title=f'PySEIR COVID19 Estimates\n{self.summary["county"]} County, {self.summary["state"]}',
                                    page_heading=f'Generated {self.summary["date_generated"]}', body_fontsize=6, title_fontsize=12)
        self.report.write_text_page(inspect.getsource(ParameterEnsembleGenerator.sample_seir_parameters),
                                    title='PySEIR Model Ensemble Parameters')

    @staticmethod
    def _run_simulation(parameter_set):
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
            print(f'Generating For Policy {suppression_policy}')
            parameter_ensemble = ParameterEnsembleGenerator(
                fips=self.summary['fips'],
                N_samples=self.summary['N_samples'],
                t_list=self.t_list,
                suppression_policy=generate_triggered_suppression_model(
                    self.t_list, lockdown_days=self.summary['n_years'] * 365, open_days=1, reduction=suppression_policy)
            ).sample_seir_parameters()

            model_ensemble = list(map(self._run_simulation, parameter_ensemble))

            print(f'Generating Report for suppression policy {suppression_policy}')
            self.generate_output(model_ensemble, suppression_policy)
        self.report.close()

        # Write out the model results to json
        model_output_filename = os.path.join(OUTPUT_DIR, self.summary['state'],
                               f"{self.summary['state']}__{self.summary['county']}__{self.summary['fips']}__ensemble_projections.json")
        with open(model_output_filename, 'w') as f:
            json.dump(self.all_outputs, f)

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

    def generate_output(self, model_ensemble, suppression_policy):
        """
        Generate a county level report.

        Parameters
        ----------
        model_ensemble

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

            # Compute the peak times for each compartment by finding the arg
            # max, and selecting the corresponding time.
            peak_indices = value_stack.argmax(axis=1)

            # TODO Convert to dates?
            outputs[compartment]['peak_times'] = [outputs['t_list'][peak_index] for peak_index in peak_indices]
            values_at_peak_index = [model[idx] for model, idx in zip(value_stack, peak_indices)]
            outputs[compartment]['peak_values'] = values_at_peak_index
            for percentile in self.output_percentiles:
                outputs[compartment]['peak_value_ci%i' % percentile] = np.percentile(values_at_peak_index, percentile).tolist()
                outputs[compartment]['peak_time_ci%i' % percentile] = np.percentile(outputs[compartment]['peak_times'], percentile).tolist()

        outputs['HICU']['capacity'] = [m.beds_ICU for m in model_ensemble]
        outputs['HVent']['capacity'] = [m.ventilators for m in model_ensemble]
        outputs['HGen']['capacity'] = [m.beds_general for m in model_ensemble]

        self.all_outputs[f'suppression_policy__{suppression_policy}'] = outputs

        # TODO This is ugly, but I am tired.. move to separate functions soon.
        if self.skip_plots:
            return

        # Add a sample model from the ensemble.
        fig = model_ensemble[0].plot_results(xlim=(0, 360))
        fig.suptitle(
            f'PySEIR COVID19 Estimates: {self.summary["county"]} County, {self.summary["state"]}. '
            f'SAMPLE OF MODEL ENSEMBLE', fontsize=16)
        self.report.add_figure(fig)

        # -----------------------------------
        # Plot each compartment distribution
        # -----------------------------------
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle(f'PySEIR COVID19 Estimates: {self.summary["county"]} County, {self.summary["state"]}. '
                     f'\nSupression Policy={suppression_policy} (1=No Suppression)' , fontsize=16)
        for i_plot, compartment in enumerate(compartments):
            plt.subplot(4, 5, i_plot + 1)
            plt.plot(outputs['t_list'], outputs[compartment]['ci_50'], color='steelblue',
                     linewidth=3, label=compartment_to_name_map[compartment])
            plt.fill_between(outputs['t_list'], outputs[compartment]['ci_32'], outputs[compartment]['ci_68'], alpha=.3, color='steelblue')
            plt.fill_between(outputs['t_list'], outputs[compartment]['ci_5'], outputs[compartment]['ci_95'], alpha=.3, color='steelblue')
            plt.yscale('log')
            plt.ylim(1e1)
            plt.xlim(0, 360)
            plt.grid(True, which='both', alpha=0.3)

            plt.xlabel('Days Since 5 Cases')
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
            plt.legend()
            self._plot_dates(log=False)

        # -----------------------------
        # Plot peak Timing
        # -----------------------------
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + list('bgrcmyk')
        plt.subplot(4, 5, len(compartments) + 1)
        for i, compartment in enumerate(['E', 'A', 'I', 'HGen', 'HICU', 'HVent']):
            median = outputs[compartment]['peak_time_ci50']
            ci5, ci95 = outputs[compartment]['peak_time_ci5'], outputs[compartment]['peak_time_ci95']
            ci32, ci68 = outputs[compartment]['peak_time_ci32'], outputs[compartment]['peak_time_ci68']
            plt.scatter(median, i, label=compartment_to_name_map[compartment], c=color_cycle[i])
            plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i])
            plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
        self._plot_dates(log=False)
        plt.legend(loc=(1.05, 0.0))
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('Peak Time After $t_0(C=5)$ [Days]')
        plt.yticks([])

        # -----------------------------
        # Plot peak capacity
        # -----------------------------
        plt.subplot(4, 5, len(compartments) + 3)
        for i, compartment in enumerate(['E', 'A', 'I', 'R', 'HGen', 'HICU', 'HVent', 'D', 'total_deaths',
                                         'HGen_cumulative', 'HICU_cumulative', 'HVent_cumulative']):
            median = outputs[compartment]['peak_value_ci50']
            ci5, ci95 = outputs[compartment]['peak_value_ci5'], outputs[compartment]['peak_value_ci95']
            ci32, ci68 = outputs[compartment]['peak_value_ci32'], outputs[compartment]['peak_value_ci68']
            plt.scatter(median, i, label=compartment_to_name_map[compartment], c=color_cycle[i])
            plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i])
            plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
            plt.xscale('log')
        plt.vlines(self.summary['population'], *plt.ylim(), label='Entire Population', alpha=0.5, color='g')
        plt.vlines(self.summary['population'] * 0.65, *plt.ylim(), label='Approx. Herd Immunity',
                   alpha=0.5, color='purple', linestyles='--', linewidths=2)
        plt.legend(loc=(1.05, 0.0))
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('Value at Peak')
        plt.yticks([])

        self.report.add_figure(fig)


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
    df = load_county_metadata()
    all_fips = df[df['state'].str.lower() == state.lower()].fips
    p = Pool()
    f = partial(_run_county, ensemble_kwargs=ensemble_kwargs)
    p.map(f, all_fips)
    p.close()
