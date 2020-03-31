import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from pyseir.models.seir_model import SEIRModel
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.models.suppression_policies import generate_triggered_suppression_model


class EnsembleRunner:

    compartment_to_name_map = {
        'S': 'Susceptible',
        'I': 'Infected',
        'E': 'Exposed',
        'A': 'Asymptomatic (Contagious)',
        'R': 'Recovered and Immune',
        'D': 'Direct Death',
        'HGen': 'Hospital Non-ICU',
        'HICU': 'Hospital ICU',
        'HVent': 'Hospital Ventilated',
        'deaths_from_hospital_bed_limits': 'Deaths: Non-ICU Capacity',
        'deaths_from_icu_bed_limits': 'Deaths: ICU Capacity',
        'deaths_from_ventilator_limits': 'Deaths: Ventilator Capacity',
        'total_deaths': 'Total Deaths (All Cause)'
    }

    output_percentiles = [5, 32, 50, 68, 95]

    def __init__(self, fips, t0, n_years=3, N_samples=1000, suppression_policy=0.6):

        t_list = np.linspace(0, 365 * n_years, 365 * n_years)
        self.summary = {
            'suppression_policy': suppression_policy,
            'fips': fips,
            'N_samples': N_samples,
            'n_years': n_years,
            't0': t0
        }

        self.parameter_ensemble = ParameterEnsembleGenerator(
            fips=fips,
            N_samples=N_samples,
            t_list=t_list,
            suppression_policy=generate_triggered_suppression_model(
                t_list, lockdown_days=n_years * 365, open_days=1, reduction=suppression_policy)
        ).sample_seir_parameters()

        self.model_ensemble = None

    @staticmethod
    def _run_simulation(parameter_set):
        model = SEIRModel(**parameter_set)
        model.run()
        return model

    def run_ensemble(self):
        p = Pool()
        self.model_ensemble = p.map(self._run_simulation, self.parameter_ensemble)
        p.close()

    def generate_output(self):

        compartments = {key: [] for key in self.model_ensemble[0].results.keys() if key not in ('t_list')}

        for model in self.model_ensemble:
            for key in compartments:
                compartments[key].append(model.results[key])

        outputs = defaultdict(dict)
        outputs['t_list'] = self.model_ensemble[0].t_list.tolist()

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

        # -----------------------------------
        # Plot each compartment distribution
        # -----------------------------------
        plt.figure(figsize=(20,20))
        for i_plot, compartment in enumerate(compartments):
            plt.subplot(4, 4, i_plot + 1)
            plt.plot(outputs['t_list'], outputs[compartment]['ci_50'], color='steelblue', linewidth=3, label=self.compartment_to_name_map[compartment])
            plt.fill_between(outputs['t_list'], outputs[compartment]['ci_32'], outputs[compartment]['ci_68'], alpha=.3, color='steelblue')
            plt.fill_between(outputs['t_list'], outputs[compartment]['ci_5'], outputs[compartment]['ci_95'], alpha=.3, color='steelblue')
            plt.yscale('log')
            plt.ylim(1e1)
            plt.xlim(0, 200)
            plt.grid(True, which='both', alpha=0.3)
            plt.legend()
            plt.xlabel('Days Since 5 Cases')

        # -----------------------------
        # Plot peak Timing
        # -----------------------------
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.subplot(4, 4, len(compartments) + 1)
        for i, compartment in enumerate(['E', 'A', 'I', 'HGen', 'HICU', 'HVent']):
            median = outputs[compartment]['peak_time_ci50']
            ci5, ci95 = outputs[compartment]['peak_time_ci5'], outputs[compartment]['peak_time_ci95']
            ci32, ci68 = outputs[compartment]['peak_time_ci32'], outputs[compartment]['peak_time_ci68']
            plt.scatter(median, i, label=self.compartment_to_name_map[compartment], c=color_cycle[i])
            plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i])
            plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
            plt.legend(loc=(1.05, 0.6))
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('Peak Time After $t_0(C=5)$ [Days]')
        plt.yticks([])

        # -----------------------------
        # Plot peak capacity
        # -----------------------------
        plt.subplot(4, 4, len(compartments) + 3)
        for i, compartment in enumerate(['E', 'A', 'I', 'HGen', 'HICU', 'HVent', 'D', 'total_deaths']):
            median = outputs[compartment]['peak_value_ci50']
            ci5, ci95 = outputs[compartment]['peak_value_ci5'], outputs[compartment]['peak_value_ci95']
            ci32, ci68 = outputs[compartment]['peak_value_ci32'], outputs[compartment]['peak_value_ci68']
            plt.scatter(median, i, label=self.compartment_to_name_map[compartment], c=color_cycle[i])
            plt.fill_betweenx([i-.3, i+.3], [ci32, ci32], [ci68, ci68], alpha=.3, color=color_cycle[i])
            plt.fill_betweenx([i-.1, i+.1], [ci5, ci5], [ci95, ci95], alpha=.3, color=color_cycle[i])
            plt.legend(loc=(-.8, 0.0))
            plt.xscale('log')
        plt.grid(True, which='both', alpha=0.3)
        plt.xlabel('Value at Peak')
        plt.yticks([])


