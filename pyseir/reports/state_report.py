import os
from copy import deepcopy
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import numpy as np
from pyseir.reports.names import compartment_to_name_map
from pyseir import load_data
from pyseir.inference import fit_results
from pyseir.reports.pdf_report_base import PDFReportBase
from pyseir import OUTPUT_DIR
import textwrap


class StateReport:

    def __init__(self, state, reference_date=datetime(day=1, month=3, year=2020),
                 plot_compartments=('HICU', 'HGen', 'HVent'),
                 primary_suppression_policy='suppression_policy__0.5'):
        self.state = state
        self.reference_date = reference_date
        self.plot_compartments = plot_compartments
        self.primary_suppression_policy = primary_suppression_policy

        # Load the county metadata and extract names for the state.
        county_metadata = load_data.load_county_metadata()
        self.counties = county_metadata[county_metadata['state'].str.lower() == self.state.lower()]['fips']
        self.ensemble_data_by_county = {fips: load_data.load_ensemble_results(fips) for fips in self.counties}
        self.county_metadata = county_metadata.set_index('fips')
        self.names = [self.county_metadata.loc[fips, 'county'].replace(' County', '') for fips in self.counties]
        self.filename = os.path.join(OUTPUT_DIR, self.state, 'reports', f"summary__{self.state.title()}__state_report.pdf")

    def generate_report(self):
        """
        Generate a full report for the state.
        """
        report = PDFReportBase(filename=self.filename)
        for compartment in self.plot_compartments:
            fig = self.plot_compartment(compartment)
            report.add_figure(fig=fig)
        report.close()

    def plot_compartment(self, compartment):
        """
        Plot state level data on a compartment.

        Parameters
        ----------
        compartment: str
            Compartment of the model to plot.
        primary_suppression_policy: str
            Best estimate of the true suppression policy. Gets a little extra
            love in the plots, such as confidence intervals.
        """
        fig = plt.figure(figsize=(30, 20))
        plt.suptitle(f'{self.state.title()}: Median Peak Estimates for {compartment_to_name_map[compartment]} Surges', fontsize=20)

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + list('bgrcmyk')
        for i_plt, suppression_policy in enumerate(list(self.ensemble_data_by_county.values())[0].keys()):
            # ---------------------------------------------------------
            # Plot Peak Times and values These need to be shifter by t0
            # ---------------------------------------------------------
            plt.subplot(1, 2, 1)
            peak_times = [fit_results.load_t0(fips) + timedelta(days=self.ensemble_data_by_county[fips][suppression_policy][compartment]['peak_time_ci50'])
                          for fips in self.counties]

            sorted_times = sorted(deepcopy(peak_times))
            median_statewide_peak = sorted_times[len(sorted_times)//2]

            plt.scatter(peak_times, self.names, label=f'{suppression_policy}', c=color_cycle[i_plt])
            plt.vlines(median_statewide_peak, 0, len(self.names), alpha=1, linestyle='-.', colors=color_cycle[i_plt], label=f'State Median: {suppression_policy}')

            plt.subplot(1, 2, 2)
            peak_values = [self.ensemble_data_by_county[fips][suppression_policy][compartment]['peak_value_ci50'] for fips in self.counties]
            plt.scatter(peak_values, self.names, label=suppression_policy, c=color_cycle[i_plt])

            if suppression_policy == self.primary_suppression_policy:
                plt.subplot(121)
                for i, fips in enumerate(self.counties):
                    value='peak_time'
                    d = self.ensemble_data_by_county[fips][suppression_policy][compartment]
                    t0 = fit_results.load_t0(fips)

                    plt.fill_betweenx([i-.2, i+.2],
                                      [t0 + timedelta(days=d[f'{value}_ci5'])]*2,
                                      [t0 + timedelta(days=d[f'{value}_ci95'])]*2,
                                      alpha=.3, color=color_cycle[i_plt])

                    plt.fill_betweenx([i-.2, i+.2],
                                      [t0 + timedelta(days=d[f'{value}_ci32'])]*2,
                                      [t0 + timedelta(days=d[f'{value}_ci68'])]*2,
                                      alpha=.3, color=color_cycle[i_plt])
                    plt.grid(alpha=.4)
                    plt.xlabel(value)

                ticks = []
                for month in range(1, 13):
                    ticks.append(datetime(month=month, day=1, year=2020))
                    ticks.append(datetime(month=month, day=15, year=2020))
                for month in range(1, 13):
                    ticks.append(datetime(month=month, day=1, year=2021))
                    ticks.append(datetime(month=month, day=15, year=2021))
                plt.xticks(ticks, rotation=30)

                # --------------------------
                # Plot Peak Values
                # --------------------------
                plt.subplot(1, 2, 2)
                plot_capacity = 'capacity' in self.ensemble_data_by_county[fips][suppression_policy][compartment]
                if plot_capacity:
                    capacities = np.median(np.vstack([self.ensemble_data_by_county[fips][suppression_policy][compartment]['capacity'] for fips in self.counties]), axis=1)
                    plt.scatter(capacities, self.names, marker='<', s=100, c='r', label='Estimated Capacity')

                for i, fips in enumerate(self.counties):
                    value = 'peak_value'
                    d = self.ensemble_data_by_county[fips][suppression_policy][compartment]
                    plt.fill_betweenx([i-.2, i+.2], [d[f'{value}_ci5']]*2, [d[f'{value}_ci95']]*2, alpha=.3, color=color_cycle[i_plt])
                    plt.fill_betweenx([i-.2, i+.2], [d[f'{value}_ci32']]*2, [d[f'{value}_ci68']]*2, alpha=.3, color=color_cycle[i_plt])
                    plt.grid(which='both', alpha=.4)
                    plt.xlabel('Required Surge Capacity at Peak', fontsize=14)
                    plt.xscale('log')

                if plot_capacity:
                    up_lim = plt.xlim()[1]
                    for i, (capacity, peak_value) in enumerate(zip(capacities, peak_values)):
                        if np.isnan(capacity) or capacity == 0:
                            try:
                                plt.text(up_lim * 1.3, i - .5, f'UNKNOWN CAPACITY: %s NEEDED' % int(peak_value), color='r')
                            except ValueError:
                                print('Error estimating peak. NaN')
                        else:
                            plt.text(up_lim*1.3, i-.5, f'Surge {peak_value / capacity * 100:.0f}%: {peak_value - capacity:.0f} Needed', color='r')

                    plt.text(.01, .01, f'Surge Capacity Listed for {suppression_policy}', transform=plt.gca().transAxes, color='r', fontsize=16)

        plt.subplot(121)
        caption = textwrap.fill(textwrap.dedent("""
            Surge Peak Timing: Timing of the surge peak under different
            suppression policies. A suppression policy of 0.7 implies contact is
            reduced by 30% (i.e. 30% efficacy of social distancing). Overall
            trends show that higher suppression leads to much longer time until
            surge peak. Several rural counties have imputed start times which
            may be artificially biased to peak sooner. Suppression values below
            ~0.25 (not shown) drive R0 < 1 and decay over time though it is
            unlikely this is achievable.
            
            Error bars represent (68%, 95%) CL based on en ensemble of
            parameters sampled in the appendix for a "best-guess" suppression
            model. Dashed lines indicate the state-wide median. Notably, the
            impact of distancing measures is significantly larger than variance
            associated the epidemiological model suggesting that policy may be
            used to spread these peaks relative to each other to reduce
            coincident surge."""), width=120)
        plt.text(0, 1.01, caption, ha='left', va='bottom', transform=plt.gca().transAxes)
        plt.legend()
        plt.subplot(122)

        caption = textwrap.fill(textwrap.dedent(f"""
            Surge Peak Levels: Value of the surge peak under different
            suppression policies. A suppression policy of 0.7 implies contact is
            reduced by 30% (i.e. 30% efficacy of social distancing). Overall
            trends show that higher suppression leads to much lower peak levels
            as the "curve is flattened".

            Error bars represent (68%, 95%) CL based on en ensemble of
            parameters sampled in the appendix for a "best-guess" suppression
            model: {suppression_policy}. Capacity is estimated based on
            aggregating hospital estimates from "Definitive"" to the county
            level. For beds these estimates are (N_total - utilized + estimated
            increase) with each term based on Definitive projections which
            account for utilization (Checking this!). For ventilators, we
            estimate nationally 1.1 ventilators per ICU bed which includes
            national emergency stockpile and a ~30% efficacy of an estimated
            100k old ventilators."""), width=120)
        plt.text(0, 1.01, caption, ha='left', va='bottom', transform=plt.gca().transAxes)

        plt.legend()

        return fig
