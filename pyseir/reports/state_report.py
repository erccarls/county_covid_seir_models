from copy import deepcopy
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from pyseir.reports.names import compartment_to_name_map
from pyseir import load_data
import numpy as np


class StateReport:

    def __init__(self, state, reference_date=datetime(day=1, month=3, year=2020),
                 plot_compartments=('HICU', 'HGen', 'HVent', 'D')):
        self.state = state
        self.reference_date = reference_date
        self.plot_compartments = plot_compartments

        # Load the county metadata and extract names for the state.
        county_metadata = load_data.load_county_metadata()
        self.counties = county_metadata[county_metadata['state'].str.lower() == self.state.lower()]['fips']
        self.ensemble_data_by_county = {fips: load_data.load_ensemble_results(fips) for fips in self.counties}
        self.county_metadatacounty_metadata = county_metadata.set_index('fips')
        self.names = [county_metadata.loc[fips, 'county'].replace(' County', '') for fips in self.counties]


    def plot_compartment(self, compartment, primary_suppression_policy='suppression_policy__0.5'):
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
        plt.suptitle(f'California; Median Peak Times: {compartment_to_name_map[compartment]}', fontsize=20)

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] + list('bgrcmyk')
        for i_plt, suppression_policy in enumerate(self.ensemble_data_by_county.keys()):
            # ---------------------------------------------------------
            # Plot Peak Times and values These need to be shifter by t0
            # ---------------------------------------------------------
            plt.subplot(1, 2, 1)
            peak_times = [load_data.load_t0(fips) + timedelta(days=self.ensemble_data_by_county[fips][suppression_policy][compartment]['peak_time_ci50'])
                          for fips in self.counties]

            sorted_times = sorted(deepcopy(peak_times))
            median_statewide_peak = sorted_times[len(sorted_times)//2]

            plt.scatter(peak_times, self.names, label=f'{suppression_policy}', c=color_cycle[i_plt])
            plt.vlines(median_statewide_peak, 0, len(self.names), alpha=1, linestyle='-.', colors=color_cycle[i_plt], label=f'State Median: {suppression_policy}')

            if suppression_policy == primary_suppression_policy:

                for i, fips in enumerate(self.counties):
                    value='peak_time'
                    d = self.ensemble_data_by_county[fips][suppression_policy][compartment]
                    t0 = load_data.load_t0(fips)

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
                for month in range(4, 11):
                    ticks.append(datetime(month=month, day=1, year=2020))
                    ticks.append(datetime(month=month, day=15, year=2020))

                plt.xticks(ticks, rotation=30)

                # --------------------------
                # Plot Peak Values
                # --------------------------
                for i, fips in enumerate(self.counties):
                    plt.subplot(1, 2, 2)
                    peak_values = [self.all_data[fips][suppression_policy][compartment]['peak_value_ci50'] for fips in self.counties]
                    plt.scatter(np.array(peak_values), self.names)
                    value = 'peak_value'
                    d = self.ensemble_data_by_county[fips][suppression_policy][compartment]
                    plt.fill_betweenx([i-.2, i+.2], [d[f'{value}_ci5']]*2, [d[f'{value}_ci95']]*2, alpha=.3, color='steelblue')
                    plt.fill_betweenx([i-.2, i+.2], [d[f'{value}_ci32']]*2, [d[f'{value}_ci68']]*2, alpha=.3, color='steelblue')
                    plt.grid(alpha=.3)
                    plt.xlabel(value)
                    plt.xscale('log')

        plt.legend()
