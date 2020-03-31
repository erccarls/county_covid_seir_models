import click
import us
from pyseir.load_data import cache_all_data
from pyseir.inference.initial_conditions_fitter import generate_start_times_for_state
from pyseir.ensembles.ensemble_runner import run_state



@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    pass


@entry_point.command()
def download_data():
    cache_all_data()


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def impute_start_dates(state):
    if state:
        generate_start_times_for_state(state=state)
    else:
        for state in us.states.STATES:
            try:
                generate_start_times_for_state(state=state.name)
            except ValueError as e:
                print(e)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def run_ensembles(state):

    if state:
        run_state(state, ensemble_kwargs={})
    else:
        for state in us.states.STATES:
            try:
                run_state(state, ensemble_kwargs={})
            except ValueError as e:
                print(e)
