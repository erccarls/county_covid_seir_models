import click
from pyseir.load_data import cache_all_data


@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    pass


@entry_point.command()
def download_data():
    cache_all_data()
