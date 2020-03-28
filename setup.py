from setuptools import setup, find_packages


setup(
    name="pyseir",
    version='0.1',
    description=(''),
    url='https://github.com/erccarls/county_covid_seir_models',
    author='Misc DS.',
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        pyseir=pyseir.cli:entry_point
    ''',
    package_data={
              # Package data suffixes to include when installing from git:
              '': ['*.txt', '*.rst', '*.yaml', '*.csv']
      },
    packages=find_packages(),
)
