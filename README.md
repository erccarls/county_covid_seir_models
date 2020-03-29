# County COVID-19 Compartmental Models
Tools for forecasting COVID-19 Propagation at the US County Level. We implement 
a generalized comparmental model based on the SEIR formalism.

## Dynamical Model Transitions
![](docs/images/model_flow.png)

## Examples 
### Flatten the Curve

The top figure represents unsupressed COVID19 flow through the population, while
the second figure demonstrates the impact of distancing policies.

![](docs/images/no_containment.png)
![](docs/images/containment.png)


# Installation

Install miniconda python 3.7 from here [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Execute
`conda env create -f environment.yaml`

Activate the environment here..
`conda activate pyseir`

## Installing pyseir
Change to into the county_covid_seir_models directory
`pip install -e .`


## Download Data Locally

The following will cache data for local use.
`pyseir download-data`
