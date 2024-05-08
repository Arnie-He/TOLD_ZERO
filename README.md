# tdmpc2-jax

A re-implementation of [TD-MPC2](https://www.tdmpc2.com/) in Jax/Flax. JIT'ing the planning/update steps makes training 5-10x faster compared to the original PyTorch implementation.

## Usage

To install the dependencies for this project (tested on Ubuntu 22.04), run

```[bash]
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install --upgrade tqdm numpy flax optax jaxtyping einops gymnasium[mujoco]
```

## Installation

Install the package from the base directory with

```[bash]
pip install -e .
```
## Training TOLD-ZERO

The step and returns information will be logged into the tdmpc2_jax directory by running the following

```[bash]
cd tdmpc2_jax
python train.py
```

## Visualization

Copy the logged info in the txt to data.csv in visualizations directory, then run the following

```[bash]
cd ../visualizations
python viz.py
```

The visualization will be saved in visualizations/output directory


