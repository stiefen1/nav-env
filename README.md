# Installation
## Clone
Clone this repository:

```git clone git@github.com:stiefen1/nav-env.git```

## Conda Environment

Then change your current directory to nav-env and create your conda environment using the env.yml file. For this, first activate your conda environment:

```conda activate```

And create your environment using:

```conda env create --name <env-name> --file env.yml```

Don't forget to change <env-name> with the name you would like to use for your environment. Finally, activate your environment using:

```conda activate <env-name>```

## Install the library

```pip install -e .```

## Install the cython package

```python nav_env\ships\setup.py build_ext --inplace```

# Examples
To verify everything is correctly installed, run one of the examples [here](/examples/)!