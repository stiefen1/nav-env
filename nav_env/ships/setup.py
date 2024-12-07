from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("nav_env/ships/physics.pyx")
)

"""
To build the extension module, run the following command in the terminal, from the nav_env (top) directory:
python nav_env\ships\setup.py build_ext --inplace

Then in other python files, import it using:
import nav_env.ships.physics as physics

Enjoy the speedup!
"""