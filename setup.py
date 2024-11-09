from setuptools import setup, find_packages

setup(
    name='nav_env',
    version='0.1.0',
    description='A package for modelling and simulating navigation environments.',
    author='Stephen Monnet',
    author_email='stephen-monnet@hotmail.com',
    url='https://github.com/stiefen1/pso_path_planning', 
    packages=find_packages(),
    install_requires=[
        'numpy',
        'shapely',
        'matplotlib',
        'scipy',
        'pygame',
        'networkx'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)