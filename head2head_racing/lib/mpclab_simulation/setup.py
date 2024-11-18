from setuptools import setup

setup(
    name='mpclab_simulation',
    version='0.1',
    packages=['mpclab_simulation'],
    install_requires=['numpy', 
                      'scipy',
                      'casadi']
)