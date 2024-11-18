from setuptools import setup

setup(
    name='mpclab_controllers',
    version='0.1',
    packages=['mpclab_controllers'],
    install_requires=['numpy', 
                      'scipy',
                      'casadi',
                      'cvxpy',
                      'matplotlib',
                      'julia']
)
