from setuptools import setup

setup(
    name='mpclab_common',
    version='0.1',
    packages=['mpclab_common'],
    install_requires=['numpy', 
                      'scipy',
                      'casadi',
                      'pyyaml',
                      'matplotlib',
                      'sqlite3']
)
