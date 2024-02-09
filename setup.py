from setuptools import setup, find_packages

setup(
    name='mseetc',
    version='0.0.1',
    url='https://github.com/dkouzoup/ms-eetc.git',
    author='Dimitris Kouzoupis',
    description='Multiple shooting for computationally \
        efficient train trajectory optimization',
    packages=find_packages(),    
    install_requires=['numpy == 1.22.2', 'pandas == 1.4.0', \
        'casadi==3.6.3', 'matplotlib==3.5.1', 'progressbar2'],
)
