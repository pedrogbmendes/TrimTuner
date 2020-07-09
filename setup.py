import sys, os, setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requires = [
    'george',
    'emcee',
    'pyrfr',
    'pybnn',
    'cython',
    'scipy >= 0.13',
    'numpy >= 1.7',
    'sklearn',
    'torch',
    'torchvision',
    'nose',
    'pyyaml',
    'jinja2',
    'pybind11',
    'pybnn',
    'george',
    'direct',
    'cma',
    'theano',
    'matplotlib',
    'lasagne',
    'sgmcmc',
    'hpolib2',
    'robo']

setuptools.setup(name='trimtuner',
                version='0.0.1',
                author='Pedro Mendes, Maria Casimiro, Paolo Romano',
                author_email='pedrogoncalomendes@tecnico.ulisboa.pt',
                url='https://github.com/pedrogbmendes/TrimTuner',
                description='Optimization of the training of machine learning models in the cloud',
                long_description=long_description,
                keywords='Cloud Optimization, Machine Learining training in the Cloud, Bayesian Optimization',
                packages=setuptools.find_packages(),
                license='LICENSE.txt',
                test_suite='robo',
                install_requires=requires,
                python_requires='>=3.6')