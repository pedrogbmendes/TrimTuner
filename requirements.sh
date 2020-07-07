#!/bin/sh
sudo apt-get update
sudo apt-get install libeigen3-dev swig gfortran

all_requirements="
    emcee>=2.1.0
    scipy>=0.13.3
    sklearn
    nose
    torch 
    torchvision
    pyrfr
    pyyaml
    jinja2
    pybind11
    git+https://github.com/automl/pybnn.git
    git+https://github.com/automl/george.git@development
    direct
    cma
    theano
    matplotlib
    lasagne
    git+https://github.com/stokasto/sgmcmc.git
    git+https://github.com/automl/HPOlib2.git
    git+https://github.com/automl/RoBO.git"

for req in $all_requirements; do
    printf "Installing" $req  
    pip3 install $req; 
done
#python3 setup.py install
