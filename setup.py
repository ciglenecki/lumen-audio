#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="",
    author_email="",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
