
import os
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='synpick',
    version='1.0.0',
    description='SynPick',
    packages=find_packages(),
    ext_modules=[
    ],
    cmdclass={
    }
)
