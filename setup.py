# -*- coding: utf-8 -*-



from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='indexes',
    version='1',
    description='Package for using cluster validation indexes',
    long_description=readme,
    author='Nuno Silva',
    author_email='nuno.da.silva@tecnico.ulisboa.pt',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

