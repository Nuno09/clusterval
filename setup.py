# -*- coding: utf-8 -*-



from setuptools import setup, find_packages



install_requires = [
    'sklearn', 'scipy', 'numpy', 'pandas', 'matplotlib','sympy', 'fastcluster'
]

with open('README.md') as f:
    long_description = f.read()


setup(
    name='clusterval',
    version='1.0.2',
    description='Package useful for clustering validation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Nuno Silva',
    author_email='nuno.da.silva@tecnico.ulisboa.pt',
    url='https://github.com/Nuno09/clusterval',
    license='LICENSE',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=install_requires,
    tests_require='pytest',
    setup_requires='pytest-runner',
    package_data={'clusterval': ['datasets/*.csv']}
)

