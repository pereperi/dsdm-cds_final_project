try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import analysis_library


def get_requirements(requirements_path='requirements.txt'):
    with open(requirements_path) as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='mylib',
    version=mylib.__version__,
    description='Final Project',
    author='Pere, Alvaro, Sebastien and Guillem',
    packages=find_packages(where='', exclude=['tests']),
    install_requires=get_requirements(),
    setup_requires=['pytest-runner', 'wheel'],
    url='https://github.com/pereperi/dsdm-cds_final_project',
    classifiers=[
        'Programming Language :: Python :: 3.10.7'
    ]
)
