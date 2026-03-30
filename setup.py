from setuptools import setup, find_packages

setup(
    name='neuro_ews',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'matplotlib>=3.10.8',
        'torch>=2.2.2'
    ],
)