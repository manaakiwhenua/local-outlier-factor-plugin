from setuptools import setup

setup(name='local-outlier-factor-plugin',
    version='1.0.1',
    description='pygeoapi plugin that wraps sklearn.neighbors.LocalOutlierFactor for doing outlier/novelty detection on geospatial (point) datasets',
    url='https://github.com/manaakiwhenua/local-outlier-factor-plugin',
    author='Richard Law',
    author_email='lawr@landcareresearch.co.nz',
    license='MIT',
    packages=['plugin'],
    install_requires=[
        'geopandas>=0.8.1',
        'numpy>=1.19.1',
        'sklearn>=0.23.1'
    ],
    zip_safe=False
)
