from setuptools import setup

setup(name='local_outlier_factor_plugin',
    version='1.1.0',
    description='pygeoapi plugin that wraps sklearn.neighbors.LocalOutlierFactor for doing outlier/novelty detection on geospatial (point) datasets',
    url='https://github.com/manaakiwhenua/local-outlier-factor-plugin',
    author='Richard Law',
    author_email='lawr@landcareresearch.co.nz',
    license='MIT',
    packages=['local_outlier_factor_plugin'],
    install_requires=[
        'geopandas>=0.8.1',
        'numpy>=1.19.1',
        'scikit-learn>=0.23.1'
    ],
    zip_safe=False
)
