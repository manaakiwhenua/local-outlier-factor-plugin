# =================================================================
#
# Authors: Richard Law <lawr@landcareresearch.co.nz>
#
# Copyright (c) 2020 Richard Law
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# =================================================================

from os import makedirs, path
from logging import getLogger
from numpy import vstack

from pygeoapi.process.base import BaseProcessor

from geopandas import read_file
from sklearn.neighbors import LocalOutlierFactor

LOGGER = getLogger(__name__)

#: Process metadata and description
PROCESS_METADATA = {
    'version': '1.0.0',
    'id': 'local-outlier-factor',
    'title': 'Local outlier factor (LOF)',
    'description': 'The local outlier factor (LOF) algorithm computes a score indicating the degree of abnormality of each input (observation), in a set of such observations. It measures the local density deviation of a given data point with respect to its neighbors. It considers as outliers the samples that have a substantially lower density than their neighbors.',
    'keywords': ['local outliter factor', 'LOF', 'outlier detection'],
    'links': [{
        'type': 'text/html',
        'rel': 'canonical',
        'title': 'information',
        'href': 'https://scikit-learn.org/stable/modules/outlier_detection.html#local-outlier-factor',
        'hreflang': 'en-US'
    }],
    'inputs': [
    {
        'id': 'training_dataset',
        'title': 'Training dataset',
        'abstract': 'CSV dataset of points, in one CRS, which represent data not polluted by outliers. The non-training dataset/s will be considered as new observations and outliers ("novelties") will be detected within them. If training data is omitted, each dataset will be considered in isolation for outlier detection.',
        'input': {
            'formats': [
                {
                    'default': True,
                    'mimeType': 'text/csv'
                }
            ]
        },
        'minOccurs': 0,
        'maxOccurs': 1,
        'keywords': ['csv', 'point data', 'training data']
    },
    {
        'id': 'datasets',
        'title': 'Datasets',
        'abstract': 'CSV dataset/s of points, in one CRS, for which LOF scores should be computed. If included alongside training data, the training data will be used to fit the isolation estimator for all non-training datasets (novelty detection). Otherwise each non-training dataset will be used in isolation for outlier detection',
        'input': {
            'formats': [
                {
                    'default': True,
                    'mimeType': 'text/csv'
                }
            ]
        },
        'minOccurs': 1,
        'keywords': ['csv', 'point data']
    },
    {
        'id': 'n_neighbors',
        'title': 'Number of neighbors',
        'abstract': 'Number of neighbors to use by default for `kneighbors` queries. If `n_neighbors` is larger than the number of samples provided, all samples will be used.',
        'minOccurs': 1,
        'maxOccurs': 1,
        'input': {
            'literalDataDomain': {
                'dataType': 'integer',
                'valueDefinition': {
                    'defaultValue': 20,
                    'anyValue': True
                }
            }
        }
    }, {
        'id': 'algorithm',
        'title': 'Algorithm',
        'abstract': 'Algorithm used to compute the nearest neighbors',
        'minOccurs': 1,
        'maxOccurs': 1,
        'input': {
            'literalDataDomain': {
                'dataType': 'enum',
                'valueDefinition': {
                    'anyValue': False,
                    'defaultValue': 'auto',
                    'possibleValues': ['ball_tree', 'kd_tree', 'brute', 'auto']
                }
            }
        }
    }, {
        'id': 'leaf_size',
        'title': 'Leaf size',
        'abstract': 'Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.',
        'minOccurs': 0,
        'maxOccurs': 1,
        'input': {
            'literalDataDomain': {
                'dataType': 'integer',
                'valueDefinition': {
                    'defaultValue': 30,
                    'anyValue': True
                }
            }
        }
    }, {
        'id': 'metric',
        'title': 'Distance computation metric',
        'abstract': 'Metric used for the distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.',
        'minOccurs': 1,
        'maxOccurs': 1,
        'input': {
            'literalDataDomain': {
                'dataType': 'enum',
                'valueDefinition': {
                    'anyValue': False,
                    'defaultValue': 'minkowski',
                    'possibleValues': [
                        'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
                        'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
                    ]
                }
            }
        }
    }, {
        'id': 'p',
        'title': 'Minkowski metric parameter',
        'abstract': 'Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.',
        'minOccurs': 0,
        'maxOccurs': 1,
        'input': {
            'literalDataDomain': {
                'dataType': 'integer',
                'valueDefinition': {
                    'defaultValue': 2,
                    'anyValue': True
                }
            }
        }
    },
    # {
    #     'id': 'metric_params',
    #     'title': 'Additional arguments for the metric function',
    #     'minOccurs': 0,
    #     'maxOccurs': 1,
    #     # TODO dict type?
    # }
    # {
    #     'id': 'contamination',
    #     'title': 'Contamination of the dataset',
    #     'abstract': 'The amount of contamination of the data set, i.e. the proportion of outliers in the data set. When fitting this is used to define the threshold on the scores of the samples. If "auto", the threshold is determined as in the original paper, if a float, the contamination should be in the range [0, 0.5].',
    #     # TODO how to do a mixed-type arg? ("auto" or float in range [0,0.5] ?)
    # }
    {
        'id': 'output_column',
        'title': 'Output column name',
        'abstract': 'Name of the column in which to store output metric. If this column exists, an error will be thrown',
        'minOccurs': 0,
        'maxOccurs': 1,
        'input': {
            'literalDataDomain': {
                'dataType': 'string',
                'valueDefinition': {
                    'anyValue': True,
                    'defaultValue': 'abnormality'
                }
            }
        }
    },
    # {
    #     'id': 'separator',
    #     'title': 'Output column separator',
    #     'abstract': 'String of length 1. Field delimiter for the output file',
    #     'minOccurs': 0,
    #     'maxOccurs': 1,
    #     'input': {
    #         'literalDataDomain': {
    #             'dataType': 'string',
    #             'valueDefinition': {
    #                 'anyValue': True,
    #                 'defaultValue': ','
    #             }
    #         }
    #     }
    # },
    ],
    'outputs': [{
        'id': 'dataset',
        'title': 'Dataset',
        'description': 'The (or one of) the original non-training CSV dataset(s), with an additional LOF column with the computed outlier or novelty score for each row. The new column will be labelled according to the input `output_column` parameter. Inliners are labelled 1, while outliers are labelled -1.',
        'output': {
            'formats': [{
                'default': True,
                'mimeType': 'text/csv'
            }]
        }
    }],
    'example': {}
}

# Parameters that are NOT passed directly to sklearn.neighbors.LocalOutlierFactor
LOF_OMIT = ['training_dataset', 'datasets', 'output_column']

class LOFProcessor(BaseProcessor):
    """Local outlier factor (LOF) processor"""

    def __init__(self, processor_def):
        """
        Initialize object
        :param processor_def: provider definition
        :returns: pygeoapi.process.hello_world.HelloWorldProcessor
        """

        BaseProcessor.__init__(self, processor_def, PROCESS_METADATA)

    def execute(self, data):
        LOGGER.debug(data)
        data['p'] = int(data.get('p', 2))
        data['leaf_size'] = int(data.get('leaf_size', 30))
        data['n_neighbors'] = int(data.get('n_neighbors', 20))
        colName = data.get('output_column', 'abnormality')
        trainingDataset = data.get('training_dataset', None)
        if trainingDataset:
            clf = LocalOutlierFactor(
                novelty=True,
                **{k:v for k,v in data.items()
                if k not in LOF_OMIT}
            )
            gdf = read_file(trainingDataset)
            X_train = vstack([gdf.geometry.x.ravel(), gdf.geometry.y.ravel()]).T
            clf.fit(X_train)
            predictMethod = clf.predict
        else:
            clf = LocalOutlierFactor(
                novelty=False,
                **{k:v for k,v in data.items()
                if k not in LOF_OMIT}
            )
            predictMethod = clf.fit_predict

        datasets = data.get('datasets', [])
        if not isinstance(datasets, list):
            datasets = [datasets]
        for dataset in datasets:
            gdf = read_file(dataset)
            X = vstack([gdf.geometry.x.ravel(), gdf.geometry.y.ravel()]).T
            y_pred = predictMethod(X)
            if colName in gdf.columns:
                raise Exception(f'{colName} exists in input and will not be overwritten')
            gdf[colName] = y_pred
            LOGGER.debug(gdf)
            # Make output directory at same level as input
            outputDir = path.join(path.dirname(dataset), 'output')
            outputFile = path.split(dataset)[-1]
            outputPath = path.join(outputDir, outputFile)
            # Write output
            # mode = 0o444 # Read-only (for owner, group and others)
            mode = 0o777 # TODO dangerous
            makedirs(outputDir, mode=mode, exist_ok=True)
            LOGGER.debug(outputPath)
            gdf.to_csv(outputPath)
        # Return path to output, perhaps direct URL?
        # TODO return URL to dataset for download, or return actual dataset
        return list(
            map(
                lambda dataset: {
                    'id': 'dataset',
                    'value': {
                        'href': f'/{outputFile}',
                        'mimeType': 'text/csv'
                    }
                },
                datasets
            )
        )

    def __repr__(self):
        return '<LOFProcessor> {}'.format(self.name)
