# coding: utf8

import os
from os import path
import pickle

import numpy as np

from workflow import TsvRFWf, TsvSVMWf

__author__ = "Jorge Samper Gonzalez"
__copyright__ = "Copyright 2016-2018, The Aramis Lab Team"
__credits__ = ["Jorge Samper Gonzalez"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Jorge Samper Gonzalez"
__email__ = "jorge.samper-gonzalez@inria.fr"
__status__ = "Development"


def rf_classifications(model_name, columns, data_tsv_template, indices_template, output_dir, months, n_iterations=250,
                       test_size=0.2, n_threads=40, balanced=True, n_estimators_range=100, max_depth_range=5,
                       min_samples_split_range=2, max_features_range='auto', inner_cv=False):

    for i in months:

        with open(indices_template % i, 'r') as ind:
            splits_indices = pickle.load(ind)

        # #### Random Forest classifications ##### #
        classification_dir = path.join(output_dir, '%s_months' % i, model_name)

        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

        print("Running %s" % classification_dir)

        wf = TsvRFWf(data_tsv_template % i, columns, classification_dir, n_threads=n_threads, n_iterations=n_iterations,
                     test_size=test_size, balanced=balanced, n_estimators_range=n_estimators_range,
                     max_depth_range=max_depth_range, min_samples_split_range=min_samples_split_range,
                     max_features_range=max_features_range, splits_indices=splits_indices, inner_cv=inner_cv)
        wf.run()


def svm_classifications(model_name, columns, data_tsv_template, indices_template, output_dir, months, n_iterations=250,
                        test_size=0.2, n_threads=40, grid_search_folds=10, balanced=True,
                        c_range=np.logspace(-6, 2, 17)):

    for i in months:

        with open(indices_template % i, 'r') as ind:
            splits_indices = pickle.load(ind)

        # #### SVM classifications ##### #
        classification_dir = path.join(output_dir, '%s_months' % i, model_name)

        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

        print("Running %s" % classification_dir)

        wf = TsvSVMWf(data_tsv_template % i, columns, classification_dir, n_threads=n_threads,
                      n_iterations=n_iterations, test_size=test_size, grid_search_folds=grid_search_folds,
                      balanced=balanced, c_range=c_range, splits_indices=splits_indices)
        wf.run()
