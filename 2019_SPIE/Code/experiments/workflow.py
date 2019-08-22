
import os
from os import path

import numpy as np
import pandas as pd

from clinica.pipelines.machine_learning import base, algorithm, validation
import clinica.pipelines.machine_learning.svm_utils as utils


class TsvInput(base.MLInput):
    def __init__(self, data_tsv, columns=None):
        """

        Args:
            caps_directory:
            subjects_visits_tsv:
        """
        self._dataframe = pd.io.parsers.read_csv(data_tsv, sep='\t')
        self._columns = columns
        self._x = None
        self._y = None
        self._kernel = None

    def get_x(self):
        self._x = self._dataframe.as_matrix(self._columns)
        return self._x

    def get_y(self):
        unique = list(set(self._dataframe["diagnosis"]))
        self._y = np.array([unique.index(x) for x in self._dataframe["diagnosis"]])
        return self._y

    def get_kernel(self, kernel_function=utils.gram_matrix_linear, recompute_if_exists=False):
        """

        Returns: a numpy 2d-array.

        """
        if self._kernel is not None and not recompute_if_exists:
            return self._kernel

        if self._x is None:
            self.get_x()

        print "Computing kernel ..."
        self._kernel = kernel_function(self._x)
        print "Kernel computed"
        return self._kernel


class TsvRFWf(base.MLWorkflow):

    def __init__(self, data_tsv, columns, output_dir, n_threads=20, n_iterations=250, test_size=0.2,
                 grid_search_folds=10, balanced=True, n_estimators_range=(100, 200, 400), max_depth_range=[None],
                 min_samples_split_range=[2], max_features_range=('auto', 0.25, 0.5), splits_indices=None,
                 inner_cv=False):

        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._n_estimators_range = n_estimators_range
        self._max_depth_range = max_depth_range
        self._min_samples_split_range = min_samples_split_range
        self._max_features_range = max_features_range
        self._splits_indices = splits_indices
        self._inner_cv = inner_cv

        self._input = TsvInput(data_tsv, columns)
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()

        self._algorithm = algorithm.RandomForest(x, y, balanced=self._balanced,
                                                 grid_search_folds=self._grid_search_folds,
                                                 n_estimators_range=self._n_estimators_range,
                                                 max_depth_range=self._max_depth_range,
                                                 min_samples_split_range=self._min_samples_split_range,
                                                 max_features_range=self._max_features_range,
                                                 n_threads=self._n_threads)

        self._validation = validation.RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations,
                                                      test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._splits_indices,
                                                                     inner_cv=self._inner_cv)

        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, classifier_dir)

        self._validation.save_results(self._output_dir)


class TsvSVMWf(base.MLWorkflow):

    def __init__(self, data_tsv, columns, output_dir, n_threads=20, n_iterations=250, test_size=0.2,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), splits_indices=None):

        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range
        self._splits_indices = splits_indices

        self._input = TsvInput(data_tsv, columns)
        self._validation = None
        self._algorithm = None

    def run(self):

        y = self._input.get_y()
        kernel = self._input.get_kernel()

        self._algorithm = algorithm.DualSVMAlgorithm(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads)

        self._validation = validation.RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations,
                                                      test_size=self._test_size)
        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._splits_indices)

        classifier_dir = os.path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        # weights = self._algorithm.save_weights(classifier, classifier_dir)
        self._validation.save_results(self._output_dir)
