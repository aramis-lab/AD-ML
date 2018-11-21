# coding: utf8


import os
from os import path
import numpy as np

from clinica.pipelines.machine_learning import base, algorithm
from multipleInput import MultipleCAPSVoxelBasedInput, MultipleCAPSRegionBasedInput
from balancedLearningCurve import BalancedLearningCurveRepeatedHoldOut

__author__ = "Jorge Samper Gonzalez"
__copyright__ = "Copyright 2016-2018, The Aramis Lab Team"
__credits__ = ["Jorge Samper Gonzalez"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__maintainer__ = "Jorge Samper Gonzalez"
__email__ = "jorge.samper-gonzalez@inria.fr"
__status__ = "Development"


# This code is an example of implementation of machine learning pipelines


class MCRB_LearningCurveRepHoldOut_DualSVM(base.MLWorkflow):
    def __init__(self, caps_directory, diagnoses_caps_tsv, group_id, image_type, atlas,
                 output_dir, pvc=None, precomputed_kernel=None, n_threads=15, n_iterations=100, test_size=0.3,
                 n_learning_points=10, splits_indices=None, grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17)):

        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._n_learning_points = n_learning_points
        self._splits_indices = splits_indices
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range

        self._input = MultipleCAPSRegionBasedInput(caps_directory, diagnoses_caps_tsv, group_id, image_type, atlas, pvc, precomputed_kernel)
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel()

        self._algorithm = algorithm.DualSVMAlgorithm(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads)

        self._validation = BalancedLearningCurveRepeatedHoldOut(self._algorithm,
                                                                   n_iterations=self._n_iterations,
                                                                   test_size=self._test_size,
                                                                   n_learning_points=self._n_learning_points)

        classifier, best_params, results = self._validation.validate(y, splits_indices=self._splits_indices, n_threads=self._n_threads)

        for learning_point in range(self._n_learning_points):

            learning_point_dir = path.join(self._output_dir, 'learning_split-' + str(learning_point))

            classifier_dir = path.join(learning_point_dir, 'classifier')
            if not path.exists(classifier_dir):
                os.makedirs(classifier_dir)

            self._algorithm.save_classifier(classifier[learning_point], classifier_dir)
            self._algorithm.save_parameters(best_params[learning_point], classifier_dir)
            weights = self._algorithm.save_weights(classifier[learning_point], x, classifier_dir)

            self._input.save_weights_as_nifti(weights, classifier_dir)

        self._validation.save_results(self._output_dir)


class MCVB_LearningCurveRepHoldOut_DualSVM(base.MLWorkflow):
    def __init__(self, caps_directory, diagnoses_caps_tsv, group_id, image_type, output_dir, fwhm=0,
                 modulated="on", pvc=None, precomputed_kernel=None, mask_zeros=True, n_threads=15, n_iterations=100,
                 test_size=0.3, n_learning_points=10, splits_indices=None, grid_search_folds=10, balanced=True,
                 c_range=np.logspace(-6, 2, 17)):
        self._output_dir = output_dir
        self._n_threads = n_threads
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._n_learning_points = n_learning_points
        self._splits_indices = splits_indices
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range

        self._input = MultipleCAPSVoxelBasedInput(caps_directory, diagnoses_caps_tsv, group_id, image_type, fwhm, modulated, pvc, mask_zeros, precomputed_kernel)

        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel()

        self._algorithm = algorithm.DualSVMAlgorithm(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads)

        self._validation = BalancedLearningCurveRepeatedHoldOut(self._algorithm,
                                                                n_iterations=self._n_iterations,
                                                                test_size=self._test_size,
                                                                n_learning_points=self._n_learning_points)

        classifier, best_params, results = self._validation.validate(y, splits_indices=self._splits_indices, n_threads=self._n_threads)

        for learning_point in range(self._n_learning_points):

            learning_point_dir = path.join(self._output_dir, 'learning_split-' + str(learning_point))

            classifier_dir = path.join(learning_point_dir, 'classifier')
            if not path.exists(classifier_dir):
                os.makedirs(classifier_dir)

            print classifier_dir

            self._algorithm.save_classifier(classifier[learning_point], classifier_dir)
            self._algorithm.save_parameters(best_params[learning_point], classifier_dir)
            weights = self._algorithm.save_weights(classifier[learning_point], x, classifier_dir)

            self._input.save_weights_as_nifti(weights, classifier_dir)

        self._validation.save_results(self._output_dir)
