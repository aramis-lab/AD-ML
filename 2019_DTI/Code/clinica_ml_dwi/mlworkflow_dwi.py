# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

import os
from os import path

import numpy as np
from clinica.pipelines.machine_learning import algorithm, validation
from clinica.pipelines.machine_learning.base import MLWorkflow
from input_dwi import DWIRegionInput, DWIVoxelBasedInput
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectPercentile, SelectFromModel
import clinica.pipelines.machine_learning.svm_utils as utils
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR

class DWI_RB_RepHoldOut_DualSVM(MLWorkflow):

    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, atlas, dwi_map,
                 output_dir, n_threads=15, n_iterations=100, test_size=0.3,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), splits_indices=None):
        ## TODO, the parameter balanced here by default is True, I changed it to False just for DTI paper.

        self._output_dir = output_dir # DTI_SVM folder to contain the outputf
        self._n_threads = n_threads # number of threds is to use
        self._n_iterations = n_iterations # number of iteration to train the training set
        self._test_size = test_size # split the dataset into training and test dataset
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range # the hyperparameter to tune
        self._splits_indices = splits_indices

        self._input = DWIRegionInput(caps_directory, subjects_visits_tsv, diagnoses_tsv, atlas, dwi_map)
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel() # By default, the kernel is just a linear kernel
        if y[0] == 0:
            print 'The first label of diagnose is 0, it means the voxels with negative coefficients in the weight image are more likely to be classified as the first label in the diagnose tsv'
        else:
            print 'The first label of diagnose is 1, it means the voxels with positive coefficients in the weight image are more likely to be classified as the second label in the diagnose tsv'


        ### choose the algo used, returning a class object
        self._algorithm = algorithm.DualSVMAlgorithm(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads)

        ### choose the validation method used, returning a class object
        self._validation = validation.RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)

        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads, splits_indices=self._splits_indices)
        classifier_dir = path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)

        self._input.save_weights_as_nifti(weights, classifier_dir)

        self._validation.save_results(self._output_dir)


class DWI_VB_RepHoldOut_DualSVM(MLWorkflow):
    """
    This is a class for DWI voxel based features classification. Currently, only DTI features have been included,
    other model's feautures will be addded in the future. e.g. NODDI maps
    """

    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, output_dir, fwhm=None,
                 n_threads=15, n_iterations=100, test_size=0.3,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), splits_indices=None):
        ## TODO, the parameter balanced here by default is True, I changed it to False just for DTI paper.

        self._output_dir = output_dir # DTI_SVM folder to contain the outputf
        self._n_threads = n_threads # number of threds is to use
        self._n_iterations = n_iterations # number of iteration to train the training set
        self._test_size = test_size # split the dataset into training and test dataset
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range # the hyperparameteWr to tune
        self._splits_indices = splits_indices

        self._input = DWIVoxelBasedInput(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, fwhm)
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()
        kernel = self._input.get_kernel() 
        if y[0] == 0:
            print 'The first label of diagnose is 0, it means the voxels with negative coefficients in the weight image are more likely to be classified as the first label in the diagnose tsv'
        else:
            print 'The first label of diagnose is 1, it means the voxels with positive coefficients in the weight image are more likely to be classified as the second label in the diagnose tsv'


        self._algorithm = algorithm.DualSVMAlgorithm(kernel,
                                                     y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads)

        self._validation = validation.RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)
        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads, splits_indices=self._splits_indices)
        classifier_dir = path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)

        self._input.save_weights_as_nifti(weights, classifier_dir)

        self._validation.save_results(self._output_dir)

class DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNonNested(MLWorkflow):
    """
    This is a class for DWI voxel based features classification. Currently, only DTI features have been included,
    other model's feautures will be addded in the future. e.g. NODDI maps
    """

    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, output_dir, fwhm=None,
                 n_threads=15, n_iterations=100, test_size=0.3,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), splits_indices=None, top_k=50, feature_selection_method=None):
        ## TODO, the parameter balanced here by default is True, I changed it to False just for DTI paper.

        self._output_dir = output_dir # DTI_SVM folder to contain the outputf
        self._n_threads = n_threads # number of threds is to use
        self._n_iterations = n_iterations # number of iteration to train the training set
        self._test_size = test_size # split the dataset into training and test dataset
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range # the hyperparameteWr to tune
        self._splits_indices = splits_indices
        self._top_k = top_k
        self._feature_selection_method = feature_selection_method

        self._input = DWIVoxelBasedInput(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, fwhm)
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()

        ## feature selection for all the data
        ## This is ANOVA test Univariate FS
        ## get the training and testing data for feature selection
        if self._feature_selection_method == 'ANOVA':
            selector = SelectPercentile(f_classif, percentile=self._top_k)
            selector.fit(x, y)
        elif self._feature_selection_method == 'RF':
            clf = RandomForestClassifier(n_estimators=250, random_state=0, n_jobs=-1)
            clf.fit(x, y)
            selector = SelectFromModel(clf, threshold= self._top_k)
            selector.fit(x, y)
        elif self._feature_selection_method == 'PCA':
            selector = PCA(n_components=self._top_k)
            selector.fit(x)
        elif self._feature_selection_method == 'RFE':
            svc = SVR(kernel="linear")
            selector = RFE(estimator=svc, n_features_to_select=int(0.01 * self._top_k * x.shape[1]), step=0.5)
            selector.fit(x, y)

        else:
            print('Method has not been implemented')
	
	x_after_fs = selector.transform(x)

        print 'In total, there are %d voxels in this task' % x.shape[1]
        print 'The threshold is %f' % (self._top_k)
        print 'We select the %d most discriminative voxels' % x_after_fs.shape[1]

        kernel = utils.gram_matrix_linear(x_after_fs)
        if y[0] == 0:
            print 'The first label of diagnose is 0, it means the voxels with negative coefficients in the weight image are more likely to be classified as the first label in the diagnose tsv'
        else:
            print 'The first label of diagnose is 1, it means the voxels with positive coefficients in the weight image are more likely to be classified as the second label in the diagnose tsv'

        self._algorithm = algorithm.DualSVMAlgorithm(kernel, y,
                                                                     balanced=self._balanced,
                                                                     grid_search_folds=self._grid_search_folds,
                                                                     c_range=self._c_range,
                                                                     n_threads=self._n_threads)

        self._validation = validation.RepeatedHoldOut(self._algorithm, n_iterations=self._n_iterations,
                                                                       test_size=self._test_size)
        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads,
                                                                     splits_indices=self._splits_indices
                                                                     )
        classifier_dir = path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)

        self._input.save_weights_as_nifti(weights, classifier_dir)

        self._validation.save_results(self._output_dir)

class DWI_VB_RepHoldOut_DualSVM_FeatureSelectionNested(MLWorkflow):
    """
    This is a class for DWI voxel based features classification. Currently, only DTI features have been included,
    other model's feautures will be addded in the future. e.g. NODDI maps
    """

    def __init__(self, caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, output_dir, fwhm=None,
                 n_threads=15, n_iterations=100, test_size=0.3,
                 grid_search_folds=10, balanced=True, c_range=np.logspace(-6, 2, 17), splits_indices=None, top_k=50, feature_selection_method=None):

        self._output_dir = output_dir # DTI_SVM folder to contain the outputf
        self._n_threads = n_threads # number of threds is to use
        self._n_iterations = n_iterations # number of iteration to train the training set
        self._test_size = test_size # split the dataset into training and test dataset
        self._grid_search_folds = grid_search_folds
        self._balanced = balanced
        self._c_range = c_range # the hyperparameteWr to tune
        self._splits_indices = splits_indices
        self._top_k = top_k
        self._feature_selection_method= feature_selection_method

        self._input = DWIVoxelBasedInput(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map, tissue_type, threshold, fwhm)
        self._validation = None
        self._algorithm = None

    def run(self):

        x = self._input.get_x()
        y = self._input.get_y()
        # kernel = self._input.get_kernel()
        if y[0] == 0:
            print 'The first label of diagnose is 0, it means the voxels with negative coefficients in the weight image are more likely to be classified as the first label in the diagnose tsv'
        else:
            print 'The first label of diagnose is 1, it means the voxels with positive coefficients in the weight image are more likely to be classified as the second label in the diagnose tsv'


        self._algorithm = algorithm.DualSVMAlgorithmFeatureSelection(x, y,
                                                     balanced=self._balanced,
                                                     grid_search_folds=self._grid_search_folds,
                                                     c_range=self._c_range,
                                                     n_threads=self._n_threads,
                                                     feature_selection_method=self._feature_selection_method)

        self._validation = validation.RepeatedHoldOutFeautureSelection(self._algorithm, n_iterations=self._n_iterations, test_size=self._test_size)
        classifier, best_params, results = self._validation.validate(y, n_threads=self._n_threads, splits_indices=self._splits_indices, top_k=self._top_k)
        classifier_dir = path.join(self._output_dir, 'classifier')
        if not path.exists(classifier_dir):
            os.makedirs(classifier_dir)

        self._algorithm.save_classifier(classifier, classifier_dir)
        self._algorithm.save_parameters(best_params, classifier_dir)
        weights = self._algorithm.save_weights(classifier, x, classifier_dir)

        self._input.save_weights_as_nifti(weights, classifier_dir)

        self._validation.save_results(self._output_dir)

