
import os
from os import path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from multiprocessing.pool import ThreadPool

from clinica.pipelines.machine_learning import base


class BalancedLearningCurveRepeatedHoldOut(base.MLValidation):

    def __init__(self, ml_algorithm, n_iterations=100, test_size=0.3, n_learning_points=10):
        self._ml_algorithm = ml_algorithm
        self._split_results = []
        self._classifier = None
        self._best_params = None
        self._cv = None
        self._n_iterations = n_iterations
        self._test_size = test_size
        self._n_learning_points = n_learning_points
        self._error_resampled_t = None
        self._error_corrected_resampled_t = None
        self._bal_accuracy_resampled_t = None
        self._bal_accuracy_corrected_resampled_t = None

    def validate(self, y, splits_indices=None, n_threads=15):

        if splits_indices is None:
            splits = StratifiedShuffleSplit(n_splits=self._n_iterations, test_size=self._test_size)
            self._cv = list(splits.split(np.zeros(len(y)), y))
        else:
            self._cv = splits_indices['outer']
        async_pool = ThreadPool(n_threads)
        async_result = {}

        for i in range(self._n_iterations):
            train_index, test_index = self._cv[i]
            async_result[i] = {}

            if splits_indices is None:
                skf = StratifiedKFold(n_splits=self._n_learning_points, shuffle=False)
                inner_cv = list(skf.split(np.zeros(len(y[train_index])), y[train_index]))
            else:
                inner_cv = splits_indices['inner'][i]

            for j in range(self._n_learning_points):
                inner_train_index = np.concatenate([indexes[1] for indexes in inner_cv[:j + 1]]).ravel()
                async_result[i][j] = async_pool.apply_async(self._ml_algorithm.evaluate, (train_index[inner_train_index], test_index))

        async_pool.close()
        async_pool.join()

        for j in range(self._n_learning_points):
            learning_point_results = []
            for i in range(self._n_iterations):
                learning_point_results.append(async_result[i][j].get())

            self._split_results.append(learning_point_results)

        self._classifier = []
        self._best_params = []
        for j in range(self._n_learning_points):
            classifier, best_params = self._ml_algorithm.apply_best_parameters(self._split_results[j])
            self._classifier.append(classifier)
            self._best_params.append(best_params)

        return self._classifier, self._best_params, self._split_results

    def save_results(self, output_dir):
        if self._split_results is None:
            raise Exception("No results to save. Method validate() must be run before save_results().")

        for learning_point in range(self._n_learning_points):

            all_results_list = []
            all_subjects_list = []

            learning_point_dir = path.join(output_dir, 'learning_split-' + str(learning_point))

            for iteration in range(self._n_iterations):

                iteration_dir = path.join(learning_point_dir, 'iteration-' + str(iteration))
                if not path.exists(iteration_dir):
                    os.makedirs(iteration_dir)
                iteration_subjects_df = pd.DataFrame({'y': self._split_results[learning_point][iteration]['y'],
                                                      'y_hat': self._split_results[learning_point][iteration]['y_hat'],
                                                      'y_index': self._split_results[learning_point][iteration]['y_index']})
                iteration_subjects_df.to_csv(path.join(iteration_dir, 'subjects.tsv'),
                                             index=False, sep='\t', encoding='utf-8')
                all_subjects_list.append(iteration_subjects_df)

                print self._split_results[learning_point][iteration].keys()

                iteration_results_df = pd.DataFrame(
                        {'balanced_accuracy': self._split_results[learning_point][iteration]['evaluation']['balanced_accuracy'],
                         'auc': self._split_results[learning_point][iteration]['auc'],
                         'accuracy': self._split_results[learning_point][iteration]['evaluation']['accuracy'],
                         'sensitivity': self._split_results[learning_point][iteration]['evaluation']['sensitivity'],
                         'specificity': self._split_results[learning_point][iteration]['evaluation']['specificity'],
                         'ppv': self._split_results[learning_point][iteration]['evaluation']['ppv'],
                         'npv': self._split_results[learning_point][iteration]['evaluation']['npv'],

                         'train_balanced_accuracy': self._split_results[learning_point][iteration]['evaluation_train']['balanced_accuracy'],
                         'train_accuracy': self._split_results[learning_point][iteration]['evaluation_train']['accuracy'],
                         'train_sensitivity': self._split_results[learning_point][iteration]['evaluation_train']['sensitivity'],
                         'train_specificity': self._split_results[learning_point][iteration]['evaluation_train']['specificity'],
                         'train_ppv': self._split_results[learning_point][iteration]['evaluation_train']['ppv'],
                         'train_npv': self._split_results[learning_point][iteration]['evaluation_train']['npv']}, index=['i', ])

                iteration_results_df.to_csv(path.join(iteration_dir, 'results.tsv'),
                                            index=False, sep='\t', encoding='utf-8')

                mean_results_df = pd.DataFrame(iteration_results_df.apply(np.nanmean).to_dict(),
                                               columns=iteration_results_df.columns, index=[0, ])
                mean_results_df.to_csv(path.join(iteration_dir, 'mean_results.tsv'),
                                       index=False, sep='\t', encoding='utf-8')
                all_results_list.append(mean_results_df)

            all_subjects_df = pd.concat(all_subjects_list)
            all_subjects_df.to_csv(path.join(learning_point_dir, 'subjects.tsv'),
                                   index=False, sep='\t', encoding='utf-8')

            all_results_df = pd.concat(all_results_list)
            all_results_df.to_csv(path.join(learning_point_dir, 'results.tsv'),
                                  index=False, sep='\t', encoding='utf-8')

            mean_results_df = pd.DataFrame(all_results_df.apply(np.nanmean).to_dict(),
                                           columns=all_results_df.columns, index=[0, ])
            mean_results_df.to_csv(path.join(learning_point_dir, 'mean_results.tsv'),
                                   index=False, sep='\t', encoding='utf-8')

            self.compute_error_variance(learning_point)
            self.compute_accuracy_variance(learning_point)

            variance_df = pd.DataFrame({'bal_accuracy_resampled_t': self._bal_accuracy_resampled_t,
                                        'bal_accuracy_corrected_resampled_t': self._bal_accuracy_corrected_resampled_t,
                                        'error_resampled_t': self._error_resampled_t,
                                        'error_corrected_resampled_t': self._error_corrected_resampled_t}, index=[0, ])

            variance_df.to_csv(path.join(learning_point_dir, 'variance.tsv'),
                               index=False, sep='\t', encoding='utf-8')

    def _compute_variance(self, test_error_split):

        # compute average test error
        num_split = self._n_iterations  # J in the paper

        # compute mu_{n_1}^{n_2}
        average_test_error = np.mean(test_error_split)

        approx_variance = np.sum((test_error_split - average_test_error)**2)/(num_split - 1)

        # compute variance (point 2 and 6 of Nadeau's paper)
        resampled_t = approx_variance / num_split
        corrected_resampled_t = (1/num_split + self._test_size/(1 - self._test_size)) * approx_variance

        return resampled_t, corrected_resampled_t

    def compute_error_variance(self, learning_point):
        num_split = self._n_iterations
        test_error_split = np.zeros((num_split, 1))  # this list will contain the list of mu_j hat for j = 1 to J
        for i in range(num_split):
            test_error_split[i] = self._compute_average_test_error(self._split_results[learning_point][i]['y'],
                                                                   self._split_results[learning_point][i]['y_hat'])

            self._error_resampled_t, self._error_corrected_resampled_t = self._compute_variance(test_error_split)

        return self._error_resampled_t, self._error_corrected_resampled_t

    def _compute_average_test_error(self, y_list, yhat_list):
        # return the average test error (denoted mu_j hat)
        return float(len(np.where(y_list != yhat_list)[0]))/float(len(y_list))

    def compute_accuracy_variance(self, learning_point):
        num_split = self._n_iterations
        test_accuracy_split = np.zeros((num_split, 1))  # this list will contain the list of mu_j hat for j = 1 to J
        for i in range(num_split):
            test_accuracy_split[i] = self._compute_average_test_accuracy(self._split_results[learning_point][i]['y'],
                                                                         self._split_results[learning_point][i]['y_hat'])

        self._bal_accuracy_resampled_t, self._bal_accuracy_corrected_resampled_t = self._compute_variance(test_accuracy_split)

        return self._bal_accuracy_resampled_t, self._bal_accuracy_corrected_resampled_t

    def _compute_average_test_accuracy(self, y_list, yhat_list):

        from clinica.pipelines.machine_learning.svm_utils import evaluate_prediction

        return evaluate_prediction(y_list, yhat_list)['balanced_accuracy']
