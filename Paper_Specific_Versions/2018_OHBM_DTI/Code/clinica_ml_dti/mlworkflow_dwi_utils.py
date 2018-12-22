# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Nipype"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

import os
import errno
from os import path
from pandas.io import parsers
from mlworkflow_dwi import DWI_VB_RepHoldOut_DualSVM, DWI_RB_RepHoldOut_DualSVM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from clinica.pipelines.machine_learning.ml_workflows import RB_RepHoldOut_DualSVM, VB_RepHoldOut_DualSVM
from os import path
from scipy import stats



def run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, task, n_threads, n_iterations, test_size, grid_search_folds,
                                   balanced_down_sample=False, modality='dwi', dwi_maps=['fa', 'md'], fwhm=[8],
                                   tissue_type = ['WM', 'GM', 'GM_WM'], threshold=[0.3], group_id='ADNIbl'):
    """
    This is a function to run the Voxel-based calssification tasks_imbalanced after the imaging processing pipeline of ADNI
    :param caps_directory: caps directory for Clinica outputs
    :param diagnoses_tsv:
    :param subjects_visits_tsv:
    :param output_dir: the path to store the classification outputs
    :param tissue_type: one of these:
    :param threshold: the threshold to masking the features
    :param fwhm: the threshold to smooth the mask of tissues
    :param dwi_maps: the maps based on DWI, currently, we just have maps from DTI model.
    :param balanced_down_sample: int, how many times to repeat for the downsampling procedures to creat the balanced data, default is 0, which means we do not force the data to be balanced
    :param task: the name of the task to store the classification results
    :param n_threads: number of cores to use for this classification
    :param n_iterations: number of runs for the RepHoldOut
    :param test_size: propotion for the testing dataset
    :param grid_search_folds: number of runs to search the hyperparameters

    :return:
    """

    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size, balanced=balanced_down_sample)

    if modality == 'dwi':
        # ## run the classification
        if balanced_down_sample: ###
            print "Using balanced data to do classifications!!!"
            for dwi_map in dwi_maps:
                for i in tissue_type:
                    for j in threshold:
                        for k in fwhm:
                            classification_dir = path.join(output_dir, 'RandomBalanced',
                                                           task + '_' + i + '_' + str(j) + '_' + str(k),
                                                           dwi_map)
                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print "\nRunning %s" % classification_dir

                                wf = DWI_VB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map,
                                                               i, j, classification_dir, fwhm=k,
                                                               n_threads=n_threads, n_iterations=n_iterations, test_size=test_size,
                                                               grid_search_folds=grid_search_folds, splits_indices=splits_indices)
                                wf.run()
                            else:
                                print "This combination has been classified, just skip: %s " % classification_dir


        else: ## original classifcation
            print "Using raw data to do classifications!!!"
            for dwi_map in dwi_maps:
                for i in tissue_type:
                    for j in threshold:
                        for k in fwhm:
                            classification_dir = path.join(output_dir,
                                                           task + '_' + i + '_' + str(j) + '_' + str(k),
                                                           dwi_map)
                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print "\nRunning %s" % classification_dir
                                wf = DWI_VB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, dwi_map,
                                                               i, j, classification_dir, fwhm=k,
                                                               n_threads=n_threads, n_iterations=n_iterations,
                                                               test_size=test_size,
                                                               grid_search_folds=grid_search_folds, splits_indices=splits_indices)
                                wf.run()
                            else:
                                print "This combination has been classified, just skip: %s " % classification_dir

    elif modality == 'T1': ## Run T1 classification
        if balanced_down_sample: ###
            print "Using balanced data to do classifications!!!"
            for k in fwhm:
                classification_dir = path.join(output_dir, 'RandomBalanced',
                                               task + '_fwhm_' + str(k))
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print "\nRunning %s" % classification_dir

                    wf = VB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id, modality,
                                               classification_dir, fwhm=k, n_threads=n_threads, n_iterations=n_iterations,
                                               test_size=test_size, splits_indices=splits_indices)

                    wf.run()
                else:
                    print "This combination has been classified, just skip: %s " % classification_dir

        else: ## original classifcation
            print "Using raw data to do classifications!!!"
            #### have the same lists for all the iterations
            for k in fwhm:
                classification_dir = path.join(output_dir,
                                               task + '_fwhm_' + str(k))
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print "\nRunning %s" % classification_dir
                    wf = VB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id, modality,
                                               classification_dir, fwhm=k, n_threads=n_threads,
                                               n_iterations=n_iterations,
                                               test_size=test_size, splits_indices=splits_indices)
                    wf.run()
                else:
                    print "This combination has been classified, just skip: %s " % classification_dir
    else:
        pass


def run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=False,
                                 dwi_maps=['fa', 'md'], modality='dwi', group_id='ADNIbl'):
    """
    This is a function to run the Voxel-based calssification tasks_imbalanced after the imaging processing pipeline of ADNI
    :param caps_directory: caps directory for Clinica outputs
    :param diagnoses_tsv:
    :param subjects_visits_tsv:
    :param output_dir: the path to store the classification outputs
    :param atlas: one of these: ['JHUDTI81', 'JHUTracts0', 'JHUTracts25']
    :param dwi_maps: the maps based on DWI, currently, we just have maps from DTI model.
    :param balanced_down_sample: int, how many times to repeat for the downsampling procedures to creat the balanced data, default is 0, which means we do not force the data to be balanced
    :param task: the name of the task to store the classification results
    :param n_threads: number of cores to use for this classification
    :param n_iterations: number of runs for the RepHoldOut
    :param test_size: propotion for the testing dataset
    :param grid_search_folds: number of runs to search the hyperparameters

    :return:
    """
    splits_indices, splits_indices_pickle = split_subjects_to_pickle(diagnoses_tsv, n_iterations=n_iterations,
                                                                     test_size=test_size, balanced=balanced_down_sample)

    if modality == 'dwi':
        ## run the classification
        if balanced_down_sample: ###
            print "Using balanced data to do classifications!!!"
            for dwi_map in dwi_maps:
                for i in atlas:
                            classification_dir = path.join(output_dir, 'RandomBalanced',
                                                           task + '_' + i,
                                                           dwi_map)
                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print "\nRunning %s" % classification_dir

                                wf = DWI_RB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, i, dwi_map,
                                                           classification_dir,
                                                           n_threads=n_threads, n_iterations=n_iterations,
                                                           test_size=test_size,
                                                           grid_search_folds=grid_search_folds,
                                                               splits_indices=splits_indices)
                                wf.run()
                            else:
                                print "This combination has been classified, just skip: %s " % classification_dir


        else:
            print "Using raw data to do classifications!!!"

            for dwi_map in dwi_maps:
                for i in atlas:
                            classification_dir = path.join(output_dir,
                                                           task + '_' + i,
                                                           dwi_map)
                            if not path.exists(classification_dir):
                                os.makedirs(classification_dir)

                                print "\nRunning %s" % classification_dir
                                wf = DWI_RB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, i,
                                                               dwi_map,
                                                               classification_dir,
                                                               n_threads=n_threads, n_iterations=n_iterations,
                                                               test_size=test_size,
                                                               grid_search_folds=grid_search_folds, splits_indices=splits_indices)
                                wf.run()
                            else:
                                print "This combination has been classified, just skip: %s " % classification_dir


    elif modality == 'T1':
        ## run the classification
        if balanced_down_sample:  ###
            print "Using balanced data to do classifications!!!"
            for i in atlas:
                classification_dir = path.join(output_dir, 'RandomBalanced',
                                               task + '_' + i)
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print "\nRunning %s" % classification_dir

                    wf = RB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id, modality, i,
                                                classification_dir, n_threads=n_threads, n_iterations=n_iterations, test_size=test_size,
                                               splits_indices=splits_indices)

                    wf.run()
                else:
                    print "This combination has been classified, just skip: %s " % classification_dir


        else:
            print "Using raw data to do classifications!!!"
            for i in atlas:
                classification_dir = path.join(output_dir,
                                               task + '_' + i)
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                    print "\nRunning %s" % classification_dir
                    wf = RB_RepHoldOut_DualSVM(caps_directory, subjects_visits_tsv, diagnoses_tsv, group_id,
                                               modality, i,
                                               classification_dir, n_threads=n_threads, n_iterations=n_iterations,
                                               test_size=test_size, splits_indices=splits_indices)
                    wf.run()
                else:
                    print "This combination has been classified, just skip: %s " % classification_dir

    else:
        pass

def classification_performances_violin_plot_imbalanced_vs_balanced(classification_result_path, tasks_imbalanced, tasks_balanced,
                                                                         n_iterations, raw_classification=True, feature_type='voxel', modality='dwi', figure_number=0):
    """
    This is a function to plot the classification performances among different tasks_imbalanced:
        First subplot is to plot the three tissue combinations for each tasks_imbalanced for raw data classification
        Second subplot is for balanced classifications.
    :param classification_result_path: str, should be absolute path to the classification results.
    :param tasks_imbalanced: list, containing several binary classification tasks_imbalanced
    :param n_iterations: number of iterations to use for CV process
    :param dwi_map: str, one of diffusion maps, e.g, fa
    :return:
    """
    if modality == 'dwi' and figure_number == 0:
        results_balanced_acc_fa_imbalanced = []
        results_balanced_acc_fa_balanced = []
        results_balanced_acc_md_imbalanced = []
        results_balanced_acc_md_balanced = []
        if feature_type == 'voxel':
            tissue_combinations = ['WM', 'GM', 'GM_WM']
            ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
            # ticklabels_imbalanced = ['CN vs AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
            ticklabels_balanced = [i.replace('_', ' ') for i in tasks_balanced]
            # ticklabels_balanced = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']

            if raw_classification == True:
                print "Plot for original classifications"
                ## get FA
                for task in tasks_imbalanced:
                    for tissue in tissue_combinations:
                        tsvs_path = os.path.join(classification_result_path, task + '_VB_' + tissue + '_0.3_8', 'fa')
                        balanced_accuracy = []
                        for i in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' +str(i), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_fa_imbalanced.append(balanced_accuracy)

                ## get MD
                for task in tasks_imbalanced:
                    for tissue in tissue_combinations:
                        tsvs_path = os.path.join(classification_result_path, task + '_VB_' + tissue + '_0.3_8', 'md')
                        balanced_accuracy = []
                        for i in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_md_imbalanced.append(balanced_accuracy)

                ##### FAs
                ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
                metric = np.array(results_balanced_acc_fa_imbalanced).transpose()
                ## define the violin's postions
                pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
                color = ['#FF0000', '#87CEFA', '#90EE90'] *len(tasks_imbalanced)# red, blue and green
                legend = ['WM', 'GM', 'GM+WM']

                ## define the size of th image
                fig, ax = plt.subplots(2,figsize=[15, 10])
                line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[0].grid(axis='y', which='major', linestyle='dotted')
                ax[0].set_xticks([2, 6, 10, 14])
                ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
                ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].set_ylim(0.1, 1)
                ax[0].set_title('A: FA Voxel-based classifications', fontsize=15)

                ##### MD
                ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
                metric = np.array(results_balanced_acc_md_imbalanced).transpose()
                ## define the size of th image
                line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[1].grid(axis='y', which='major', linestyle='dotted')
                ax[1].set_xticks([2, 6, 10, 14])
                ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
                ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].set_ylim(0.1, 1)
                ax[1].set_title('B: MDVoxel-based classifications', fontsize=15)

                plt.savefig(os.path.join(classification_result_path,
                                         'voxel_violin_imbalanced.png'), additional_artists=plt.legend, bbox_inches="tight")

            else:
                print "Plot for balanced classifications"
                ## get FA
                for task in tasks_balanced:
                    for tissue in tissue_combinations:
                        balanced_accuracy = []
                        tsvs_path = os.path.join(classification_result_path, 'RandomBalanced', task + '_VB_' + tissue + '_0.3_8', 'fa')
                        for k in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_fa_balanced.append(balanced_accuracy)

                ## get MD
                for task in tasks_balanced:
                    for tissue in tissue_combinations:
                        balanced_accuracy = []
                        tsvs_path = os.path.join(classification_result_path, 'RandomBalanced', task + '_VB_' + tissue + '_0.3_8', 'md')
                        for k in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append((pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
        errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_md_balanced.append(balanced_accuracy)

                ##### FA
                ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
                metric = np.array(results_balanced_acc_fa_balanced).transpose()
                ## define the violin's postions
                pos = [1, 2, 3, 5, 6, 7, 9, 10, 11]
                color = ['#FF0000', '#87CEFA', '#90EE90'] *len(tasks_imbalanced)# red, blue and green
                legend = ['WM', 'GM', 'GM+WM']

                ## define the size of th image
                fig, ax = plt.subplots(2, figsize=[15, 10])
                line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[0].grid(axis='y', which='major', linestyle='dotted')
                ax[0].set_xticks([2, 6, 10, 14])
                ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
                ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].set_ylim(0.1, 1)
                ax[0].set_title('A: FA Voxel-based classification with balanced data', fontsize=15)

                ##### MD
                ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
                metric = np.array(results_balanced_acc_md_balanced).transpose()
                ## define the size of th image
                line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[1].grid(axis='y', which='major', linestyle='dotted')
                ax[1].set_xticks([2, 6, 10, 14])
                ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
                ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].set_ylim(0.1, 1)
                ax[1].set_title('B: MD Voxel-based classification with balanced data', fontsize=15)

                plt.savefig(os.path.join(classification_result_path,
                                         'voxel_violin_balanced.png'), additional_artists=plt.legend, bbox_inches="tight")
        else:  ##### for DWI regions
            atlases = ['JHUDTI81', 'JHUTracts25']
            ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
            # ticklabels_imbalanced = ['CN vs AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
            ticklabels_balanced = [i.replace('_', ' ') for i in tasks_balanced]
            # ticklabels_balanced = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']

            if raw_classification == True:
                print "Plot for original classifications"
                ## get FA
                for task in tasks_imbalanced:
                    for atlas in atlases:
                        tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'fa')
                        balanced_accuracy = []
                        for i in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append(
                                    (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
                                    errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_fa_imbalanced.append(balanced_accuracy)

                ## get MD
                for task in tasks_imbalanced:
                    for atlas in atlases:
                        tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'md')
                        balanced_accuracy = []
                        for i in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append(
                                    (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
                                    errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_md_imbalanced.append(balanced_accuracy)

                ##### FAs
                ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
                metric = np.array(results_balanced_acc_fa_imbalanced).transpose()
                ## define the violin's postions
                pos = [1, 2, 4, 5, 7, 8, 10, 11]
                # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
                color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue
                legend = ['JHULabel', 'JHUTract']

                ## define the size of th image
                fig, ax = plt.subplots(2, figsize=[15, 10])
                line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[0].grid(axis='y', which='major', linestyle='dotted')
                ax[0].set_xticks([1.5, 4.5, 7.5, 10.5])
                ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
                ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].set_ylim(0.1, 1)
                ax[0].set_title('C: FA Region-based classifications', fontsize=15)

                ##### MD
                ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
                metric = np.array(results_balanced_acc_md_imbalanced).transpose()
                ## define the size of th image
                line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[1].legend(legend, loc='upper right', fontsize=10)
                ax[1].grid(axis='y', which='major', linestyle='dotted')
                ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
                ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
                ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].set_ylim(0.1, 1)
                ax[1].set_title('D: MD Region-based classifications', fontsize=15)

                plt.savefig(os.path.join(classification_result_path,
                                         'region_violin_imbalanced.png'), additional_artists=plt.legend,
                            bbox_inches="tight")

            else:
                print "Plot for balanced classifications"
                ## get FA
                for task in tasks_balanced:
                    for atlas in atlases:
                        balanced_accuracy = []
                        tsvs_path = os.path.join(classification_result_path, 'RandomBalanced',
                                                 task + '_RB_' + atlas, 'fa')
                        for k in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append(
                                    (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
                                    errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_fa_balanced.append(balanced_accuracy)

                ## get MD
                for task in tasks_balanced:
                    for atlas in atlases:
                        balanced_accuracy = []
                        tsvs_path = os.path.join(classification_result_path, 'RandomBalanced',
                                                 task + '_RB_' + atlas, 'md')
                        for k in xrange(n_iterations):
                            result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                            if os.path.isfile(result_tsv):
                                balanced_accuracy.append(
                                    (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                            else:
                                raise OSError(
                                    errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                        results_balanced_acc_md_balanced.append(balanced_accuracy)

                ##### FA
                ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
                metric = np.array(results_balanced_acc_fa_balanced).transpose()
                ## define the violin's postions
                pos = [1, 2, 4, 5, 7, 8]
                color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
                legend = ['JHULabel', 'JHUTract']

                ## define the size of th image
                fig, ax = plt.subplots(2, figsize=[15, 10])
                line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[0].grid(axis='y', which='major', linestyle='dotted')
                ax[0].set_xticks([1.5, 4.5, 7.5, 10.5])
                ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
                ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[0].set_ylim(0.1, 1)
                ax[0].set_title('C: FA Region-based classification with balanced data', fontsize=15)

                ##### MD
                ### transfer the list into an array with this shape: n_iterations*n_tasks_balanced
                metric = np.array(results_balanced_acc_md_balanced).transpose()
                ## define the size of th image
                line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
                for cc, ln in enumerate(line_coll['bodies']):
                    ln.set_facecolor(color[cc])

                ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
                ax[1].grid(axis='y', which='major', linestyle='dotted')
                ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
                ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
                ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
                mean = np.mean(metric, 0)
                std = np.std(metric, 0)
                inds = np.array(pos)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
                ax[1].set_ylim(0.1, 1)
                ax[1].set_title('D: MD Region-based classification with balanced data', fontsize=15)

                plt.savefig(os.path.join(classification_result_path,
                                         'region_violin_balanced.png'), additional_artists=plt.legend,
                            bbox_inches="tight")

        print 'finish DWI'
    elif modality == 'T1':
        results_balanced_acc_voxel_imbalanced = []
        results_balanced_acc_regional_imbalanced = []
        tissue_combinations = ['GM+WM']
        ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]

        if raw_classification == True:
            print "Plot for original classification, to compare T1 with DWI, we use GM+WM"
            ## T1
            for task in tasks_imbalanced:
                tsvs_path = os.path.join(classification_result_path, task + '_VB_T1_fwhm_8')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
            ## GM+WM FA
            for task in tasks_imbalanced:
                for tissue in tissue_combinations:
                    tsvs_path = os.path.join(classification_result_path,
                                             task + '_VB_' + tissue + '_0.3_8', 'fa')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)

            ## GM+WM MD
            for task in tasks_imbalanced:
                for tissue in tissue_combinations:
                    tsvs_path = os.path.join(classification_result_path,
                                             task + '_VB_' + tissue + '_0.3_8', 'md')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)

            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_balanced_acc_voxel_imbalanced).transpose()
            ## reorder the order of the column to make sure the right order in the image
            metric_new = metric[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]

            ## define the violin's postions
            pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
            legend = ['T1w+GM', 'DTI+GM+FA', 'DTI+GM+MD']

            ## define the size of th image
            fig, ax = plt.subplots(2, figsize=[15, 10])
            line_coll = ax[0].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[0].grid(axis='y', which='major', linestyle='dotted')
            ax[0].set_xticks([2, 6, 10, 14])
            ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
            ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric_new, 0)
            std = np.std(metric_new, 0)
            inds = np.array(pos)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].set_ylim(0.1, 1)
            ax[0].set_title('A: Voxel-based classifications for T1w and diffusion MRI', fontsize=15)


            ### T1 atlaes
            atlases_T1 = ['AAL2']
            for task in tasks_imbalanced:
                for atlas in atlases_T1:
                    tsvs_path = os.path.join(classification_result_path, task + '_RB_T1_' + atlas)
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_regional_imbalanced.append(balanced_accuracy)

            atlases_DTI = ['JHUDTI81']
            ## get DTI atlases FA
            for task in tasks_imbalanced:
                for atlas in atlases_DTI:
                    tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'fa')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
            #
            # ## get DTI atlases MD
            # for task in tasks_imbalanced:
            #     for atlas in atlases_DTI:
            #         tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'md')
            #         balanced_accuracy = []
            #         for i in xrange(n_iterations):
            #             result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
            #             if os.path.isfile(result_tsv):
            #                 balanced_accuracy.append(
            #                     (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
            #             else:
            #                 raise OSError(
            #                     errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            #         results_balanced_acc_regional_imbalanced.append(balanced_accuracy)

            ##### FAs
            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_balanced_acc_regional_imbalanced).transpose()
            ## reorder the order of the column to make sure the right order in the image
            metric_new = metric[:, [0, 4, 1, 5, 2, 6, 3, 7]]

            ## define the violin's postions
            pos = [1, 2, 4, 5, 7, 8, 10, 11]
            color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
            legend = ['T1w+AAL2', 'DTI+JHULabel+FA']

            ## define the size of th image
            line_coll = ax[1].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[1].grid(axis='y', which='major', linestyle='dotted')
            ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
            ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
            ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric_new, 0)
            std = np.std(metric_new, 0)
            inds = np.array(pos)
            ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[1].set_ylim(0.1, 1)
            ax[1].set_title('B: Region-based classifications for T1w and diffusion MRI', fontsize=15)
            plt.savefig(os.path.join(classification_result_path,
                                     'violin_T1_compare_dwi.png'), additional_artists=plt.legend,
                        bbox_inches="tight")
        #
        print 'finish T1'

    else:
        pass

    if figure_number == 3:
        results_acc_fa_voxel = []
        results_acc_md_voxel = []
        results_acc_fa_region = []

        if feature_type == 'voxel':
            tissue_combinations = ['GM_WM']
            ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]
            # ticklabels_imbalanced = ['CN vs AD', 'CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
            ticklabels_balanced = [i.replace('_', ' ') for i in tasks_balanced]
            # ticklabels_balanced = ['CN_vs_MCI', 'CN_vs_pMCI', 'sMCI_vs_pMCI']
            atlases = ['JHUDTI81']

            print "Plot for figure 3"

            ## for region
            ## get FA region imbalanced
            for task in tasks_imbalanced:
                for atlas in atlases:
                    tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'fa')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_acc_fa_region.append(balanced_accuracy)

            ## get FA region balanced
            for task in tasks_balanced:
                for atlas in atlases:
                    balanced_accuracy = []
                    tsvs_path = os.path.join(classification_result_path, 'RandomBalanced',
                                             task + '_RB_' + atlas, 'fa')
                    for k in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_acc_fa_region.append(balanced_accuracy)

            ##### FAs
            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_acc_fa_region).transpose()
            metric = metric[:, [0, 3, 1, 4, 2, 5]]

            ## define the violin's postions
            pos = [1, 2, 4, 5, 7, 8]
            # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
            color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue
            legend = ['JHULabel_raw', 'JHUTract_balanced']

            ## define the size of th image
            fig, ax = plt.subplots(2, figsize=[15, 10])
            line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[0].grid(axis='y', which='major', linestyle='dotted')
            ax[0].set_xticks([1.5, 4.5, 7.5])
            ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
            ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric, 0)
            std = np.std(metric, 0)
            inds = np.array(pos)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].set_ylim(0.1, 1)
            ax[0].set_title('C: FA Region-based classifications', fontsize=15)
            plt.savefig(os.path.join(classification_result_path,
                                     'figure3_C.png'), additional_artists=plt.legend,
                        bbox_inches="tight")
            ### for voxel
            ## get FA raw
            for task in tasks_imbalanced:
                for tissue in tissue_combinations:
                    tsvs_path = os.path.join(classification_result_path, task + '_VB_' + tissue + '_0.3_8', 'fa')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_acc_fa_voxel.append(balanced_accuracy)

            ## get MD raw
            for task in tasks_imbalanced:
                for tissue in tissue_combinations:
                    tsvs_path = os.path.join(classification_result_path, task + '_VB_' + tissue + '_0.3_8', 'md')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_acc_md_voxel.append(balanced_accuracy)

            ## get FA balanced
            for task in tasks_balanced:
                for tissue in tissue_combinations:
                    balanced_accuracy = []
                    tsvs_path = os.path.join(classification_result_path, 'RandomBalanced',
                                             task + '_VB_' + tissue + '_0.3_8', 'fa')
                    for k in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_acc_fa_voxel.append(balanced_accuracy)

            ## get MD balanced
            for task in tasks_balanced:
                for tissue in tissue_combinations:
                    balanced_accuracy = []
                    tsvs_path = os.path.join(classification_result_path, 'RandomBalanced',
                                             task + '_VB_' + tissue + '_0.3_8', 'md')
                    for k in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(k), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_acc_md_voxel.append(balanced_accuracy)

            ##### FAs
            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_acc_fa_voxel).transpose()
            metric = metric[:, [0, 3, 1, 4, 2, 5]]

            ## define the violin's postions
            # pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            pos = [1, 2, 4, 5, 7, 8]
            # color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
            color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
            legend = ['GM+WM_raw', 'GM+WM_balanced']

            ## define the size of th image
            fig, ax = plt.subplots(2, figsize=[15, 10])
            line_coll = ax[0].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[0].grid(axis='y', which='major', linestyle='dotted')
            ax[0].set_xticks([1.5, 4.5, 7.5])
            ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[0].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
            ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric, 0)
            std = np.std(metric, 0)
            inds = np.array(pos)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].set_ylim(0.1, 1)
            ax[0].set_title('A: FA Voxel-based classifications', fontsize=15)

            ##### MD
            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_acc_md_voxel).transpose()
            metric = metric[:, [0, 3, 1, 4, 2, 5]]
            ## define the size of th image
            line_coll = ax[1].violinplot(metric, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[1].grid(axis='y', which='major', linestyle='dotted')
            ax[1].set_xticks([1.5, 4.5, 7.5])
            ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[1].set_xticklabels(ticklabels_balanced, rotation=0, fontsize=15)  # 'vertical'
            ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric, 0)
            std = np.std(metric, 0)
            inds = np.array(pos)
            ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[1].set_ylim(0.1, 1)
            ax[1].set_title('B: MD Voxel-based classifications', fontsize=15)

            plt.savefig(os.path.join(classification_result_path,
                                     'figure3_AB.png'), additional_artists=plt.legend,
                        bbox_inches="tight")
        print 'finish Figure 3'


    elif figure_number == 4:
        results_balanced_acc_voxel_imbalanced = []
        results_balanced_acc_regional_imbalanced = []
        tissue_combinations = ['GM_WM']
        ticklabels_imbalanced = [i.replace('_', ' ') for i in tasks_imbalanced]

        if raw_classification == True:
            print "Plot for original classification, to compare T1 with DWI, we use GM+WM"
            ## T1
            for task in tasks_imbalanced:
                tsvs_path = os.path.join(classification_result_path, task + '_VB_T1_fwhm_8')
                balanced_accuracy = []
                for i in xrange(n_iterations):
                    result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                    if os.path.isfile(result_tsv):
                        balanced_accuracy.append(
                            (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                    else:
                        raise OSError(
                            errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)
            ## GM+WM FA
            for task in tasks_imbalanced:
                for tissue in tissue_combinations:
                    tsvs_path = os.path.join(classification_result_path,
                                             task + '_VB_' + tissue + '_0.3_8', 'fa')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)

            ## GM+WM MD
            for task in tasks_imbalanced:
                for tissue in tissue_combinations:
                    tsvs_path = os.path.join(classification_result_path,
                                             task + '_VB_' + tissue + '_0.3_8', 'md')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_voxel_imbalanced.append(balanced_accuracy)

            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_balanced_acc_voxel_imbalanced).transpose()
            ## reorder the order of the column to make sure the right order in the image
            metric_new = metric[:, [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]]

            ## define the violin's postions
            pos = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
            color = ['#FF0000', '#87CEFA', '#90EE90'] * len(tasks_imbalanced)  # red, blue and green
            legend = ['GM-T1w', 'GM+WM-FA', 'GM+WM-MD']

            ## define the size of th image
            fig, ax = plt.subplots(2, figsize=[15, 10])
            line_coll = ax[0].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[0].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[0].grid(axis='y', which='major', linestyle='dotted')
            ax[0].set_xticks([2, 6, 10, 14])
            ax[0].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[0].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
            ax[0].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric_new, 0)
            std = np.std(metric_new, 0)
            inds = np.array(pos)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[0].set_ylim(0.1, 1)
            ax[0].set_title('A: Voxel-based classifications for T1w and diffusion MRI', fontsize=15)


            ### T1 atlaes
            atlases_T1 = ['AAL2']
            for task in tasks_imbalanced:
                for atlas in atlases_T1:
                    tsvs_path = os.path.join(classification_result_path, task + '_RB_T1_' + atlas)
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_regional_imbalanced.append(balanced_accuracy)

            atlases_DTI = ['JHUDTI81']
            ## get DTI atlases FA
            for task in tasks_imbalanced:
                for atlas in atlases_DTI:
                    tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'fa')
                    balanced_accuracy = []
                    for i in xrange(n_iterations):
                        result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
                        if os.path.isfile(result_tsv):
                            balanced_accuracy.append(
                                (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
                        else:
                            raise OSError(
                                errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
                    results_balanced_acc_regional_imbalanced.append(balanced_accuracy)
            #
            # ## get DTI atlases MD
            # for task in tasks_imbalanced:
            #     for atlas in atlases_DTI:
            #         tsvs_path = os.path.join(classification_result_path, task + '_RB_' + atlas, 'md')
            #         balanced_accuracy = []
            #         for i in xrange(n_iterations):
            #             result_tsv = os.path.join(tsvs_path, 'iteration-' + str(i), 'results.tsv')
            #             if os.path.isfile(result_tsv):
            #                 balanced_accuracy.append(
            #                     (pd.io.parsers.read_csv(result_tsv, sep='\t')).balanced_accuracy[0])
            #             else:
            #                 raise OSError(
            #                     errno.ENOENT, os.strerror(errno.ENOENT), result_tsv)
            #         results_balanced_acc_regional_imbalanced.append(balanced_accuracy)

            ##### FAs
            ### transfer the list into an array with this shape: n_iterations*n_tasks_imbalanced
            metric = np.array(results_balanced_acc_regional_imbalanced).transpose()
            ## reorder the order of the column to make sure the right order in the image
            metric_new = metric[:, [0, 4, 1, 5, 2, 6, 3, 7]]

            ## define the violin's postions
            pos = [1, 2, 4, 5, 7, 8, 10, 11]
            color = ['#FF0000', '#87CEFA'] * len(tasks_imbalanced)  # red, blue and green
            legend = ['AAL2-T1w', 'JHULabel-FA']

            ## define the size of th image
            line_coll = ax[1].violinplot(metric_new, pos, widths=0.5, bw_method=0.2, showmeans=True, showextrema=False)
            for cc, ln in enumerate(line_coll['bodies']):
                ln.set_facecolor(color[cc])

            ax[1].legend(legend, loc='upper right', fontsize=10, frameon=True)
            ax[1].grid(axis='y', which='major', linestyle='dotted')
            ax[1].set_xticks([1.5, 4.5, 7.5, 10.5])
            ax[1].set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax[1].set_xticklabels(ticklabels_imbalanced, rotation=0, fontsize=15)  # 'vertical'
            ax[1].set_ylabel('Balanced accuracy', rotation=90, fontsize=15)  # 'vertical'
            mean = np.mean(metric_new, 0)
            std = np.std(metric_new, 0)
            inds = np.array(pos)
            ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[1].vlines(inds, mean - std, mean + std, color='k', linestyle='solid', lw=0.5)
            ax[1].hlines(mean - std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[1].hlines(mean + std, inds - 0.1, inds + 0.1, color='k', linestyle='solid', lw=0.5)
            ax[1].set_ylim(0.1, 1)
            ax[1].set_title('B: Region-based classifications for T1w and diffusion MRI', fontsize=15)
            plt.savefig(os.path.join(classification_result_path,
                                     'violin_T1_compare_dwi.png'), additional_artists=plt.legend,
                        bbox_inches="tight")
        #
        print 'finish T1'



def random_donsample_subjects(diagnoses_tsv, n=None):
    """
    This function is to randomly downsample the subjects.
    :param diagnoses_tsv:
    :return:
    """
    from collections import Counter
    import os

    diagnoses = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')
    print 'Do random subsampling the majority group to number of subjects of minority group:'
    counts = Counter(list(diagnoses.diagnosis))
    label1 = counts.keys()[0]
    label2 = counts.keys()[1]
    count_label1 = counts[label1]
    count_label2 = counts[label2]
    if count_label1 < count_label2:
        print '%s is the majority group and will be randomly downsampled.' % label2
        majority_df_index = diagnoses.index[diagnoses['diagnosis'] == label2]
        drop_index = np.random.choice(majority_df_index, count_label2 - count_label1, replace= False)
        diagnoses_balanced = diagnoses.drop(drop_index)

    elif count_label1 > count_label2:
        print '%s is the majority group and will be randomly downsampled.' % label1
        majority_df_index = diagnoses.index[diagnoses['diagnosis'] == label1]
        drop_index = np.random.choice(majority_df_index, count_label1 - count_label2, replace= False)
        diagnoses_balanced = diagnoses.drop(drop_index)
    else:
        raise Exception("""The data is balanced already, please deactivate the balanced_down_sample flag""")
    # save the balanced tsv
    if n == None:
        if os.path.isfile(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_balanced.tsv')):
            pass
        else:
            diagnoses_balanced.to_csv(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_balanced.tsv'), sep='\t', index=False)
    else:
        if os.path.isfile(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_' + str(n) + '_balanced.tsv')):
            pass
        else:
            diagnoses_balanced.to_csv(os.path.join(os.path.dirname(diagnoses_tsv), os.path.basename(diagnoses_tsv).split('.')[0] + '_' + str(n) + '_balanced.tsv'), sep='\t', index=False)

    list_diagnoses = list(diagnoses_balanced.diagnosis)
    list_subjects = list(diagnoses_balanced.participant_id)
    list_sessions = list(diagnoses_balanced.session_id)

    return list_subjects, list_sessions, list_diagnoses

def split_subjects_to_pickle(diagnoses_tsv, n_iterations=250, test_size=0.2, balanced = False):

    from os import path
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    import pickle
    from collections import Counter

    diagnoses = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')
    if 'diagnosis' not in list(diagnoses.columns.values):
        raise Exception('Diagnoses file is not in the correct format.')
    diagnoses_list = list(diagnoses.diagnosis)
    unique = list(set(diagnoses_list))
    y = np.array([unique.index(x) for x in diagnoses_list])

    if balanced == False:
        splits_indices_pickle = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '.pkl')
    else:
        splits_indices_pickle = path.join(path.dirname(diagnoses_tsv), path.basename(diagnoses_tsv).split('.')[0] + '_balanced.pkl')

    ## try to see if the shuffle has been done
    if os.path.isfile(splits_indices_pickle):
        splits_indices = pickle.load(open(splits_indices_pickle, 'rb'))
    else:
        splits = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size)
        if balanced == False:
            splits_indices = list(splits.split(np.zeros(len(y)), y))
        else:
            print 'Do random subsampling the majority group to number of subjects of minority group:'
            splits_indices = []
            n_iteration = 0
            for train_index, test_index in splits.split(np.zeros(len(y)), y):

                # for training
                train_label1 = []
                train_label2 = []
                counts = Counter(diagnoses.diagnosis[train_index])
                label1 = counts.keys()[0]
                label2 = counts.keys()[1]
                count_label1 = counts[label1]
                count_label2 = counts[label2]
                for i  in train_index:
                    if diagnoses.diagnosis[i] == label2:
                        train_label2.append(i)
                    else:
                        train_label1.append(i)
                if count_label1 < count_label2:
                    print 'In training data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label2)
                    drop_index_train = np.random.choice(train_label2, count_label2 - count_label1, replace=False)
                    train_index_balanced = np.asarray([item for item in train_index if item not in drop_index_train])
                elif count_label1 > count_label2:
                    print 'In training data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label1)
                    drop_index_train = np.random.choice(train_label1, count_label1 - count_label2, replace=False)
                    train_index_balanced = np.asarray([item for item in train_index if item not in drop_index_train])
                else:
                    raise Exception("""The data is balanced already, please deactivate the balanced_down_sample flag""")

                # for test
                test_label1 = []
                test_label2 = []
                counts = Counter(diagnoses.diagnosis[test_index])
                label1 = counts.keys()[0]
                label2 = counts.keys()[1]
                count_label1 = counts[label1]
                count_label2 = counts[label2]
                for i  in test_index:
                    if diagnoses.diagnosis[i] == label2:
                        test_label2.append(i)
                    else:
                        test_label1.append(i)
                if count_label1 < count_label2:
                    print 'In test data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label2)
                    drop_index_test= np.random.choice(test_label2, count_label2 - count_label1, replace=False)
                    test_index_balanced = np.asarray([item for item in test_index if item not in drop_index_test])
                elif count_label1 > count_label2:
                    print 'In test data for iteration %d, %s is the majority group and will be randomly downsampled.' % (n_iteration, label1)
                    drop_index_test = np.random.choice(test_label1, count_label1 - count_label2, replace=False)
                    test_index_balanced = np.asarray([item for item in test_index if item not in drop_index_test])
                else:
                    raise Exception("""The data is balanced already, please deactivate the balanced_down_sample flag""")
                ##
                n_iteration += 1
                splits_indices.append((train_index_balanced, test_index_balanced))
                ## save each iteration as tsv files
                diagnoses_balanced_tsv = diagnoses.drop(np.append(drop_index_train, drop_index_test))
                diagnoses_balanced_tsv.to_csv(os.path.join(os.path.dirname(diagnoses_tsv),
                                                       os.path.basename(diagnoses_tsv).split('.')[0] + '_' + str(
                                                           n_iteration) + '_balanced.tsv'), sep='\t', index=False)
    with open(splits_indices_pickle, 'wb') as s:
        pickle.dump(splits_indices, s)

    return splits_indices, splits_indices_pickle

def compute_t(subjects_1_tsv, subjects_2_tsv, test_size=0.2):
    """
    This is a function to compute the corrected resampled paired t-test based on the paper of Nadeau and Bengio 2003.
    Also please refer this post to understand the different metrics used to compare two classifiers (https://stats.stackexchange.com/questions/217466/for-model-selection-comparison-what-kind-of-test-should-i-use)

    Also, please refer the package mlxtend.evaluate, but they did not include the corrected resampled paired t-test

    :param subjects_1_tsv:
    :param subjects_2_tsv:
    :param test_size:
    :return:
    """

    subjects_1 = pd.io.parsers.read_csv(subjects_1_tsv, sep='\t')
    subjects_2 = pd.io.parsers.read_csv(subjects_2_tsv, sep='\t')

    num_split = len(subjects_1.iteration.unique())
    n_subj = subjects_1.shape[0] / num_split

    test_error_split = np.zeros((num_split, 1))  # this list will contain the list of mu_j hat for j = 1 to J

    q1 = (subjects_1.y == subjects_1.y_hat) * 1.0
    q2 = (subjects_2.y == subjects_2.y_hat) * 1.0
    l = q1 - q2

    for i in range(num_split):
        test_error_split[i] = np.mean(l[(i * n_subj):((i + 1) * n_subj)])

    # compute mu_{n_1}^{n_2}
    average_test_error = np.mean(test_error_split)

    # compute S2_{mu_J}
    approx_variance = np.sum((test_error_split - average_test_error) ** 2)

    resampled_t = average_test_error * np.sqrt(num_split) / np.sqrt(approx_variance / (num_split - 1))

    resampled_p_value = stats.t.sf(np.abs(resampled_t), num_split - 1) * 2.

    corrected_resampled_t = average_test_error * np.sqrt(num_split) / np.sqrt((test_size / (1 - test_size) + 1/(num_split - 1)) * approx_variance)

    corrected_resampled_p_value = stats.t.sf(np.abs(corrected_resampled_t), num_split - 1) * 2.

    return resampled_t, resampled_p_value, corrected_resampled_t, corrected_resampled_p_value

