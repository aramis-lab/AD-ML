
import os
from os import path

import pandas as pd
import numpy as np
import nibabel as nib

from clinica.pipelines.machine_learning.input import CAPSVoxelBasedInput
import clinica.pipelines.machine_learning.svm_utils as utils
import clinica.pipelines.machine_learning.voxel_based_io as vbio
from sklearn.metrics import roc_auc_score


caps_dir = '/AIBL/CAPS'
adni_caps_dir = '/ADNI/CAPS'

adni_output_dir = '/ADNI/CLASSIFICATION/OUTPUTS'
output_dir = '/AIBL/CLASSIFICATION/OUTPUTS'

group_id = 'ADNIbl'
image_types = ['T1']

tasks_dir = '/AIBL/SUBJECTS/lists_by_task'
adni_tasks_dir = '/ADNI/SUBJECTS/lists_by_task'
tasks = [('CN', 'AD')]

smoothing = [4]
pvcs = [None]

##### Voxel based classifications ######

for image_type in image_types:
    for fwhm in smoothing:

        for task in tasks:
            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

            adni_subjects_visits_tsv = path.join(adni_tasks_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
            adni_diagnoses_tsv = path.join(adni_tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

            for pvc in pvcs:
                if image_type == 'T1':
                    if pvc is None:
                        classification_dir = path.join(output_dir, image_type, 'voxel_based',
                                                       'smooothing-%s' % fwhm, 'linear_svm',
                                                       '%s_vs_%s' % (task[0], task[1]))
                        adni_classifier_dir = path.join(adni_output_dir, image_type, 'voxel_based',
                                                            'smooothing-%s' % fwhm, 'linear_svm',
                                                            '%s_vs_%s' % (task[0], task[1]), 'classifier')
                    else:
                        continue
                else:
                    classification_dir = path.join(output_dir, image_type, 'voxel_based', 'pvc-%s' % pvc,
                                                   'smooothing-%s' % fwhm, 'linear_svm',
                                                   '%s_vs_%s' % (task[0], task[1]))
                    adni_classifier_dir = path.join(adni_output_dir, image_type, 'voxel_based', 'pvc-%s' % pvc,
                                                        'smooothing-%s' % fwhm, 'linear_svm',
                                                        '%s_vs_%s' % (task[0], task[1]), 'classifier')
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                print "Running %s" % classification_dir

                adni_images = CAPSVoxelBasedInput(adni_caps_dir, adni_subjects_visits_tsv, adni_diagnoses_tsv, group_id,
                                                  image_type, fwhm, modulated='on', pvc=pvc, mask_zeros=False)

                input_images = CAPSVoxelBasedInput(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id,
                                                   image_type, fwhm, modulated='on', pvc=pvc, mask_zeros=False)

                adni_x, adni_orig_shape, adni_data_mask = vbio.load_data(adni_images.get_images(), mask=True)

                weights = np.loadtxt(path.join(adni_classifier_dir, 'weights.txt'))
                w = vbio.revert_mask(weights, adni_data_mask, adni_orig_shape).flatten()

                b = np.loadtxt(path.join(adni_classifier_dir, 'intersect.txt'))

                x = input_images.get_x()
                y = input_images.get_y()

                y_hat = np.dot(w, x.transpose()) + b

                y_binary = (y_hat > 0) * 1.0

                evaluation = utils.evaluate_prediction(y, y_binary)

                auc = roc_auc_score(y, y_hat)
                evaluation['AUC'] = auc

                print evaluation

                del evaluation['confusion_matrix']

                res_df = pd.DataFrame(evaluation, index=['i', ])
                res_df.to_csv(path.join(classification_dir, 'results_auc.tsv'), sep='\t')
