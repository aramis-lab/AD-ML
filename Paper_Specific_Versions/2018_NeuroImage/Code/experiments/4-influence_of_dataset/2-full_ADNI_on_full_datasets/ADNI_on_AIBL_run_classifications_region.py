
import os
from os import path

import pandas as pd
import numpy as np
import nibabel as nib

from clinica.pipelines.machine_learning.input import CAPSRegionBasedInput
import clinica.pipelines.machine_learning.svm_utils as utils
import clinica.pipelines.machine_learning.region_based_io as rbio
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

atlases = ['AAL2']

##### Region based classifications ######

for image_type in image_types:
    for atlas in atlases:
        for task in tasks:
            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

            adni_subjects_visits_tsv = path.join(adni_tasks_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
            adni_diagnoses_tsv = path.join(adni_tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

            classification_dir = path.join(output_dir, image_type, 'region_based',
                                           'atlas-%s' % atlas, 'linear_svm',
                                           '%s_vs_%s' % (task[0], task[1]))

            adni_classifier_dir = path.join(adni_output_dir, image_type, 'region_based',
                                            'atlas-%s' % atlas, 'linear_svm',
                                            '%s_vs_%s' % (task[0], task[1]), 'classifier')

            if not path.exists(classification_dir):
                os.makedirs(classification_dir)

            print "Running %s" % classification_dir

            input_images = CAPSRegionBasedInput(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id,
                                                image_type, atlas)

            w = np.loadtxt(path.join(adni_classifier_dir, 'weights.txt'))

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
            res_df.to_csv(path.join(classification_dir, 'results_auc.tsv'), sep='\t', index=False)
