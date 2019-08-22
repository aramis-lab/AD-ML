import os
from os import path

import pandas as pd
import numpy as np

import clinica.pipelines.machine_learning.svm_utils as utils
import workflow


# Paths to personalize
adni_tsv_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/TSV')
adni_caps_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/CAPS')
adni_output_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/OUTPUT')
spie_output_dir = path.join(adni_output_dir, 'SPIE')
scores_dir = path.join(spie_output_dir, 'input_data', 'svm_scores')

if not path.exists(scores_dir):
            os.makedirs(scores_dir)

group_id = 'ADNIbl'
image_types = ['T1', 'fdg']

tasks = [('sMCI', 'pMCI')]

##### Voxel based classifications ######

for task in tasks:
    subjects_visits_tsv = path.join(adni_tsv_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
    diagnoses_tsv = path.join(adni_tsv_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

    subjects_visits = pd.io.parsers.read_csv(subjects_visits_tsv, sep='\t')
    diagnoses_df = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')

    for image_type in image_types:
        svm_output_dir = path.join(adni_output_dir, image_type, 'voxel_based/linear_svm/CN-_vs_AD+/classifier')
        input_images = workflow.VBAgeCorrectedInput(adni_caps_dir, subjects_visits_tsv, diagnoses_tsv, image_type, None)

        x = input_images.get_x()
        y = input_images.get_y()

        w = np.loadtxt(path.join(svm_output_dir, 'weights.txt'))
        b = np.loadtxt(path.join(svm_output_dir, 'intersect.txt'))

        y_hat = np.dot(w, x.transpose()) + b
        y_binary = (y_hat > 0) * 1.0
        correct = (y == y_binary) * 1.0

        print(utils.evaluate_prediction(y_binary, y))

        results_df = pd.DataFrame({'participant_id': subjects_visits.participant_id.tolist(),
                                   'session_id': subjects_visits.session_id.tolist(),
                                   'diagnosis': diagnoses_df.diagnosis.tolist(),
                                   'y': y.tolist(),
                                   'y_binary': y_binary.tolist(),
                                   'y_hat': y_hat,
                                   'correct': correct})

        results_df.to_csv(path.join(spie_output_dir, image_type + '.tsv'),
                          index=False, sep='\t', encoding='utf-8')
