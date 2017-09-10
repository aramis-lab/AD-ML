
import os
from os import path
import pandas as pd
import clinica.pipeline.machine_learning.ml_workflows as ml_wf

n_iterations = 10
n_threads = 72

caps_dir = '/ADNI/CAPS'
output_dir = '/ADNI/CLASSIFICATION/OUTPUTS'

group_id = 'ADNIbl'
image_types = ['T1', 'fdg']
tasks_dir = '/ADNI/lists_by_task'
tasks = [('CN', 'AD'),
         ('CN', 'MCI'),
         ('CN', 'pMCI'),
         ('sMCI', 'pMCI'),
         ('CN-', 'AD+'),
         ('CN-', 'MCI+'),
         ('CN-', 'pMCI+'),
         ('MCI-', 'MCI+'),
         ('sMCI+', 'pMCI+')]
smoothing = [0, 4, 8, 12]
atlases = ['AAL2', 'LPBA40', 'Neuromorphometrics', 'AICHA', 'Hammers']
pvcs = [None, 'rbv']

##### Voxel based classifications ######

for image_type in image_types:
    for fwhm in smoothing:

        for task in tasks:
            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

            for pvc in pvcs:
                if image_type == 'T1':
                    if pvc is None:
                        classification_dir = path.join(output_dir, image_type, 'voxel_based',
                                                       'smooothing-%s' % fwhm, 'linear_svm',
                                                       '%s_vs_%s' % (task[0], task[1]))
                    else:
                        continue
                else:
                    classification_dir = path.join(output_dir, image_type, 'voxel_based', 'pvc-%s' % pvc,
                                                   'smooothing-%s' % fwhm, 'linear_svm',
                                                   '%s_vs_%s' % (task[0], task[1]))
                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                print "Running %s" % classification_dir
                wf = ml_wf.VB_RepKFold_DualSVM(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id, image_type,
                                               classification_dir, fwhm=fwhm, pvc=pvc, n_iterations=n_iterations,
                                               n_threads=n_threads)
                wf.run()
                # wf.save_image()
