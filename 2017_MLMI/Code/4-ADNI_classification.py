
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

##### Voxel based classifications ######

for image_type in image_types:

        for task in tasks:
            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

            classification_dir = path.join(output_dir, image_type, 'voxel_based', 'linear_svm',
                                               '%s_vs_%s' % (task[0], task[1]))
            if not path.exists(classification_dir):
                os.makedirs(classification_dir)

            print "Running %s" % classification_dir
            wf = ml_wf.VB_RepKFold_DualSVM(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id, image_type,
                                           classification_dir, fwhm=0, pvc=None, n_iterations=n_iterations,
                                           n_threads=n_threads)
            wf.run()
            # wf.save_image()
