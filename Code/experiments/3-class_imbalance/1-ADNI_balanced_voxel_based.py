
import os
from os import path
import pickle
from clinica.pipelines.machine_learning.ml_workflows import VB_RepHoldOut_DualSVM

n_iterations = 250
n_threads = 72

caps_dir = '/ADNI/CAPS'
output_dir = '/ADNI/CLASSIFICATION/OUTPUTS_BALANCED'

group_id = 'ADNIbl'
image_types = ['T1', 'fdg']
tasks_dir = '/ADNI/SUBJECTS/lists_by_task'  
tasks = [('CN', 'AD'),
         ('CN', 'pMCI'),
         ('sMCI', 'pMCI')]
smoothing = [0, 4, 8, 12]
classifier = 'linear_svm'

##### Voxel based classifications ######

for image_type in image_types:
    for smooth in smoothing:
        for task in tasks:
            if image_type == 'T1':
                    classification_dir = path.join(output_dir, image_type, 'voxel_based',
                                                   'smoothing-%s' % smooth, classifier,
                                                   '%s_vs_%s' % (task[0], task[1]))
            else:
                classification_dir = path.join(output_dir, image_type, 'voxel_based', 'pvc-None',
                                               'smoothing-%s' % smooth, classifier,
                                               '%s_vs_%s' % (task[0], task[1]))

            if not path.exists(classification_dir):
                os.makedirs(classification_dir)

            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions_balanced.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses_balanced.tsv' % (task[0], task[1]))
            with open(path.join(tasks_dir, '%s_vs_%s_splits_indices_balanced.pkl' % (task[0], task[1])), 'r') as s:
                splits_indices = pickle.load(s)

            print "Running %s" % classification_dir
            wf = VB_RepHoldOut_DualSVM(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id, image_type,
                                       classification_dir, fwhm=smooth, pvc=None, n_iterations=n_iterations,
                                       n_threads=n_threads, splits_indices=splits_indices)
            wf.run()
