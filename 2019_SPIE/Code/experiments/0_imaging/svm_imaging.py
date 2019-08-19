import os
from os import path
import clinica.pipeline.machine_learning.ml_workflows as ml_wf

# Paths to personalize
adni_tsv_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/TSV')
adni_caps_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/CAPS')
adni_output_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/OUTPUT')


n_iterations = 250
n_threads = 8

group_id = 'ADNIbl'
tasks = [('sMCI', 'pMCI'),
         ('CN-', 'AD+')]

image_types = ['T1', 'fdg']
fwhm = 4
pvc = None


###################################
### Voxel based classifications ###
###################################

for task in tasks:
    subjects_visits_tsv = path.join(adni_tsv_dir, '%s_vs_%s_subjects_sessions.tsv' % (task[0], task[1]))
    diagnoses_tsv = path.join(adni_tsv_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

    for image_type in image_types:

        classification_dir = path.join(adni_output_dir, image_type, 'voxel_based', 'linear_svm',
                                       '%s_vs_%s' % (task[0], task[1]))
        if not path.exists(classification_dir):
            os.makedirs(classification_dir)

        print("Running %s" % classification_dir)

        wf = ml_wf.VB_RepHoldOut_DualSVM(adni_caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id, image_type,
                                         classification_dir, fwhm=fwhm, pvc=pvc, n_iterations=n_iterations,
                                         n_threads=n_threads)
        wf.run()
