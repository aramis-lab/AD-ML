
import os
from os import path
import clinica.pipelines.machine_learning.ml_workflows as ml_wf

n_iterations = 250
n_threads = 20

caps_dir = '/OASIS/CAPS'
output_dir = '/OASIS/CLASSIFICATION/OUTPUTS_GENERALIZATION'

group_id = 'ADNIbl'
image_types = ['T1']
tasks_dir = '/OASIS/SUBJECTS/lists_by_task'
tasks = [('CN', 'AD')]

smoothing = [4]
atlases = ['AAL2']

rb_classifiers = {'linear_svm': ml_wf.RB_RepHoldOut_DualSVM}

##### Voxel based classifications ######

for image_type in image_types:
    for fwhm in smoothing:

        for task in tasks:
            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions_generalization.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses_generalization.tsv' % (task[0], task[1]))

            classification_dir = path.join(output_dir, image_type, 'voxel_based',
                                           'smooothing-%s' % fwhm, 'linear_svm',
                                           '%s_vs_%s' % (task[0], task[1]))
            if not path.exists(classification_dir):
                os.makedirs(classification_dir)

            print "Running %s" % classification_dir
            wf = ml_wf.VB_RepHoldOut_DualSVM(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id, image_type,
                                             classification_dir, fwhm=fwhm, pvc=None, n_iterations=n_iterations,
                                             n_threads=n_threads)
            wf.run()

##### Region based classifications ######

for image_type in image_types:
    for atlas in atlases:
        for task in tasks:
            subjects_visits_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions_generalization.tsv' % (task[0], task[1]))
            diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses_generalization.tsv' % (task[0], task[1]))

            for classifier in rb_classifiers:
                ml_class = rb_classifiers[classifier]

                classification_dir = path.join(output_dir, image_type, 'region_based',
                                               'atlas-%s' % atlas, classifier,
                                               '%s_vs_%s' % (task[0], task[1]))

                if not path.exists(classification_dir):
                    os.makedirs(classification_dir)

                print "Running %s" % classification_dir
                wf = ml_class(caps_dir, subjects_visits_tsv, diagnoses_tsv, group_id, image_type, atlas,
                              classification_dir, pvc=None, n_iterations=n_iterations, n_threads=n_threads)
                wf.run()
