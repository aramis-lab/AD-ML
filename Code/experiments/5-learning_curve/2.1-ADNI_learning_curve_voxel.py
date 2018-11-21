
import os
from os import path
import pickle
from workflow import MCVB_LearningCurveRepHoldOut_DualSVM

n_iterations = 250
n_threads = 72
# These scritps assumes that the folder structure is /DATASETS/CAPS/Dataset1, /DATASETS/CAPS/Dataset2, etc.
caps_dir = '/DATASETS/CAPS'
output_dir = '/DATASETS/OUTPUTS/LEARNING_CURVE_ADNI'

group_id = 'ADNIbl'
image_types = ['T1', 'fdg']
tasks_dir = '/DATASETS/SUBJECTS/LEARNING_CURVE_ADNI'
tasks = [('CN', 'AD')]

smoothing = [4]
atlases = ['AAL2']

##### Voxel based classifications ######

for image_type in image_types:
    for fwhm in smoothing:

        for task in tasks:
            diagnoses_caps_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses_caps.tsv' % (task[0], task[1]))

            classification_dir = path.join(output_dir, image_type, 'voxel_based',
                                           'smooothing-%s' % fwhm, 'linear_svm',
                                           '%s_vs_%s' % (task[0], task[1]))

            with open(path.join(tasks_dir, '%s_vs_%s_splitsIndices_learningCurve.pkl' % (task[0], task[1])), 'r') as s:
                splits_indices = pickle.load(s)

            if not path.exists(classification_dir):
                os.makedirs(classification_dir)

            print "Running %s" % classification_dir
            wf = MCVB_LearningCurveRepHoldOut_DualSVM(caps_dir, diagnoses_caps_tsv, group_id, image_type,
                                                      classification_dir, fwhm=fwhm, pvc=None,
                                                      n_iterations=n_iterations, n_threads=n_threads,
                                                      splits_indices=splits_indices)
            wf.run()
