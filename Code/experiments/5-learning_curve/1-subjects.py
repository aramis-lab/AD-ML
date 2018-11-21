
from os import path
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

test_size = 0.3
n_iterations = 250
n_learning_points = 10

splitter = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size)
skf = StratifiedKFold(n_splits=n_learning_points, shuffle=False)

### ADNI ###
adni_tasks_dir = '/ADNI/SUBJECTS/lists_by_task'

tasks = [('CN', 'AD'),
         ('CN', 'pMCI'),
         ('sMCI', 'pMCI')]

for task in tasks:

    adni_dx = pd.io.parsers.read_csv(path.join(adni_tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1])), sep='\t')
    adni_dx['caps'] = "ADNI"
    adni_dx['cv_dx'] = adni_dx.diagnosis + '_ADNI'

    output_dir = '/DATASETS/SUBJECTS/LEARNING_CURVE_ADNI'
    adni_dx.to_csv(path.join(output_dir, '%s_vs_%s_diagnoses_caps.tsv' % (task[0], task[1])), index=False, sep='\t', encoding='utf-8')

    unique = list(set(merged.cv_dx))
    y = np.array([unique.index(x) for x in merged.cv_dx])
    outer_splits = list(splitter.split(np.zeros(len(y)), y))
    inner_splits = []

    for i in range(n_iterations):
        train_index, _ = outer_splits[i]
        inner_cv = list(skf.split(np.zeros(len(y[train_index])), y[train_index]))
        inner_splits.append(inner_cv)

    splits_indices = {'outer': outer_splits,
                      'inner': inner_splits}

    with open(path.join(output_dir, '%s_vs_%s_splitsIndices_learningCurve.pkl' % (task[0], task[1])), 'wb') as s:
        pickle.dump(splits_indices, s)
