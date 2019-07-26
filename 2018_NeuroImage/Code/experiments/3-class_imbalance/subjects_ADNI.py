
from os import path
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def sample_diagnoses(diagnoses_tsv, subjects_visits_tsv_out, diagnoses_tsv_out):

    dx = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')

    dx1 = dx[dx.diagnosis == dx.diagnosis.unique()[0]].sample(dx.diagnosis.value_counts().min())
    dx2 = dx[dx.diagnosis == dx.diagnosis.unique()[1]].sample(dx.diagnosis.value_counts().min())

    merged = dx1.append(dx2)

    merged.to_csv(diagnoses_tsv_out, index=False, sep='\t', encoding='utf-8')
    merged[['participant_id', 'session_id']].to_csv(subjects_visits_tsv_out, index=False, sep='\t', encoding='utf-8')


test_size = 0.3
n_iterations = 250

splitter = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size)

tasks_dir = '/ADNI/SUBJECTS/lists_by_task'
tasks = [('CN', 'AD'),
         ('CN', 'pMCI'),
         ('sMCI', 'pMCI')]

for task in tasks:
    original_diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))

    subjects_tsv = path.join(tasks_dir, '%s_vs_%s_subjects_sessions_balanced.tsv' % (task[0], task[1]))
    diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses_balanced.tsv' % (task[0], task[1]))

    sample_diagnoses(original_diagnoses_tsv, subjects_tsv, diagnoses_tsv)

    dx = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')

    unique = list(set(dx.diagnosis))
    y = np.array([unique.index(x) for x in dx.diagnosis])
    splits_indices = list(splitter.split(np.zeros(len(y)), y))

    with open(path.join(tasks_dir, '%s_vs_%s_splits_indices_balanced.pkl' % (task[0], task[1])), 'wb') as s:
        pickle.dump(splits_indices, s)
