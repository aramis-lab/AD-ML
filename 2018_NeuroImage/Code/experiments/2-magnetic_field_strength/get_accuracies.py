
from os import path
import pandas as pd

from population_stats import population_stats

import clinica.pipelines.machine_learning.svm_utils as utils

tasks_dir = '/ADNI/SUBJECTS/lists_by_task'

output_dir = '/ADNI/CLASSIFICATION/OUTPUTS/T1/voxel_based/smoothing-4/linear_svm/'

# Run a second time for region based results
#output_dir = '/ADNI/CLASSIFICATION/OUTPUTS/T1/region_based/atlas-AAL2/linear_svm/'

tasks = [('CN', 'AD'),
         ('CN', 'pMCI'),
         ('sMCI', 'pMCI')]

conversion = pd.io.parsers.read_csv('/ADNI/BIDS/conversion_info/t1_paths.tsv', sep='\t')
conversion_bl = conversion[conversion.VISCODE == 'bl']

for task in tasks:
    diagnoses_tsv = path.join(tasks_dir, '%s_vs_%s_diagnoses.tsv' % (task[0], task[1]))
    dx = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')

    field = []
    for subj in dx.participant_id:
        field.append(conversion_bl[conversion_bl.Subject_ID == subj[8:].replace('S', '_S_')].Field_Strength.values[0])

    dx['Field_Strength'] = field

    dx.to_csv(path.join(tasks_dir, '%s_vs_%s_field.tsv' % (task[0], task[1])), index=False, sep='\t', encoding='utf-8')

    dx15 = dx[dx.Field_Strength == 1.5]
    dx15.to_csv(path.join(tasks_dir, '%s_vs_%s_field_1.5T.tsv' % (task[0], task[1])), index=False, sep='\t', encoding='utf-8')

    dx3 = dx[dx.Field_Strength == 3]
    dx3.to_csv(path.join(tasks_dir, '%s_vs_%s_field_3T.tsv' % (task[0], task[1])), index=False, sep='\t', encoding='utf-8')


for task in tasks:

    field = pd.io.parsers.read_csv(path.join(tasks_dir, '%s_vs_%s_field.tsv' % (task[0], task[1])), sep='\t')
    dx15 = pd.io.parsers.read_csv(path.join(tasks_dir, '%s_vs_%s_field_1.5T.tsv' % (task[0], task[1])), sep='\t')
    dx3 = pd.io.parsers.read_csv(path.join(tasks_dir, '%s_vs_%s_field_3T.tsv' % (task[0], task[1])), sep='\t')

    results = pd.io.parsers.read_csv(path.join(output_dir, '%s_vs_%s' % (task[0], task[1]), 'test_subjects.tsv'), sep='\t')

    results['Field_Strength'] = list(field.Field_Strength[results.subject_index])

    print task
    results15 = results[results.Field_Strength == 1.5]

    print 'Mean accuracy 1.5T: '
    print utils.evaluate_prediction(list(results15.y), list(results15.y_hat))

    results3 = results[results.Field_Strength == 3]
    print 'Mean accuracy 3T: '
    print utils.evaluate_prediction(list(results3.y), list(results3.y_hat))


# Population stats
print 'ADNI 1.5T vs 3T population stats'
path_bids = '/ADNI/BIDS'
tasks_dir = '/ADNI/SUBJECTS/lists_by_task'

for task in tasks:
    print task
    print 'Subjects 1.5T'
    dx15 = path.join(tasks_dir, '%s_vs_%s_field_1.5T.tsv' % (task[0], task[1]))
    population_stats(path_bids, dx15, 'ADNI')

    print 'Subjects 3T'
    dx3 = path.join(tasks_dir, '%s_vs_%s_field_3T.tsv' % (task[0], task[1]))
    population_stats(path_bids, dx3, 'ADNI')
