from os import path, environ

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from workflow import TsvInput


adni_output_dir = path.join(environ.get('OUT_PATH'), 'ADNI/OUTPUT')
spie_output_dir = path.join(adni_output_dir, 'SPIE')

input_data_path = path.join(spie_output_dir, 'input_data')
tasks_data_path = path.join(spie_output_dir, 'tasks_lists')

subj_sessions = pd.io.parsers.read_csv(path.join(input_data_path, 'input_data.tsv'), sep='\t')

subj_sessions.loc[subj_sessions[subj_sessions.sex == 'F'].index, 'sex'] = 1
subj_sessions.loc[subj_sessions[subj_sessions.sex == 'M'].index, 'sex'] = 0


##################################################
### Table 1 - Clinical data and Imaging scores ###
##################################################

model_36 = subj_sessions[~subj_sessions.adas13.isnull()]
model_36.to_csv(path.join(input_data_path, 'table_1_clinical_imaging', 'input_data_model_36.tsv'), sep='\t', index=False)


####################################################
### Table 2 - Clinical data and ADNIMERGE scores ###
####################################################

model_36 = subj_sessions[~subj_sessions.adas13.isnull() &
                         ~subj_sessions.adni_ventricles_vol.isnull() &
                         ~subj_sessions.adni_hippocampus_vol.isnull() &
                         ~subj_sessions.adni_brain_vol.isnull() &
                         ~subj_sessions.adni_entorhinal_vol.isnull() &
                         ~subj_sessions.adni_icv.isnull() &
                         ~subj_sessions.adni_fusiform_vol.isnull() &
                         ~subj_sessions.adni_midtemp_vol.isnull()]

model_36["adni_ventricles_vol_icv"] = model_36.adni_ventricles_vol / model_36.adni_icv
model_36["adni_hippocampus_vol_icv"] = model_36.adni_hippocampus_vol / model_36.adni_icv
model_36["adni_brain_vol_icv"] = model_36.adni_brain_vol / model_36.adni_icv
model_36["adni_entorhinal_vol_icv"] = model_36.adni_entorhinal_vol / model_36.adni_icv
model_36["adni_fusiform_vol_icv"] = model_36.adni_fusiform_vol / model_36.adni_icv
model_36["adni_midtemp_vol_icv"] = model_36.adni_midtemp_vol / model_36.adni_icv

model_36.to_csv(path.join(input_data_path, 'table_2_clinical_adnimerge', 'input_data_model_36.tsv'), sep='\t', index=False)


######################################################
### Table 3 - Clinical data and Amyloid deposition ###
######################################################

model_36 = subj_sessions[~subj_sessions.adas13.isnull() &
                         (~subj_sessions.adni_pib.isnull() |
                          ~subj_sessions.adni_av45.isnull())]

model_36["amyloid"] = ((model_36.adni_pib >= 1.47) * 1.0) + ((model_36.adni_av45 >= 1.1) * 1.0)

model_36.to_csv(path.join(input_data_path, 'table_3_amyloid', 'input_data_model_36.tsv'), sep='\t', index=False)


####################################
### Table 4 - Several timepoints ###
####################################

model_36 = subj_sessions[~subj_sessions.adas13.isnull()]

for i in range(1, 7):

    model = []
    diagnoses = pd.io.parsers.read_csv(path.join(tasks_data_path, 'sMCI_vs_pMCI_diagnoses_%s.tsv' % (i * 6)), sep='\t')

    for row in diagnoses.iterrows():
        subj = row[1]
        data_subj = model_36[model_36.participant_id == subj.participant_id]
        if data_subj.shape[0] == 1:
            data_subj.diagnosis = 0
            data_subj.diagnosis = subj.diagnosis
            model.append(data_subj)

    model_t = pd.concat(model)
    model_t.to_csv(path.join(input_data_path, 'table_4_timepoints', 'input_data_model_%s.tsv' % (i * 6)), sep='\t', index=False)


########################################
### Generating splits for each table ###
########################################

n_iterations = 250
test_size = 0.2

tables_timepoints = {'table_1_clinical_imaging': [36],
                     'table_2_clinical_adnimerge': [36],
                     'table_3_amyloid': [36],
                     'table_4_timepoints': [6, 12, 18, 24, 30, 36]}

for table in tables_timepoints:
    for timepoint in tables_timepoints[table]:

        data_tsv = path.join(input_data_path, table, 'input_data_model_%s.tsv' % timepoint)
        input_data = TsvInput(data_tsv)
        y = input_data.get_y()

        splits = StratifiedShuffleSplit(n_splits=n_iterations, test_size=test_size)
        splits_indices = list(splits.split(np.zeros(len(y)), y))

        with open(path.join(input_data_path, table, 'indices_model_%s.tsv' % timepoint), 'wb') as s:
            pickle.dump(splits_indices, s)
