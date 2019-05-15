# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Simona Bottani", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

import os
import pandas as pd

def create_subjects_lists(dwi_bids, t1w_bids, output_dir, database='ADNI', label='AD'):
    """
    This is the function to extract the participant tsv files for all the experiments.
    Here, we included all the participants who are available diffusion and T1w MRI at baseline in ADNI
    :param dwi_bids: the path to the BIDS of diffusion MRI
    :param t1w_bids: the path to the BIDS of T1w MRI
    :param output_dir: the output path to contain the resulting tsv files
    :param database: the name of the database, by default is ADNI
    :param label: the name of the database, by default is ADNI
    :return:
    """
    # path to the  bids folder
    # read the participants.tsv file in ADNI
    participants_tsv = pd.io.parsers.read_csv(os.path.join(dwi_bids, 'participants.tsv'), sep='\t')
    all_subjects = participants_tsv[participants_tsv.diagnosis_bl.map(lambda x: (x.endswith(label)))].participant_id
    all_subjects = all_subjects.unique()
    sublist = all_subjects.tolist()
    subjects_with_dwi = []
    sessions_with_dwi = []
    subjects_with_t1_dwi = []
    sessions_with_t1_dwi = []
    diagnosis_with_t1_dwi = []
    to_remove = [i for i in range(len(sublist)) if sublist[i][0:4] != 'sub-']
    for idx in sorted(to_remove, reverse=True):
        del sublist[idx]
    for sub in range(len(sublist)): 
        sessions_tsv = pd.io.parsers.read_csv(os.path.join(dwi_bids, sublist[sub], sublist[sub] + '_sessions.tsv'), sep='\t')
        sessions = sessions_tsv.session_id.tolist()
        for session in sessions:
            if os.path.isfile(os.path.join(dwi_bids, sublist[sub], session, 'dwi', '%s_%s_acq-axial_dwi.nii.gz' % (sublist[sub], session))):
                subjects_with_dwi.append(sublist[sub])
                sessions_with_dwi.append(session)
                # For the chosen DWI, check if the corresponding T1w exists
                # if os.path.isfile(os.path.join(t1w_bids, sublist[sub], session, 'anat', sublist[sub] + '_%s_T1w.nii.gz', session)):
                if os.path.isfile(os.path.join(t1w_bids, sublist[sub], session, 'anat', sublist[sub] + '_ses-M00_T1w.nii.gz')):
                    subjects_with_t1_dwi.append(sublist[sub])
                    sessions_with_t1_dwi.append(session)

                    diagnose = participants_tsv[participants_tsv.participant_id == sublist[sub]].diagnosis_bl.tolist()
                    # print diagnose
                    diagnosis_with_t1_dwi.append(diagnose[0])

    list_T1 = pd.DataFrame({'participant_id': subjects_with_t1_dwi,
                            'session_id': sessions_with_t1_dwi
                            })
    list_T1_diagnosis = pd.DataFrame({'participant_id': subjects_with_t1_dwi,
                                      'session_id': sessions_with_t1_dwi,
                                      'diagnosis': diagnosis_with_t1_dwi
                                      })

    list_T1.to_csv(os.path.join(output_dir, label + '_' + database + '.tsv'), sep='\t', index=False, encoding='utf8', columns=['participant_id', 'session_id'])
    list_T1_diagnosis.to_csv(os.path.join(output_dir, label + '_diagnosis_' + database + '.tsv'), sep='\t', index=False, encoding='utf8', columns=['participant_id', 'session_id', 'diagnosis'])

