# -*- coding: utf-8 -*-
__author__ = ["Junhao Wen", "Simona Bottani", "Jorge Samper-Gonzalez"]
__copyright__ = "Copyright 2016-2018 The Aramis Lab Team"
__credits__ = ["Junhao Wen"]
__license__ = "See LICENSE.txt file"
__version__ = "0.1.0"
__status__ = "Development"

import numpy as np
import pandas as pd
import os

def statistics_cn_ad_mci_M00(dwi_bids, output_dir):
    '''
    This is a function to calculate the demographic information for the chosen population
    :param output_dir: where files with lists have been saved from the previous step
    :param dwi_bids: BIDS directory for dwi
    :return: it prints all the statistics (the subjects are the same presents in dict which are from the subjects_list chosen)

    '''

    diagnosis = ['AD', 'CN', 'MCI', 'pMCI', 'sMCI']#, 'CN-', 'CN+', 'MCI-', 'MCI+', 'pMCI+', 'pMCI-', 'sMCI+', 'sMCI-']
    participants_tsv = pd.io.parsers.read_csv(os.path.join(dwi_bids, 'participants.tsv'), sep='\t')
    for label in diagnosis:
        path_diagnosis = pd.io.parsers.read_csv(os.path.join(output_dir, label + '_ADNI' +'.tsv'),
                                              sep='\t').participant_id
        sex = []
        age = []
        mmse = []
        cdr = []
        for sub in path_diagnosis.values:
            ses = pd.io.parsers.read_csv(os.path.join(dwi_bids, sub, sub + '_sessions.tsv'), sep='\t')
            sex.append(participants_tsv[participants_tsv.participant_id == sub].sex.item())
            age.append(ses[ses.session_id == 'ses-M00'].age.item())
            mmse.append(ses[ses.session_id == 'ses-M00'].MMS.item())
            cdr.append(ses[ses.session_id == 'ses-M00'].cdr_global.item())

        age_m = np.mean(np.asarray(age))
        age_u = np.std(np.asarray(age))
        mmse_m = np.mean(np.asarray(mmse))
        mmse_u = np.std(np.asarray(mmse))

        N_women = len([x for x in range(len(age)) if sex[x] == 'F'])
        N_men = len([x for x in range(len(age)) if sex[x] == 'M'])

        print ('****   ' + label + '   *****')
        print ('Group of len : ' + str(len(age)))
        print ('N male = ' + str(N_men) + ' N female = ' + str(N_women))
        print ('AGE = ' + str(age_m) + ' +/- ' + str(age_u) + ' range ' + str(np.min(np.asarray(age))) + ' / ' + str(
            np.max(np.asarray(age))))
        print (
        'MMSE = ' + str(mmse_m) + ' +/- ' + str(mmse_u) + ' range ' + str(np.min(np.asarray(mmse))) + ' / ' + str(
            np.max(np.asarray(mmse))))

        print ('CDR:' + str(cdr.count(0)) + '(0); ' + str(cdr.count(0.5)) + '(0.5); ' + str(
            cdr.count(1)) + '(1); ' + str(cdr.count(2)) + '(2); ')
