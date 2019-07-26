__author__ = "Simona Bottani, Jorge Samper Gonzalez, Arnaud Marcoux"
__copyright__ = "Copyright 2016, The Aramis Lab Team"
__credits__ = ["Simona Bottani, Jorge Samper Gonzalez,  Arnaud Marcoux"]
__version__ = "0.1.0"
__maintainer__ = "Jorge Samper Gonzalez"
__email__ = "jorge.samper-gonzalez@inria.fr"
__status__ = "Development"

'''
This function contains all the method to compute the statistics of the population selected 

'''


def run_lists_stats(path_bids, output_path, database, subjects_list='T1', adnimerge=None):
    '''

    :param path_bids: path to the  dataset in bids (it works with ADNI, AIBL and OASIS)
    :param output_path: where lists will be saved
    :param subjects_list: is a flag ("T1" or "PET"): it depends on which list we need, if you select PET, participant_id will contain also patient which have a T1
    :param adnimerge: ADNIMERGE.csv (for ADNI)

    If you run this function you can see all the statistics in the command line

    Example:

    run_lists_stats('/Users/ADNI_BIDS', '/Users/lists_ADNI', 'ADNI', 'PET')


    '''

    from Code.subjects.lists_stats import statistics_cn_ad_mci_M00, statistics_cn_ad_mci_amylod_M00
    from Code.subjects.subjects_lists import run_subjects_lists, create_diagnosis_all_participants, obtain_global_list

    import pandas as pd
    import os

    run_subjects_lists(path_bids, output_path, database, subjects_list, adnimerge)

    if subjects_list == 'T1':
        subjects_list = pd.io.parsers.read_csv(os.path.join(output_path, 'list_T1_' + database + '.tsv'), sep='\t')
        subjects_list = subjects_list.participant_id
    else:
        subjects_list = pd.io.parsers.read_csv(os.path.join(output_path, 'list_PET_' + database + '.tsv'), sep='\t')
        subjects_list = subjects_list.participant_id

    [MCI_CN, MCI_AD_MCI] = create_diagnosis_all_participants(path_bids, subjects_list, output_path, database)
    [global_list, global_list_name] = obtain_global_list(output_path, database, MCI_CN, MCI_AD_MCI, N_months=36)

    statistics_cn_ad_mci_M00(output_path, database, global_list, global_list_name)

    if database == 'ADNI':
        statistics_cn_ad_mci_amylod_M00(path_bids, adnimerge, output_path, global_list, global_list_name)


def statistics_cn_ad_mci_M00(output_path, database, global_list, global_list_name):
    '''
    :param output_path: where files with lists have been saved
    :param database: name of the database
    :param global_list: list of subjects for each diagnosis (obtained through the previous method)
    :param global_list_name: list of names of diagnosis
    :return: it prints all the statistics (the subjects are the same presents in dict which are from the subjects_list chosen)

    '''

    import re
    import numpy as np
    import pandas as pd
    import os

    print('\n')
    print('******** DIAGNOSIS 36 MONTHS -  AD CN sMCI pMCI *********')


    dict = pd.io.parsers.read_csv(
        os.path.join(output_path, 'participants_parameters_for_statistics' + '_' + database + '.tsv'), sep='\t')
    age = np.asarray(dict.age)
    sex = np.asarray(dict.sex)
    mmse = np.asarray(dict.mmscore)
    cdr = np.asarray(dict.cdr)

    for li in range(len(global_list)):
        age_bl = []
        sex_bl = []
        mmse_bl = []
        cdr_bl = []
        for sub in global_list[li]:
            idx = dict.participant_id[dict.participant_id == sub].index.tolist()
            idx = idx[0]
            age_bl.append(age[idx])
            sex_bl.append(sex[idx])
            mmse_bl.append(mmse[idx])
            cdr_bl.append(cdr[idx])
        
        age_m = np.mean(np.asarray(age_bl))
        age_u = np.std(np.asarray(age_bl))

        mmse_m = np.mean(np.asarray(mmse_bl))
        mmse_u = np.std(np.asarray(mmse_bl))


        N_women = len([i for i in range(len(age_bl)) if sex_bl[i] == 'F'])
        N_men = len([i for i in range(len(age_bl)) if sex_bl[i] == 'M'])

        print ('+-+-+-+-+-+-+-+' + global_list_name[li] + '-+-+-+-+-+-+-+-+-+-+-+-+-')
        print ('Group of len : ' + str(len(age_bl)) + ' has age = ' + str(age_m) + ' +/- ' + str(
            age_u) + ' and range = ' + str(np.min(np.asarray(age_bl))) + ' / ' + str(np.max(np.asarray(age_bl))))
        print ('N male = ' + str(N_men) + ' N female = ' + str(N_women))
        print ('MMSE = ' + str(mmse_m) + ' +/- ' + str(mmse_u) + ' and range = ' + str(
            np.min(np.asarray(mmse_bl))) + ' / ' + str(np.max(np .asarray(mmse_bl))))
        print ('CDR:' + str(cdr_bl.count(0)) + '(0); ' + str(cdr_bl.count(0.5)) + '(0.5); ' + str(
            cdr_bl.count(1)) + '(1); ' + str(cdr_bl.count(2)) + '(2); ')


def statistics_cn_ad_mci_amylod_M00(path_bids, output_path):

    '''

    :param path_bids:
    :param adnimerge: csv file (ADNIMERGE.csv) downloaded with the clinical data of ADNI
    :param output_path: path where outputs are saved
    :param global_list: list of subjects for each diagnosis (obtained through the previous method)
    :param global_list_name: list of names of diagnosis
    :return: it prints all the statistics (the subjects are the same presents in dict which are from the subjects_list chosen

    This method is only for ADNI since it contains information to identify the amiloyid status for the patients

    '''

    import os
    import pandas as pd
    import numpy as np
    diagnosis = ['AD-', 'AD+', 'CN-', 'CN+', 'MCI-', 'MCI+', 'pMCI+', 'pMCI-', 'sMCI+', 'sMCI-']
    participants_tsv = pd.io.parsers.read_csv(os.path.join(path_bids, 'participants.tsv'), sep='\t')
    for i in diagnosis:
        path_amyloid = pd.io.parsers.read_csv(os.path.join(output_path, 'tasks_ADNI_' + i + '.tsv'),
                                              sep='\t').participant_id
        sex = []
        age = []
        mmse = []
        cdr = []
        for j in path_amyloid.values:
            ses = pd.io.parsers.read_csv(os.path.join(path_bids, j, j + '_sessions.tsv'), sep='\t')
            sex.append(participants_tsv[participants_tsv.participant_id == j].sex.item())
            age.append(ses[ses.session_id == 'ses-M00'].age.item())
            mmse.append(ses[ses.session_id == 'ses-M00'].MMS.item())
            cdr.append(ses[ses.session_id == 'ses-M00'].cdr_global.item())

        age_m = np.mean(np.asarray(age))
        age_u = np.std(np.asarray(age))
        mmse_m = np.mean(np.asarray(mmse))
        mmse_u = np.std(np.asarray(mmse))

        N_women = len([x for x in range(len(age)) if sex[x] == 'F'])
        N_men = len([x for x in range(len(age)) if sex[x] == 'M'])

        print ('****   ' + i + '   *****')
        print ('Group of len : ' + str(len(age)))
        print ('N male = ' + str(N_men) + ' N female = ' + str(N_women))
        print ('AGE = ' + str(age_m) + ' +/- ' + str(age_u) + ' range ' + str(np.min(np.asarray(age))) + ' / ' + str(
            np.max(np.asarray(age))))
        print (
        'MMSE = ' + str(mmse_m) + ' +/- ' + str(mmse_u) + ' range ' + str(np.min(np.asarray(mmse))) + ' / ' + str(
            np.max(np.asarray(mmse))))

        print ('CDR:' + str(cdr.count(0)) + '(0); ' + str(cdr.count(0.5)) + '(0.5); ' + str(
            cdr.count(1)) + '(1); ' + str(cdr.count(2)) + '(2); ')




