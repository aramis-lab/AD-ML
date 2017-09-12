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


def run_lists_stats(path_bids, output_path, database, subjects_list, adnimerge):
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
        subjects_list = pd.io.parsers.read_csv(os.path.join(output_path, 'list_T1_' + database + '.tsv'), sep='\t')
        subjects_list = subjects_list.participant_id

    [MCI_CN, MCI_AD_MCI] = create_diagnosis_all_participants(path_bids, subjects_list, output_path, database)
    [global_list, global_list_name] = obtain_global_list(output_path, database, MCI_CN, MCI_AD_MCI, N_months=36)

    statistics_cn_ad_mci_M00(output_path, database, global_list, global_list_name)

    if database == 'ADNI':
        statistics_cn_ad_mci_amylod_M00(adnimerge, output_path, global_list, global_list_name)


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

    for li in range(len(global_list)):
        age_bl = []
        sex_bl = []
        mmse_bl = []
        for sub in global_list[li]:
            idx = dict.participant_id[dict.participant_id == sub].index.tolist()
            idx = idx[0]
            age_bl.append(age[idx])
            sex_bl.append(sex[idx])
            mmse_bl.append(mmse[idx])
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
            np.min(np.asarray(mmse_bl))) + ' / ' + str(np.max(np.asarray(mmse_bl))))


def statistics_cn_ad_mci_amylod_M00(adnimerge, output_path, global_list, global_list_name):

    '''
    
    :param adnimerge: csv file (ADNIMERGE.csv) downloaded with the clinical data of ADNI
    :param output_path: path where outputs are saved
    :param global_list: list of subjects for each diagnosis (obtained through the previous method)
    :param global_list_name: list of names of diagnosis
    :return: it prints all the statistics (the subjects are the same presents in dict which are from the subjects_list chosen)
    
    This method is only for ADNI since it contains information to identify the amiloyid status for the patients
    
    '''


    import os
    import pandas as pd
    import numpy as np
    import re

    participants = pd.io.parsers.read_csv(adnimerge, sep=';')
    N_months = 36
    sub_study = os.path.join(output_path, 'diagnosis_' + str(N_months) + '_ADNI.tsv')
    sub_parsed = pd.io.parsers.read_csv(sub_study, sep='\t')
    participant_id = list(participants.PTID)
    vis = list(participants.VISCODE)
    sex = np.asarray(list(participants.PTGENDER))
    age = np.asarray(list(participants.AGE))
    mmse = np.asarray(list(participants.MMSE))

    av45 = np.asarray(list(participants.AV45))
    pib = np.asarray(list(participants.PIB))

    # list of subjects from the previous method
    sub_36 = list(sub_parsed.participant_id)
    diag_36 = list(sub_parsed.diagnosis)
    all_subjects = []
    all_diagnosis = []


    print('\n')
    print('*************** AMYLOID STATUS *************')

    for li in range(len(global_list)):
        age_bl = []
        sex_bl = []
        mmse_bl = []

        am_negatif = []
        am_positif = []

        am_age_p = []
        am_sex_p = []
        am_mmse_p = []

        am_age_n = []
        am_sex_n = []
        am_mmse_n = []

        # statistics for the subjects which have got also information about the amyloid status
        for sub in global_list[li]:
            sub_orig = sub
            sub_ = re.search('sub-ADNI([0-9].*)S([0-9].*)', str(sub))
            sub = str(sub_.group(1) + '_S_' + sub_.group(2))
            idx = [i for i in range(len(participant_id)) if participant_id[i] == sub]
            idx_second = sub_36.index(sub_orig)
            for ses in idx:
                if vis[ses] == 'bl':
                    diag_string = diag_36[idx_second]
                    if diag_string == 'MCIc':
                        diag_string = 'pMCI'
                    elif diag_string == 'MCInc':
                        diag_string = 'sMCI'
                    age_bl.append(age[ses])
                    sex_bl.append(sex[ses])
                    mmse_bl.append(mmse[ses])
                    if not (np.isnan(av45[ses]) and np.isnan(pib[ses])) and participant_id[ses] != '126_S_2360':
                        if not np.isnan(av45[ses]):

                            # threshold to classifify patients as amyloid positive or negative
                            if av45[ses] <= 1.10:
                                am_negatif.append(participant_id[ses])
                                am_age_n.append(age[ses])
                                am_sex_n.append(sex[ses])
                                am_mmse_n.append(mmse[ses])
                                all_subjects.append(
                                    'sub-ADNI' + participant_id[ses][0:3] + 'S' + participant_id[ses][6:])
                                all_diagnosis.append(diag_string + '-')
                            elif av45[ses] > 1.10:
                                am_positif.append(participant_id[ses])
                                am_age_p.append(age[ses])
                                am_sex_p.append(sex[ses])
                                am_mmse_p.append(mmse[ses])
                                all_subjects.append(
                                    'sub-ADNI' + participant_id[ses][0:3] + 'S' + participant_id[ses][6:])
                                all_diagnosis.append(diag_string + '+')
                            else:
                                raise Exception('Error for subject ' + participant_id[ses] + ' with values ' + str(
                                    av45[ses]) + ' of type ' + str(type(av45[ses])))
                        elif not np.isnan(pib[ses]):
                            if pib[ses] <= 1.47:
                                am_negatif.append(participant_id[ses])
                                am_age_n.append(age[ses])
                                am_sex_n.append(sex[ses])
                                am_mmse_n.append(mmse[ses])
                                all_subjects.append(
                                    'sub-ADNI' + participant_id[ses][0:3] + 'S' + participant_id[ses][6:])
                                all_diagnosis.append(diag_string + '-')
                            elif pib[ses] > 1.47:
                                am_positif.append(participant_id[ses])
                                am_age_p.append(age[ses])
                                am_sex_p.append(sex[ses])
                                am_mmse_p.append(mmse[ses])
                                all_subjects.append(
                                    'sub-ADNI' + participant_id[ses][0:3] + 'S' + participant_id[ses][6:])
                                all_diagnosis.append(diag_string + '+')
                            else:
                                raise Exception('Error for subject ' + participant_id[ses])

        age_m = np.mean(np.asarray(age_bl))
        age_u = np.std(np.asarray(age_bl))

        mmse_m = np.mean(np.asarray(mmse_bl))
        mmse_u = np.std(np.asarray(mmse_bl))

        N_women = len([i for i in range(len(age_bl)) if sex_bl[i] == 'Female'])
        N_men = len([i for i in range(len(age_bl)) if sex_bl[i] == 'Male'])

        print ('+-+-+-+-+-+-+-+' + global_list_name[li] + '-+-+-+-+-+-+-+-+-+-+-+-+-')
        print ('Group of len : ' + str(len(age_bl)) + ' has age = ' + str(age_m) + ' +/- ' + str(age_u))
        print ('N male = ' + str(N_men) + ' N female = ' + str(N_women))
        print ('MMSE = ' + str(mmse_m) + ' +/- ' + str(mmse_u))

        print('\t----AMYLOID STATUS REPORT----')
        N_w_amn = len([i for i in range(len(am_negatif)) if am_sex_n[i] == 'Female'])
        N_m_amn = len([i for i in range(len(am_negatif)) if am_sex_n[i] == 'Male'])
        N_w_amp = len([i for i in range(len(am_positif)) if am_sex_p[i] == 'Female'])
        N_m_amp = len([i for i in range(len(am_positif)) if am_sex_p[i] == 'Male'])
        print ('Amyloid negatif N male = ' + str(N_m_amn) + ' N female = ' + str(N_w_amn))
        print ('Amyloid positif N male = ' + str(N_m_amp) + ' N female = ' + str(N_w_amp))
        age_m_amn = np.mean(np.asarray(am_age_n))
        age_u_amn = np.std(np.asarray(am_age_n))
        mmse_m_amn = np.mean(np.asarray(am_mmse_n))
        mmse_u_amn = np.std(np.asarray(am_mmse_n))

        age_m_amp = np.mean(np.asarray(am_age_p))
        age_u_amp = np.std(np.asarray(am_age_p))
        mmse_m_amp = np.mean(np.asarray(am_mmse_p))
        mmse_u_amp = np.std(np.asarray(am_mmse_p))

        print('Amyloid negatif of group size : ' + str(len(am_age_n)) + ' has age = ' + str(age_m_amn) + ' +/- ' + str(
            age_u_amn) + ' and range = ' + str(np.min(np.asarray(am_age_n))) + ' / ' + str(
            np.max(np.asarray(am_age_n))))
        print('Amyloid positif of group size : ' + str(len(am_age_p)) + ' has age = ' + str(age_m_amp) + ' +/- ' + str(
            age_u_amp) + ' and range = ' + str(np.min(np.asarray(am_age_p))) + ' / ' + str(
            np.max(np.asarray(am_age_p))))
        print ('MMSE amyloid negatif = ' + str(mmse_m_amn) + ' +/- ' + str(mmse_u_amn) + ' and range = ' + str(
            np.min(np.asarray(am_mmse_n))) + ' / ' + str(np.max(np.asarray(am_mmse_n))))
        print ('MMSE amyloid positif = ' + str(mmse_m_amp) + ' +/- ' + str(mmse_u_amp) + ' and range = ' + str(
            np.min(np.asarray(am_mmse_p))) + ' / ' + str(np.max(np.asarray(am_mmse_p))))
        print('\n')
