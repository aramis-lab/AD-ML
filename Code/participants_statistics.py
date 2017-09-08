def max_vis(mylist):
    maxi = 0
    for e in range(len(mylist)):
        num = int(mylist[e][5:])
        if num > maxi:
            maxi = num
    return maxi


def create_subjects_lists(path_bids):

    '''

    :param path_bids: path to the  dataset in bids (it works with ADNI, AIBL and OASIS)

    :return: list of subjects with T1 and PET
    '''

    # path to the adni bids folder
    import os
    import numpy as np
    import pandas as pd

    # read the participants.tsv file in ADNI
    all_subjects = pd.io.parsers.read_csv(os.path.join(path_bids, 'participants.tsv'), sep='\t')
    all_subjects = all_subjects.participant_id
    sublist = all_subjects.tolist()
    subjects_with_t1 = []
    subjects_with_pet = []

    to_remove = [i for i in range(len(sublist)) if sublist[i][0:4] != 'sub-']
    for idx in sorted(to_remove, reverse=True):
        del sublist[idx]
    for sub in range(len(sublist)):

        # read the session.tsv for each subject
        if os.path.isfile(os.path.join(path_adni_bids, sublist[sub], sublist[sub] + '_sessions.tsv')):
            read_session = pd.io.parsers.read_csv(
                os.path.join(path_adni_bids, sublist[sub], sublist[sub] + '_sessions.tsv'), sep='\t')

            if os.path.exists(os.path.join(path_adni_bids, sublist[sub], 'ses-M00')):

                folder_in_bl = os.listdir(os.path.join(path_adni_bids, sublist[sub], 'ses-M00'))

                has_anat = len([i for i in range(len(folder_in_bl)) if folder_in_bl[i] == 'anat']) == 1

                has_pet = len([i for i in range(len(folder_in_bl)) if folder_in_bl[i] == 'pet']) == 1

                if not has_anat:
                    print(
                    sublist[sub] + ' does not have a anat folder for its T1-MRI in ses-M00 folder. Subject dissmissed')
                else:
                    if os.path.isfile(os.path.join(path_adni_bids, sublist[sub], 'ses-M00', 'anat',
                                                   sublist[sub] + '_ses-M00_T1w.nii.gz')):

                        subjects_with_t1.append(sublist[sub])

                        #we consider only the subjects with t1 in the PET list
                        if os.path.isfile(os.path.join(path_adni_bids, sublist[sub], 'ses-M00', 'pet',
                                                       sublist[sub] + '_ses-M00_task-rest_acq-fdg_pet.nii.gz')):
                            subjects_with_pet.append(sublist[sub])

    return subjects_with_pet, subjects_with_t1

subjects_with_pet, subjects_with_t1 = create_subjects_lists(path_bids)


def create_diagnosis_all_participants_ADNI(path_adni_bids, output_path):
    '''

    :param path_adni_bids: path to the adni dataset in bids 
    :param all_subjects: participants_tsv file
    :param subjects_with_pet: list which contains all the subjects with pet
    :return: 
    '''

    import os
    import numpy as np
    import pandas as pd

    # read the participants.tsv file in ADNI
    all_subjects = pd.io.parsers.read_csv(os.path.join(path_adni_bids, 'participants.tsv'), sep='\t')
    all_subjects = all_subjects.participant_id

    sublist = all_subjects.tolist()

    to_remove = [i for i in range(len(sublist)) if sublist[i][0:4] != 'sub-']
    for idx in sorted(to_remove, reverse=True):
        del sublist[idx]

    final_subject_list = []
    final_diagnosis_list = []

    # analysis based on the visits in 3 yeas
    N_months = 36
    discarded = 0
    discarded_list = []

    do_not_have_anat = 0
    do_not_have_pet = 0
    MCI_CN = []
    MCI_inconsistent = 0
    MCI_AD_MCI = []
    CN_change = 0
    AD_change = 0
    no_diagnosis_bl = 0

    # iteration for all the subjects in the participants.tsv
    for sub in range(len(sublist)):

        #read the session.tsv for each subject
        read_session = pd.io.parsers.read_csv(os.path.join(path_adni_bids, sublist[sub], sublist[sub] + '_sessions.tsv'), sep='\t')
        session = list(read_session.session_id)
        diagnosis = list(read_session.diagnosis)

        #flag to indicate if the subjects have a pet
        has_pet = False
        if os.path.exists(os.path.join(path_adni_bids, sublist[sub], 'ses-M00')):
            folder_in_bl = os.listdir(os.path.join(path_adni_bids, sublist[sub], 'ses-M00'))
            has_anat = len([i for i in range(len(folder_in_bl)) if folder_in_bl[i] == 'anat']) == 1
            if not has_anat:
                print(sublist[sub] + ' does not have a anat folder for its T1-MRI in ses-M00 folder. Subject dissmissed')
                do_not_have_anat += 1
            if sublist[sub] in subjects_with_pet:
                has_pet = True
            else:
                has_pet = False
                do_not_have_pet += 1
                print(sublist[sub] + ' does not have a PET')
                continue
        else:
            has_anat = False
            print(sublist[sub] + ' does not have a ses-M00 folder. Subject dissmissed')

        #for each participant included (because it have a T1 and a PET image) a diagnosis at the baseline is assigned
        idx_bl = session.index('ses-M00')
        diagnosis_bl = diagnosis[idx_bl]
        to_remove = False

        #classification of the different type of MCI (MCI which remains stable, MCI which converts in AD and MCI whose future status is unknown becase they havent been followed for 36 months at least)
        if diagnosis_bl == 'MCI':
            has_converted = False
            for ses in range(len(diagnosis)):
                if int(session[ses][5:]) <= N_months and diagnosis[ses] == 'AD':
                    has_converted = True
                if diagnosis[ses] == 'CN':
                    if not to_remove:
                        print (sublist[sub] + ' is MCI in baseline but moves back to CN in one of its timepoints. '
                           + 'Subject discarded')
                        MCI_CN.append(sublist[sub])
                    diagnosis_bl = 'Inconsistent'
            if has_converted:
                diagnosis_bl = 'pMCI'
            else:
                if max_vis(session) < N_months:
                    diagnosis_bl = 'uMCI'
                    if not to_remove:
                        print(sublist[sub] + ' is undetermined MCI : not enough timepoint to say if sMCI or pMCI.'
                          + 'Subject discarded')
                        MCI_inconsistent += 1
                else:
                    diagnosis_bl = 'sMCI'
            need_MCI_AD_MCI_check = False
            for ses in range(len(diagnosis)):
                if diagnosis[ses] == 'AD':
                    need_MCI_AD_MCI_check = True
            if need_MCI_AD_MCI_check:
                last_AD_time = 0
                last_MCI_time = 0
                for ses in range(len(diagnosis)):
                    if diagnosis[ses] == 'AD':
                        if int(session[ses][5:]) > last_AD_time:
                            last_AD_time = int(session[ses][5:])
                    if diagnosis[ses] == 'MCI':
                        if int(session[ses][5:]) > last_MCI_time:
                            last_MCI_time = int(session[ses][5:])
                if last_MCI_time > last_AD_time:
                    if not to_remove:
                        print(sublist[sub] + ' is MCI at baseline, then goes AD, and then back to MCI. Subject discarded')
                        MCI_AD_MCI.append(sublist[sub])
        elif diagnosis_bl == 'CN':
            for l in range(len(session)):
                if int(session[l][5:]) <= N_months and diagnosis[l] != 'CN' and diagnosis[l] == diagnosis[l]:
                    if not to_remove:
                        print(sublist[sub] + ' is CN at baseline and change within the first ' + str(N_months) + ' months')
                        CN_change += 1
        elif diagnosis_bl == 'AD':
            for l in range(len(session)):
                if diagnosis[l] != 'AD' and diagnosis[l] == diagnosis[l]:
                    if not to_remove:
                        print(sublist[sub] + ' is AD at baseline and change within the following months. Subject discarded')
                        AD_change += 1
        elif diagnosis_bl != diagnosis_bl:
            to_remove = True
            print(sublist[sub] + ' do not have a diagnosis at baseline, subject discarded')
            no_diagnosis_bl += 1

        #final subject list
        if has_anat and has_pet and not to_remove:
            final_subject_list.append(sublist[sub])
            final_diagnosis_list.append(diagnosis_bl)
        else:
            discarded_list.append(sublist[sub])
            discarded += 1

    print ('subjects discarded = ' + str(discarded))

    #Dataframe with the participant_id and the corresponding diagnosis
    dict = pd.DataFrame({'participant_id': final_subject_list,
                     'diagnosis': final_diagnosis_list
                     })

    #results is saved in a tsv file
    dict.to_csv(os.path.join(output_path, 'adni_diagnosis_' + str(N_months) + '.tsv'), sep='\t', index=False, encoding='utf8',
                columns=['participant_id', 'diagnosis'])

    return [MCI_CN, MCI_AD_MCI]



def obtain_global_list(path_bids, output_path):
    import numpy as np
    import pandas as pd
    import os
    import re

    subjects_with_pet, subjects_with_t1 = create_diagnosis_all_participants_ADNI(path_adni_bids, output_path)

    # parameters are taken from adnimerge

    # the subjects taken into account derived from the list obtained with the previous method
    sub_study = os.path.join(output_path, 'adni_diagnosis_' + str(n_months) + '.tsv')
    sub_study_parsed = pd.io.parsers.read_csv(sub_study, sep='\t')
    sub_36 = list(sub_study_parsed.participant_id)
    diag_36 = list(sub_study_parsed.diagnosis)

    # list for each diagnosis to make the statistics
    CN_list = [sub_36[i] for i in range(len(sub_36)) if diag_36[i] == 'CN']
    AD_list = [sub_36[i] for i in range(len(sub_36)) if diag_36[i] == 'AD']
    MCIc_list = [sub_36[i] for i in range(len(sub_36)) if diag_36[i] == 'pMCI']
    MCInc_list = [sub_36[i] for i in range(len(sub_36)) if diag_36[i] == 'sMCI']
    uMCI_list = [sub_36[i] for i in range(len(sub_36)) if diag_36[i] == 'uMCI']
    [MCI_CN, MCI_AD_MCI] = create_diagnosis_all_participants_ADNI(path_adni_bids, output_path)
    MCI_list = list(set(uMCI_list + MCIc_list + MCInc_list + MCI_CN + MCI_AD_MCI))

    global_list = [CN_list, AD_list, MCIc_list, MCInc_list, uMCI_list, MCI_list]
    global_list_name = ['CN', 'AD', 'pMCI', 'sMCI', 'uMCI', 'MCI']

    return global_list, global_list_name
    # iteration to make the statistics for each diagnosis

def statistics_cn_ad_mci_M00(path_bids, output_path):
    import re
    import numpy as np
    import pandas as pd
    import os

    global_list, global_list_name = obtain_global_list(path_bids, output_path)

    participants = pd.io.parsers.read_csv(os.path.join(path_bids, 'participants.tsv'), sep='\t')
    participants_id = list(participants.participant_id)
    #participant_id = list(participants.PTID)
    vis = list(participants.VISCODE)
    #sex = np.asarray(list(participants.PTGENDER))
    sex=np.asarray(participants.sex)

    age = np.asarray(list(participants.AGE))
    #mmse = np.asarray(list(participants.MMSE))
    mmse=np.asarray(participants.mmse_bl)
    n_months = 36

    print('\n')
    print('******** DIAGNOSIS 36 MONTHS -  AD CN sMCI pMCI *********')


    for li in range(len(global_list)):
        age_bl = []
        sex_bl = []
        mmse_bl = []
        for sub in global_list[li]:
            sub_ = re.search('sub-ADNI([0-9].*)S([0-9].*)', str(sub))
            sub = str(sub_.group(1) + '_S_' + sub_.group(2))
            idx = [i for i in range(len(participant_id)) if participant_id[i] == sub]
            for ses in idx:

                # we are interested only at the analysis in the bl
                if vis[ses] == 'bl':
                    age_bl.append(age[ses])
                    sex_bl.append(sex[ses])
                    mmse_bl.append(mmse[ses])

        age_m = np.mean(np.asarray(age_bl))
        age_u = np.std(np.asarray(age_bl))

        mmse_m = np.mean(np.asarray(mmse_bl))
        mmse_u = np.std(np.asarray(mmse_bl))

        N_women = len([i for i in range(len(age_bl)) if sex_bl[i] == 'Female'])
        N_men = len([i for i in range(len(age_bl)) if sex_bl[i] == 'Male'])

        print ('+-+-+-+-+-+-+-+' + global_list_name[li] + '-+-+-+-+-+-+-+-+-+-+-+-+-')
        print ('Group of len : ' + str(len(age_bl)) + ' has age = ' + str(age_m) + ' +/- ' + str(
            age_u) + ' and range = ' + str(np.min(np.asarray(age_bl))) + ' / ' + str(np.max(np.asarray(age_bl))))
        print ('N male = ' + str(N_men) + ' N female = ' + str(N_women))
        print ('MMSE = ' + str(mmse_m) + ' +/- ' + str(mmse_u) + ' and range = ' + str(
            np.min(np.asarray(mmse_bl))) + ' / ' + str(np.max(np.asarray(mmse_bl))))



def statistics_cn_ad_mci_amylod_M00(adnimerge, output_path):
    import os
    import pandas as pd
    import numpy as np
    import re
    subjects_with_pet, subjects_with_t1 = create_subjects_lists(path_bids)
    participants = pd.io.parsers.read_csv(adnimerge, sep=';')
    N_months = 36
    sub_study = os.path.join(output_path, 'adni_diagnosis_' + str(N_months) + '.tsv')
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

    # list obtained previously
    global_list, global_list_name = obtain_global_list(adnimerge, output_path)


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

    # Datatrame to resume the amyloid status
    dict = pd.DataFrame({'participant_id': all_subjects,
                         'diagnosis': all_diagnosis
                         })
    dict.to_csv(os.path.join(output_path, 'amyloid_adni.tsv'), sep='\t', index=False, encoding='utf8',
                columns=['participant_id', 'diagnosis'])




def obtain_lists_diagnosis(amyloid_list, diagnosis_list):
    import os
    import pandas as pd
    import numpy as np
    from collections import defaultdict

    amyloid_list = pd.io.parsers.read_csv(amyloid_list, sep='\t')
    dx = pd.io.parsers.read_csv(diagnosis_list, sep='\t')
    am = amyloid_list.drop_duplicates()

    d = defaultdict(list)
    for i in xrange(len(dx.participant_id)):

        diagnosis = dx.loc[(dx["participant_id"] == dx.participant_id[i]), "diagnosis"]

        if diagnosis[i] == "AD":
            d["AD"].append(dx.participant_id[i])
        if diagnosis[i] == "CN":
            d["CN"].append(dx.participant_id[i])
        if diagnosis[i] == "pMCI":
            d["pMCI"].append(dx.participant_id[i])
            d["MCI"].append(dx.participant_id[i])
        if diagnosis[i] == "sMCI":
            d["sMCI"].append(dx.participant_id[i])
            d["MCI"].append(dx.participant_id[i])
        if diagnosis[i] == "uMCI":
            d["uMCI"].append(dx.participant_id[i])
            d["MCI"].append(dx.participant_id[i])

        sub_id = dx.participant_id[i]
        index = am.diagnosis[am.participant_id == sub_id].index.tolist()
        if index != []:
            diagnosis_am = am.diagnosis[index]
            index = index[0]

            if diagnosis_am[index] == "AD+":
                d["AD+"].append(dx.participant_id[i])
            if diagnosis_am[index] == "CN-":
                d["CN-"].append(dx.participant_id[i])
            if diagnosis_am[index] == "AD-":
                d["AD-"].append(dx.participant_id[i])
            if diagnosis_am[index] == "CN+":
                d["CN+"].append(dx.participant_id[i])
            if diagnosis_am[index] == "pMCI-":
                d["pMCI-"].append(dx.participant_id[i])
                d["MCI-"].append(dx.participant_id[i])
            if diagnosis_am[index] == "pMCI+":
                d["pMCI+"].append(dx.participant_id[i])
                d["MCI+"].append(dx.participant_id[i])
            if diagnosis_am[index] == "sMCI-":
                d["sMCI-"].append(dx.participant_id[i])
                d["MCI-"].append(dx.participant_id[i])
            if diagnosis_am[index] == "sMCI+":
                d["sMCI+"].append(dx.participant_id[i])
                d["MCI+"].append(dx.participant_id[i])
            if diagnosis_am[index] == "uMCI-":
                d["MCI-"].append(dx.participant_id[i])
            if diagnosis_am[index] == "uMCI+":
                d["MCI+"].append(dx.participant_id[i])

    names = ['AD', 'AD+.tsv', 'AD-.tsv', 'CN', 'CN+.tsv', 'CN-.tsv', 'pMCI.tsv', 'pMCI-.tsv', 'pMCI+.tsv', 'MCI+.tsv',
             'MCI-.tsv', 'MCI', 'sMCI.tsv', 'sMCI-.tsv', 'sMCI+.tsv', 'uMCI.tsv']

    session_id = 'ses-M00'

    for r in range(len(d)):
        list_tsv = pd.DataFrame({'participant_id': d[names[r]],
                                 'session_id': session_id
                                 })
        list_tsv.to_csv(os.path.join(output_path, 'tasks_ADNI_new' + names[r]), sep='\t', index=False, encoding='utf8',
                        columns=['participant_id', 'session_id'])


def find_parameters_statistics_aibl(participant_tsv,path_to_bids,aibl_dartel,output_table):
    #output_table=where to save the final tsv
    #this function create a tsv file where for each participant of aibl_dartel it's reported the age, diagnosis and mmscore at the baseline

    import re
    import pandas
    participants = pd.io.parsers.read_csv(os.path.join(participant_tsv), sep='\t')
    dartel=pd.io.parsers.read_csv(os.path.join(aibl_dartel), sep='\t')
    rid_dartel=dartel.participant_id
    rid=participants.alternative_id_1
    path=participants.participant_id
    #sub_without_exam_date=[]
    dict=[]
    all_gender=[]
    all_diagnosis=[]
    all_date=[]
    all_mmscore=[]
    #for i in path:
    for i in rid_dartel:
        #for each patient used in the dartel for the paper
        session_file=os.path.join(path_to_bids,i,i+ '_sessions.tsv')
        if os.path.exists(session_file):
            session_file_read=pd.io.parsers.read_csv(session_file, sep='\t')
        if i!='sub-AIBL1503':
            date_bl=session_file_read.examination_date[0]
            #if date_bl!=-4:
            year_bl=re.search('[0-9].*/[0-9].*/([0-9].*)', str(date_bl)).group(1)  # string from image directory
        else:
            year_bl=2014

        dob = participants.loc[(participants["participant_id"] == i), 'date_of_birth']
        index_gender = participants.participant_id[participants.participant_id == i].index.tolist()
        index_gender = index_gender[0]
        gender = participants.sex[index_gender][0]
        #gender=participants.iloc[(participants["participant_id"] == i), 'sex']
        if gender == '[F]' :
            gender = 'F'
        else:
            gender = 'M'
        index = session_file_read.session_id[session_file_read.session_id == 'M00'].index.tolist()
        diagnosis = session_file_read.diagnosis[index][0]
        mmscore=session_file_read.MMS[index][0]
        all_mmscore.append(mmscore) #mmscore at M00
        all_diagnosis.append(diagnosis) #diagnosis at M00
        age=int(int(year_bl)-dob) #age at M00
        all_date.append(age)
        all_gender.append(gender) #gender


    all_sex=[]
    for j in xrange(len(all_gender)):
        sex=all_gender[j]
        all_sex.append(sex)

    dict = pandas.DataFrame({'subjects': rid_dartel,
                             'sex': all_sex,
                             'age': all_date,
                             'diagnosis': all_diagnosis,
                             'mmscore': all_mmscore
                             })
    dict.to_csv(os.path.join(output_table, 'table.tsv'), sep='\t', index=False, encoding='utf8')

