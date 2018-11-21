
def population_stats(path_bids, subjects_diagnoses_list, dataset):

    import pandas as pd
    from os import path
    import numpy as np
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    subjects_list = pd.io.parsers.read_csv(subjects_diagnoses_list, sep='\t')
    participants = pd.io.parsers.read_csv(path.join(path_bids, 'participants.tsv'), sep='\t')

    diagnoses = subjects_list.diagnosis.unique()

    mmse_key = {'ADNI': 'MMSE', 'AIBL': 'MMS', 'OASIS': 'MMS'}

    for dx in diagnoses:
        dx_subjects = subjects_list[subjects_list.diagnosis == dx]
        dx_participants = participants[participants.participant_id.isin(dx_subjects.participant_id)]

        age = []
        cdr = []
        mmse = []

        for row in dx_participants.iterrows():
            participant = row[1]
            sessions = pd.io.parsers.read_csv(path.join(path_bids, participant.participant_id, '%s_sessions.tsv' % participant.participant_id), sep='\t')
            bl_session = sessions[sessions.session_id == 'ses-M00']

            cdr.append(bl_session.cdr_global.values[0])
            mmse.append(bl_session[mmse_key[dataset]].values[0])

            if dataset == 'ADNI':
                age.append(bl_session.age.values[0])
            elif dataset == 'AIBL':
                d1 = datetime(year=participant.date_of_birth, month=1, day=1)
                d2 = datetime(year=int(bl_session.examination_date.values[0][-4:]), month=1, day=1)
                age.append(relativedelta(d2, d1).years)
            elif dataset == 'OASIS':
                age.append(participant.age_bl)
            else:
                raise Exception('Unknown dataset')

        print ('+-+-+-+-+-+-+-+' + dx + '-+-+-+-+-+-+-+-+-+-+-+-+-')
        print ('Group of len : ' + str(len(dx_participants)) + ' has age = ' + str(np.mean(age)) + ' +/- ' + str(np.std(age)) +
               ' and range = ' + str(np.min(age)) + ' / ' + str(np.max(age)))

        print dx_participants.sex.value_counts()

        print ('MMSE = ' + str(np.mean(mmse)) + ' +/- ' + str(np.std(mmse)) + ' and range = ' +
               str(np.min(mmse)) + ' / ' + str(np.max(mmse)))

        print ('CDR:' + str(cdr.count(0)) + '(0); ' + str(cdr.count(0.5)) + '(0.5); ' + str(cdr.count(1)) + '(1); ' +
               str(cdr.count(2)) + '(2); ')
