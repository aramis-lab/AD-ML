import subjects.subjects_lists as sl

adni_bids = 'PATH/TO/ADNI/BIDS'
subjects_path = 'PATH/TO/SUBJECTS'

t1_pet_subjects, t1_subjects = sl.create_subjects_lists(adni_bids)

sl.create_diagnosis_all_participants(adni_bids, t1_pet_subjects, subjects_list)

# TODO
# SAVE LISTS
