import subjects_lists.subjects_lists as sl

#you can create all the lists you need for the statistics on the population and for the classification.

path_bids = 'PATH/TO/ADNI/BIDS'
output_path = 'PATH/TO/SUBJECTS'
database = 'ADNI' #ADNI, AIBL or OASIS
subjects_list = 'T1' #T1 or PET if ADNI
adnimerge = 'PATH/TO/ADNIMERGE.csv' #if ADNI

adni_bids = 'PATH/TO/ADNI/BIDS'
subjects_path = 'PATH/TO/SUBJECTS'

sl.run_subjects_lists(path_bids, output_path, database, subjects_list, adnimerge)
