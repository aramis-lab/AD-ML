import subjects.lists_stats as stat

#All the statistics will be shown in the command line

path_bids = 'PATH/TO/ADNI/BIDS'
output_path = 'PATH/TO/SUBJECTS'
database = 'ADNI' #ADNI, AIBL or OASIS
subjects_list = 'T1' #T1 or PET if ADNI
adnimerge = 'PATH/TO/ADNIMERGE.csv' #if ADNI

stat.run_lists_stats(path_bids, output_path, database, subjects_list, adnimerge)
