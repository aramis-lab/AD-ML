from subjects_lists.subject_lists import create_subjects_lists

dwi_bids = 'PATH/TO/BIDS/DWI'
t1w_bids = 'PATH/TO/BIDS/T1'
output_dir = 'PATH/TO/OUTPUT/TSV'

# for AD
create_subjects_lists(dwi_bids, t1w_bids, output_dir, label='AD')

# for CN
create_subjects_lists(dwi_bids, t1w_bids, output_dir, label='CN')

# for MCI
create_subjects_lists(dwi_bids, t1w_bids, output_dir, label='MCI')
