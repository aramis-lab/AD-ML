
__author__ = "Simona Bottani, Jorge Samper Gonzalez, Arnaud Marcoux"
__copyright__ = "Copyright 2016, The Aramis Lab Team"
__credits__ = ["Simona Bottani, Jorge Samper Gonzalez,  Arnaud Marcoux"]
__version__ = "0.1.0"
__maintainer__ = "Jorge Samper Gonzalez"
__email__ = "jorge.samper-gonzalez@inria.fr"
__status__ = "Development"


def create_lists_statistics (path_bids, output_path, database, subjects_list, adnimerge):
    '''

    :param path_bids: path to the  dataset in bids (it works with ADNI, AIBL and OASIS)
    :param output_path: where lists will be saved
    :param subjects_list: is a flag ("T1" or "PET"): it depends on which list we need, if you select PET, participant_id will contain also patient which have a T1
    :param adnimerge: ADNIMERGE.csv (for ADNI)

    If you run this function you can create all the lists you need for the statistics on the population and for the classification. 
    You can see them in the output_path and in the command line you can see the results of the statistics.
     
    All the function derives from subjects_lists.py and lists_stats.py 
    
    Example:
    
    create_lists_statistics('/Users/ADNI_BIDS', '/Users/lists_ADNI', 'ADNI', 'PET')
    

    '''

    from Code.subjects.lists_stats import statistics_cn_ad_mci_M00, statistics_cn_ad_mci_amylod_M00
    from Code.subjects.subjects_lists import max_vis, create_subjects_lists, create_diagnosis_all_participants, obtain_global_list, find_parameters_statistics, obtain_lists_diagnosis, obtain_lists_diagnosis_amyloid, parameters_cn_ad_mci_amylod_M00
    import os
    import pandas as pd

    create_subjects_lists(path_bids, output_path, database)

    if subjects_list =='T1':
        subjects_list = pd.io.parsers.read_csv(os.path.join(output_path, 'list_T1_' + database + '.tsv'),  sep='\t')
        subjects_list = subjects_list.participant_id
    else:
        subjects_list = pd.io.parsers.read_csv(os.path.join(output_path, 'list_T1_' + database + '.tsv'), sep='\t')
        subjects_list = subjects_list.participant_id

    [MCI_CN, MCI_AD_MCI] =  create_diagnosis_all_participants(path_bids, subjects_list, output_path, database)
    [global_list, global_list_name] = obtain_global_list(output_path,database, MCI_CN, MCI_AD_MCI, N_months = 36)

    find_parameters_statistics(path_bids, subjects_list, output_path, database)
    obtain_lists_diagnosis(output_path, database, N_months=36)
    statistics_cn_ad_mci_M00(output_path, database, global_list, global_list_name)

    if database == 'ADNI':
        parameters_cn_ad_mci_amylod_M00(adnimerge, output_path, global_list, global_list_name)
        statistics_cn_ad_mci_amylod_M00(adnimerge, output_path, global_list, global_list_name)
        obtain_lists_diagnosis_amyloid(output_path)
