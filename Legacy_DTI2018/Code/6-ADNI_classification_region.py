from clinica_ml_dti.mlworkflow_dwi_utils import run_roi_based_classification

# ########################
# ### Original classification
# ########################

caps_directory= PATH/TO/CAPS
output_dir = PATH/TO/CLASSIFICATION_OUTPUT
atlas = ['JHUDTI81', 'JHUTracts25']
atlas_T1 = ['AAL2']
dwi_maps = ['fa', 'md']
n_threads = 72
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
balanced_down_sample = False
modality = 'T1'

######  CN vs AD
task = 'AD_vs_CN_RB'
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, dwi_maps=dwi_maps)

#####  CN vs pMCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'CN_vs_pMCI_RB'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

######  CN vs MCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'CN_vs_MCI_RB'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

######  sMCI vs pMCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'sMCI_vs_pMCI_RB'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

######  CN vs AD
task='AD_vs_CN_RB_T1'
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas_T1,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)

#####  CN vs pMCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'CN_vs_pMCI_RB_T1'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas_T1,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)

######  CN vs MCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'CN_vs_MCI_RB_T1'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas_T1,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)

######  sMCI vs pMCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'sMCI_vs_pMCI_RB_T1'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas_T1,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)


# ########################
# ### Balanced classification
# ########################
balanced_down_sample = True

#####  CN vs pMCI
task = 'CN_vs_pMCI_RB'
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)

######  CN vs MCI
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
task = 'CN_vs_MCI_RB'
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_RB'
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/SUBJECTS_VISITS_TSV
run_roi_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir, atlas,
                                task, n_threads, n_iterations, test_size, grid_search_folds, balanced_down_sample=balanced_down_sample)