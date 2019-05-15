from clinica_ml_dwi.mlworkflow_dwi_utils import run_voxel_based_classification

# ########################
# ### Original classification
# ########################

caps_directory= 'PATH/TO/CAPS_DIR'
output_dir = 'PATH/TO/CLASSIFICATION_OUTPUT'
n_threads = 72
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
modality = 'T1'

# ########################
# ### DWI
# ########################

######  CN vs AD
task='AD_vs_CN_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

#####  CN vs pMCI
task='CN_vs_pMCI_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

######  CN vs MCI
task='CN_vs_MCI_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_VB'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds)

# ########################
# ### T1
# ########################

######  CN vs AD
task='AD_vs_CN_VB_T1'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)

#####  CN vs pMCI

task='CN_vs_pMCI_VB_T1'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)

######  CN vs MCI
task='CN_vs_MCI_VB_T1'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)

######  sMCI vs pMCI
task='sMCI_vs_pMCI_VB_T1'
diagnoses_tsv = 'PATH/TO/DIAGONISIS_TSV'
subjects_visits_tsv = 'PATH/TO/SUBJECTS_VISITS_TSV'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, modality=modality)
