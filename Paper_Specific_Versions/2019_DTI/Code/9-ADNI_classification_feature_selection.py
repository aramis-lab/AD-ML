from clinica_ml_dwi.mlworkflow_dwi_utils import run_voxel_based_classification

# ########################
# ### Original classification
# ########################

caps_directory= PATH/TO/CAPS
output_dir = PATH/TO/CLASSIFICATION_OUTPUT
diagnoses_tsv = PATH/TO/DIAGONISIS_TSV
subjects_visits_tsv = PATH/TO/DIAGONISIS_TSV
n_threads = 72
n_iterations = 250
test_size = 0.2
grid_search_folds = 10
tissue_type=['GM_WM']
task='AD_vs_CN_VB'

# ########################
# ### ANOVA feature selection
# ########################

## Nested ANOVA
feature_selection_method='ANOVA'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, tissue_type=tissue_type, feature_selection_nested=True, feature_selection_method=feature_selection_method)

## Nested SVM-RFE
feature_selection_method='RFE'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, tissue_type=tissue_type, feature_selection_nested=True, feature_selection_method=feature_selection_method)

## Non_nested ANOVA
feature_selection_method='ANOVA'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, tissue_type=tissue_type, feature_selection_non_nested=True, feature_selection_method=feature_selection_method)

## Non_nested SVM-RFE
feature_selection_method='RFE'
run_voxel_based_classification(caps_directory, diagnoses_tsv, subjects_visits_tsv, output_dir,
                                task, n_threads, n_iterations, test_size, grid_search_folds, tissue_type=tissue_type, feature_selection_non_nested=True, feature_selection_method=feature_selection_method)