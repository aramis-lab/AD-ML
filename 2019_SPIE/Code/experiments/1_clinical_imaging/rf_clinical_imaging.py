
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from general_models import rf_classifications

# Paths to personalize
spie_output_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/OUTPUT', 'SPIE')

input_data_path = os.path.join(spie_output_dir, 'input_data', 'table_1_clinical_imaging')
data_tsv_template = input_data_path + "/input_data_model_%s.tsv"
indices_template = input_data_path + "/indices_model_%s.pkl"
output_dir = os.path.join(spie_output_dir, 'output_data', '1_clinical_imaging')

n_threads = 8
months = [36]

models = {

          # Models using only demographic and clinical data

          "base": ["sex", "education_level", "MMSE", "cdr_sb"],

          "base_logmem": ["sex", "education_level", "MMSE", "cdr_sb", "logmem_delay", "logmem_imm"],

          "base_ravlt": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate"],

          "base_logmem_ravlt": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "logmem_delay",
                                "logmem_imm"],

          "base_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis"],

          "base_ravlt_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis", "ravlt_immediate"],

          # Models including APOE

          "base_ravlt_apoe": ["sex", "education_level", "apoe4", "MMSE", "cdr_sb", "ravlt_immediate"],

          "base_adas_apoe": ["sex", "education_level", "apoe4", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                             "adas_concentration", "adas_praxis"],

          "base_ravlt_adas_apoe": ["sex", "education_level", "apoe4", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                   "adas_concentration", "adas_praxis", "ravlt_immediate"],

          # Models including imaging scores

          "base_T1score": ["sex", "education_level", "MMSE", "cdr_sb", "T1_score"],

          "base_fdgscore": ["sex", "education_level", "MMSE", "cdr_sb", "fdg_score"],

          "base_scores": ["sex", "education_level", "MMSE", "cdr_sb", "T1_score", "fdg_score"],

          "base_ravlt_scores": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "T1_score", "fdg_score"],

          "base_adas_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                               "adas_concentration", "adas_praxis", "T1_score", "fdg_score"],

          "base_adas_memtest_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                       "adas_concentration", "adas_praxis", "ravlt_immediate", "T1_score", "fdg_score"]

}

for model_name in models:
    rf_classifications(model_name, models[model_name], data_tsv_template, indices_template, output_dir, months,
                       n_threads=n_threads)
