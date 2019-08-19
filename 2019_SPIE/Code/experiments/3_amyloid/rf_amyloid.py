
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from general_models import rf_classifications

# Paths to personalize
spie_output_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/OUTPUT', 'SPIE')

input_data_path = os.path.join(spie_output_dir, 'input_data', 'table_3_amyloid')
data_tsv_template = input_data_path + "/input_data_model_%s.tsv"
indices_template = input_data_path + "/indices_model_%s.pkl"
output_dir = os.path.join(spie_output_dir, 'output_data', '3_amyloid')

n_threads = 8
months = [36]

models = {

          # Models not using amyloid data

          "base": ["sex", "education_level", "MMSE", "cdr_sb"],

          "base_T1score": ["sex", "education_level", "MMSE", "cdr_sb", "T1_score"],

          "base_fdgscore": ["sex", "education_level", "MMSE", "cdr_sb", "fdg_score"],

          "base_ravlt": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate"],

          "base_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                        "adas_concentration", "adas_praxis"],

          "base_ravlt_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis", "ravlt_immediate"],

          "base_ravlt_scores": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "T1_score", "fdg_score"],

          "base_adas_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                               "adas_concentration", "adas_praxis", "T1_score", "fdg_score"],
          "base_ravlt_adas_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                     "adas_concentration", "adas_praxis", "ravlt_immediate", "T1_score", "fdg_score"],

          # Models taking into account amyloid data

          "base_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "amyloid"],

          "base_T1score_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "T1_score", "amyloid"],

          "base_fdgscore_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "fdg_score", "amyloid"],

          "base_ravlt_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "amyloid"],

          "base_adas_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                "adas_concentration", "adas_praxis", "amyloid"],

          "base_ravlt_adas_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                      "adas_concentration", "adas_praxis", "ravlt_immediate", "amyloid"],

          "base_ravlt_scores_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "T1_score",
                                        "fdg_score", "amyloid"],

          "base_adas_scores_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                       "adas_concentration", "adas_praxis", "T1_score", "fdg_score", "amyloid"],
          "base_ravlt_adas_scores_amyloid": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                             "adas_concentration", "adas_praxis", "ravlt_immediate", "T1_score",
                                             "fdg_score", "amyloid"]

}

for model_name in models:
    rf_classifications(model_name, models[model_name], data_tsv_template, indices_template, output_dir, months,
                       n_threads=n_threads)
