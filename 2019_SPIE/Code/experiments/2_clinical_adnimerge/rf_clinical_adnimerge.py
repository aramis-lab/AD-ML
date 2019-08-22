
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from general_models import rf_classifications

# Paths to personalize
spie_output_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/OUTPUT', 'SPIE')

input_data_path = os.path.join(spie_output_dir, 'input_data', 'table_2_clinical_adnimerge')
data_tsv_template = input_data_path + "/input_data_model_%s.tsv"
indices_template = input_data_path + "/indices_model_%s.pkl"
output_dir = os.path.join(spie_output_dir, 'output_data', '2_clinical_adnimerge')

n_threads = 8
months = [36]

models = {

          # Models using only demographic and clinical data

          "base": ["sex", "education_level", "MMSE", "cdr_sb"],

          "base_ravlt": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate"],

          "base_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis"],

          "base_ravlt_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                              "adas_concentration", "adas_praxis", "ravlt_immediate"],

          # Models using also ADNIMERGE T1 and FDG data

          "adnimerge_t1": ["adni_ventricles_vol_icv", "adni_hippocampus_vol_icv", "adni_brain_vol_icv",
                           "adni_entorhinal_vol_icv", "adni_fusiform_vol_icv", "adni_midtemp_vol_icv"],

          "adnimerge_fdg": ["adni_fdg"],

          "adnimerge_t1_fdg": ["adni_ventricles_vol_icv", "adni_hippocampus_vol_icv", "adni_brain_vol_icv",
                               "adni_entorhinal_vol_icv", "adni_fusiform_vol_icv", "adni_midtemp_vol_icv", "adni_fdg"],

          "base_ravlt_adnimerge_t1_fdg_scores": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate",
                                                 "adni_ventricles_vol_icv", "adni_hippocampus_vol_icv",
                                                 "adni_brain_vol_icv", "adni_entorhinal_vol_icv",
                                                 "adni_fusiform_vol_icv", "adni_midtemp_vol_icv", "adni_fdg", "T1_score",
                                                 "fdg_score"],

          "base_ravlt_adas_adnimerge_t1_fdg_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory",
                                                      "adas_language", "adas_concentration", "adas_praxis",
                                                      "ravlt_immediate", "adni_ventricles_vol_icv",
                                                      "adni_hippocampus_vol_icv", "adni_brain_vol_icv",
                                                      "adni_entorhinal_vol_icv", "adni_fusiform_vol_icv",
                                                      "adni_midtemp_vol_icv", "adni_fdg", "T1_score", "fdg_score"],

          # Models including imaging scores

          "T1score": ["T1_score", "fdg_score"],

          "fdgscore": ["fdg_score"],

          "scores": ["T1_score", "fdg_score"],

          "base_ravlt_scores": ["sex", "education_level", "MMSE", "cdr_sb", "ravlt_immediate", "T1_score", "fdg_score"],

          "base_ravlt_adas_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                                     "adas_concentration", "adas_praxis", "ravlt_immediate", "T1_score", "fdg_score"]
          }

for model_name in models:
    rf_classifications(model_name, models[model_name], data_tsv_template, indices_template, output_dir, months,
                       n_threads=n_threads)
