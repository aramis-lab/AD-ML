
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from general_models import rf_classifications

# Paths to personalize
spie_output_dir = os.path.join(os.environ.get('OUT_PATH'), 'ADNI/OUTPUT', 'SPIE')

input_data_path = os.path.join(spie_output_dir, 'input_data', 'table_4_timepoints')
data_tsv_template = input_data_path + "/input_data_model_%s.tsv"
indices_template = input_data_path + "/indices_model_%s.pkl"
output_dir = os.path.join(spie_output_dir, 'output_data', '4_timepoints')

n_threads = 8
months = [12, 18, 24, 30, 36]

models = {"base_adas": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                        "adas_concentration", "adas_praxis"],

          "base_adas_scores": ["sex", "education_level", "MMSE", "cdr_sb", "adas_memory", "adas_language",
                               "adas_concentration", "adas_praxis", "T1_score", "fdg_score"]
          }

for model_name in models:
    rf_classifications(model_name, models[model_name], data_tsv_template, indices_template, output_dir, months,
                       n_threads=n_threads)
