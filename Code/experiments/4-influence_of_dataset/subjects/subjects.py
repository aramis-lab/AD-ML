import pandas as pd
from os import path


def sample_diagnoses(n_samples, diagnoses_tsv, subjects_visits_tsv_out, diagnoses_tsv_out, random_states=[1234, 5678]):

    dx = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')

    cn = dx[dx.diagnosis == 'CN'].sample(n_samples, random_state=random_states[0])
    ad = dx[dx.diagnosis == 'AD'].sample(n_samples, random_state=random_states[1])

    merged = cn.append(ad)

    merged.to_csv(diagnoses_tsv_out, index=False, sep='\t', encoding='utf-8')
    merged[['participant_id', 'session_id']].to_csv(subjects_visits_tsv_out, index=False, sep='\t', encoding='utf-8')


# Training and testing on ADNI
n_samples = 70

adni_tasks_dir = '/ADNI/SUBJECTS/lists_by_task'

adni_diagnoses_tsv = path.join(adni_tasks_dir, 'CN_vs_AD_diagnoses.tsv')
adni_subjects_visits_tsv_out = path.join(adni_tasks_dir, 'CN_vs_AD_subjects_sessions_generalization.tsv')
adni_diagnoses_tsv_out = path.join(adni_tasks_dir, 'CN_vs_AD_diagnoses_generalization.tsv')
sample_diagnoses(n_samples, adni_diagnoses_tsv, adni_subjects_visits_tsv_out, adni_diagnoses_tsv_out)

# Training on ADNI with 70%
n_samples = 49
adni_subjects_visits_tsv_out_49 = path.join(adni_tasks_dir, 'CN_vs_AD_subjects_sessions_generalization_49.tsv')
adni_diagnoses_tsv_out_49 = path.join(adni_tasks_dir, 'CN_vs_AD_diagnoses_generalization_49.tsv')
sample_diagnoses(n_samples, adni_diagnoses_tsv_out, adni_subjects_visits_tsv_out_49, adni_diagnoses_tsv_out_49)


n_samples = 70

# Training and testing on AIBL
aibl_tasks_dir = '/AIBL/SUBJECTS/lists_by_task'

aibl_diagnoses_tsv = path.join(aibl_tasks_dir, 'CN_vs_AD_diagnoses.tsv')
aibl_subjects_visits_tsv_out = path.join(aibl_tasks_dir, 'CN_vs_AD_subjects_sessions_generalization.tsv')
aibl_diagnoses_tsv_out = path.join(aibl_tasks_dir, 'CN_vs_AD_diagnoses_generalization.tsv')
sample_diagnoses(n_samples, aibl_diagnoses_tsv, aibl_subjects_visits_tsv_out, aibl_diagnoses_tsv_out, random_states=[475, 578])

# Training and testing on OASIS
oasis_tasks_dir = '/OASIS/SUBJECTS/lists_by_task'

oasis_diagnoses_tsv = path.join(oasis_tasks_dir, 'CN_vs_AD_diagnoses.tsv')
oasis_subjects_visits_tsv_out = path.join(oasis_tasks_dir, 'CN_vs_AD_subjects_sessions_generalization.tsv')
oasis_diagnoses_tsv_out = path.join(oasis_tasks_dir, 'CN_vs_AD_diagnoses_generalization.tsv')
sample_diagnoses(n_samples, oasis_diagnoses_tsv, oasis_subjects_visits_tsv_out, oasis_diagnoses_tsv_out, random_states=[124, 78])
