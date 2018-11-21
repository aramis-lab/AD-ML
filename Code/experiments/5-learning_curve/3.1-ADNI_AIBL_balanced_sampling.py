import pandas as pd
from os import path


def sample_diagnoses(n_samples, diagnoses_tsv, subjects_visits_tsv_out, diagnoses_tsv_out, random_states=[1234, 5678]):

    dx = pd.io.parsers.read_csv(diagnoses_tsv, sep='\t')

    cn = dx[dx.diagnosis == 'CN'].sample(n_samples, random_state=random_states[0])
    ad = dx[dx.diagnosis == 'AD'].sample(n_samples, random_state=random_states[1])

    merged = cn.append(ad)

    merged.to_csv(diagnoses_tsv_out, index=False, sep='\t', encoding='utf-8')
    merged[['participant_id', 'session_id']].to_csv(subjects_visits_tsv_out, index=False, sep='\t', encoding='utf-8')


n_samples = 72

# Training and testing on AIBL
aibl_tasks_dir = '/AIBL/SUBJECTS/lists_by_task'

aibl_diagnoses_tsv = path.join(aibl_tasks_dir, 'CN_vs_AD_diagnoses.tsv')
aibl_subjects_visits_tsv_out = path.join(aibl_tasks_dir, 'CN_vs_AD_subjects_sessions_balanced.tsv')
aibl_diagnoses_tsv_out = path.join(aibl_tasks_dir, 'CN_vs_AD_diagnoses_balanced.tsv')
sample_diagnoses(n_samples, aibl_diagnoses_tsv, aibl_subjects_visits_tsv_out, aibl_diagnoses_tsv_out)
