import pandas as pd
from os import path, environ


adni_bids = path.join(environ.get('OUT_PATH'), 'ADNI/BIDS')
adni_output_dir = path.join(environ.get('OUT_PATH'), 'ADNI/OUTPUT')
spie_output_dir = path.join(adni_output_dir, 'SPIE')

scores_dir = path.join(spie_output_dir, 'input_data/svm_scores')
output_path = path.join(spie_output_dir, 'input_data')


### Joining T1 and FDG-PET scores

T1_df = pd.io.parsers.read_csv(path.join(scores_dir, 'T1.tsv'), sep='\t')
FDG_df = pd.io.parsers.read_csv(path.join(scores_dir, 'fdg.tsv'), sep='\t')
joined = T1_df.merge(FDG_df, on=['diagnosis', 'participant_id', 'session_id'])
joined = joined.rename(columns={'y_hat_x': 'T1_score', 'y_hat_y': 'fdg_score'})
subj_sessions = joined[['participant_id', 'session_id', 'diagnosis', 'T1_score', 'fdg_score']]

participant_columns = ["sex", "education_level", "marital_status", "apoe4", "apoe_gen1", "apoe_gen2"]
session_columns = ["age",
                   # Cognitive measures
                   "MMSE", "cdr_sb", "cdr_global", "adas11", "adas13",
                   "adas_memory", "adas_language", "adas_concentration", "adas_praxis", "ravlt_immediate", "moca",
                   "TMT_A", "TMT_B", "dsst", "logmem_delay", "logmem_imm",
                   # T1 measures
                   "adni_ventricles_vol", "adni_hippocampus_vol", "adni_brain_vol", "adni_entorhinal_vol",
                   "adni_fusiform_vol", "adni_midtemp_vol", "adni_icv",
                   # PET measures
                   "adni_fdg", "adni_pib", "adni_av45",
                   # CSF measures
                   "adni_abeta", "adni_tau", "adni_ptau"]

participant_series = {}
session_series = {}
for col in participant_columns:
    participant_series[col] = []
for col in session_columns:
    session_series[col] = []

participants_tsv = pd.io.parsers.read_csv(path.join(adni_bids, "participants.tsv"), sep='\t')

for row in subj_sessions.iterrows():
    subj_sess = row[1]
    selected_participant = participants_tsv[(participants_tsv.participant_id == subj_sess.participant_id)].iloc[0]
    for col in participant_columns:
        participant_series[col].append(selected_participant[col])

    session_tsv = pd.io.parsers.read_csv(path.join(adni_bids, subj_sess.participant_id,
                                                   subj_sess.participant_id + "_sessions.tsv"), sep='\t')
    selected_session = session_tsv[(session_tsv.session_id == subj_sess.session_id)].iloc[0]
    for col in session_columns:
        session_series[col].append(selected_session[col])

for col in participant_columns:
    subj_sessions.loc[:, col] = pd.Series(participant_series[col], index=subj_sessions.index)

for col in session_columns:
    subj_sessions.loc[:, col] = pd.Series(session_series[col], index=subj_sessions.index)

subj_sessions.to_csv(path.join(output_path, 'input_data.tsv'), sep='\t', index=False)
