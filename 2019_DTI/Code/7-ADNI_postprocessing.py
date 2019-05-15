### Run postprocessing of images
## for diffusion MRI
from clinica_ml_dwi.dwi_postprocessing_dti import dwi_postprocessing_dti

CAPS= 'CAPS_DIR'
tsv= 'SUBJECTS_DIR/participant_tsv_file'
working_dir = 'WORKING/DIR'
tissue_lists = [[1], [2], [1, 2]]
mask_threshold = [0.3]
smooth = [8]

for i in tissue_lists:
    for j in smooth:
        for k in mask_threshold:
            wf = dwi_postprocessing_dti(CAPS, tsv, working_directory=working_dir, mask_tissues=i, mask_threshold=k, smooth=j)
            wf.run(plugin='MultiProc', plugin_args={'n_procs': 8})
