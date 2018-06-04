from Legacy_MLMI2017.Code.image_postprocessing import ADNI_mask_dti_from_spm

caps_directory= PATH/TO/CAPS
tsv= PATH/TO/TSV
working_dir = PATH/TO/WORKING_DIR
tissues_combinations = [[1], [2], [1, 2]]

for mask_tissue in tissues_combinations:
    if len(mask_tissue) == 1 and mask_tissue[0] == 1:
        print "Run postprocessing for GM"
    elif len(mask_tissue) == 1 and mask_tissue[0] == 2:
        print "Run postprocessing for WM"
    else:
        print "Run postprocessing for GM+WM"
        
    wf = ADNI_mask_dti_from_spm(caps_directory, tsv, working_directory=working_dir, mask_tissues=mask_tissue)
    wf.run(plugin='MultiProc', plugin_args={'n_procs': 72})
