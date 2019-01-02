#!/usr/bin/env bash
### Run processing of images
## for diffusion MRI
clinica run dwi-processing-dti '/CAPS_DIR' -tsv '/SUBJECTS_DIR/subjects_T1_dwi.tsv' -wd '/WORKING/DIR' -np 8