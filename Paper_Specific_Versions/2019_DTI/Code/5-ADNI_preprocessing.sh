#!/usr/bin/env bash
### Run preprocesssing of images
## for diffusion MRI
clinica run dwi-preprocessing-using-t1 '/BIDS_DIR' '/CAPS_DIR' y 0.14 -tsv '/SUBJECTS_DIR/subjects_T1_dwi.tsv' -wd '/WORKING/DIR' -np 8
## for T1w MRI
clinica run t1-volume-new-template '/BIDS_DIR' '/CAPS/' 'group-ADNIbl' -tsv '/SUBJECTS_DIR/subjects_T1_dwi.tsv' -wd '/WORKING/DIR' -np 8
