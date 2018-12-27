#!/usr/bin/env bash
### Run preprocesssing of images
## for diffusion MRI
clinica run dwi-preprocessing-using-t1 /BIDS /CAPS y 0.14 -tsv /SUBJECTS_DIR/subjects_T1_dwi.tsv -wd /WORKING/DIR -np 8
## for T1w MRI
clinica run t1-spm-full-prep /BIDS /CAPS/ ADNIbl -tsv /SUBJECTS_DIR/subjects_T1_dwi.tsv -wd /WORKING/DIR -np 8
