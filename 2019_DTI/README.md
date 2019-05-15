
This subdirectory contains the guideline and steps to reproduce the experiments
in the specific paper **Reproducible evaluation of diffusion MRI features for
automatic classification of patients with Alzheimer’s disease** using diffusion
MRI from ADNI dataset [ADNI](http://adni.loni.usc.edu/). It is developed by the
[ARAMIS Lab](http://www.aramislab.fr).

We assume that you have installed all the dependencies of [Clinica software
platform](http://www.clinica.run) and downloaded the ADNI original data.

# Ensure that you have installed the corresponding version of Clinica
Install Clinica in developer's mode using the commit that corresponds to the
version of Clinica used in this work. Fot that, we suggest to use an isolated
Conda environment to avoid conflicts with previos versions of libraries already
installed in your computer.

```bash
git clone git@gitlab.icm-institute.org:aramislab/clinica.git clinica_for_2019_DTI
cd clinica_for_2019_DTI/
git checkout a27de358cd4546ad59595d3c565afd3e9b8524a0
conda env create -f environment.yml --name 2019_DTI_env
conda activate 2019_DTI_env
conda install jupyter
pip install .
conda deactivate
```

# Clone this repository, install Jupyter notebook and launch the notebook
One you have installed the right version of Clinica (in order to reproduce these
experiments). You can run the notebook presented in this folder.

```bash
conda activate 2019_DTI_env
git clone git@gitlab.icm-institute.org:aramislab/AD-ML.git
git checkout 2019_DTI
jupyter notebook Experiments.ipynb
```

# Convert the original ADNI data into BIDS format

The original ADNI data should be downloaded without further touch (Data we used
in our paper was downloaded in October 2016). 

The command line to convert the data automatically:
```
bash 2-ADNI_conversion.sh
```

# Create the subjects lists

The script to choose the subjects available at baseline for diffusion MRI:
```
python 3-create_subjects_list.py
```

# Create the demographic table information

The script to get demographic information based on the chosen population:
```
python 4-population_statistics.py
```

# Run image processing pipelines

The fist two pipelines were partly integrated into Clinica software and a
postprocesing python script was also developed.

Runs these pipelines sequentially:

```
bash 5-ADNI_preprocessing.sh
```

```
bash 6-ADNI_processing.sh
```

```
python 7-ADNI_postprocessing.py
```

# Run classification tasks

The scripts to obtain the results of the experiments from the paper are as
follows:

Classification results of original data on T1-weighted and diffusion MRI:

```
python 8-ADNI_classification_original_data.py
```

Classification results of balanced data on diffusion MRI:

```
python 9-ADNI_classification_balanced_data.py
```

Classification results of feature selection bias on diffusion MRI:

```
python 10-ADNI_classification_feature_selection.py
```
