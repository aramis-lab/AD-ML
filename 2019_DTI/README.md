
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
git clone https://github.com/aramis-lab/clinica.git clinica_for_2019_DTI
cd clinica_for_2019_DTI/
git checkout 0bb46b3b8badfacd1632cf0270b096eb270629d1 
# Former commit-id : a27de358cd4546ad59595d3c565afd3e9b8524a0
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
git clone https://github.com/aramis-lab/AD-ML.git 
git checkout 2019_DTI
jupyter notebook Experiments.ipynb
```


***** TO DELETE *****

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
