
This directory contains the guidelines and steps to generic experiments for Alzheimer’s disease classification from clinical and 
neuroimaging data. The data is obtained from [ADNI](http://adni.loni.usc.edu/) dataset. 
It is developed by the [ARAMIS Lab](http://www.aramislab.fr).

We assume that you have installed all the dependencies of [Clinica software platform](http://www.clinica.run) 
and downloaded the ADNI original data.


# Ensure that you have installed the latest released version of Clinica
Please follow carefully the steps for the installation of Clinica, present in the documentation available at: https://github.com/aramis-lab/AD-ML/wiki

We suggest to use an isolated Conda environment to avoid conflicts with previous versions of libraries 
already installed in your computer.

# Clone this repository, install Jupyter notebook and launch the notebook
One you have installed the right version of Clinica (in order to reproduce these experiments). You can run the notebook presented in this folder.

```bash
conda activate MY_CLINICA_env
git clone https://github.com/aramis-lab/AD-ML.git
git checkout master
cd Generic_Version
jupyter notebook Experiments.ipynb
```
