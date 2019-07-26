
This subdirectory contains the guidelines and steps to reproduce the experiments from the specific paper **Reproducible evaluation of classification methods in Alzheimer's disease: Framework and application to MRI and PET data** using data obtained from [ADNI](http://adni.loni.usc.edu/), [AIBL](https://aibl.csiro.au/research/neuroimaging/) and [OASIS](http://www.oasis-brains.org/) datasets. It is developed by the [ARAMIS Lab](http://www.aramislab.fr).

We assume that you have installed all the dependencies of [Clinica software platform](http://www.clinica.run) and downloaded the ADNI original data.

# Ensure that you have installed the corresponding version of Clinica
Install Clinica in developer's mode using the commit that corresponds to the version of Clinica used in this work. For that, we suggest to use an isolated Conda environment to avoid conflicts with previous versions of libraries already installed in your computer.

```bash
git clone https://github.com/aramis-lab/clinica.git clinica_for_2018_NeuroImage
cd clinica_for_2018_NeuroImage/
git checkout 0bb46b3b8badfacd1632cf0270b096eb270629d1
# Former commit-id : a27de358cd4546ad59595d3c565afd3e9b8524a0
conda env create -f environment.yml --name 2018_NeuroImage_env
conda activate 2018_NeuroImage_env
conda install jupyter
pip install .
conda deactivate
```

# Clone this repository, install Jupyter notebook and launch the notebook
One you have installed the right version of Clinica (in order to reproduce these experiments). You can run the notebook presented in this folder.

```bash
conda activate 2018_NeuroImage_env
git clone https://github.com/aramis-lab/AD-ML.git
git checkout 2018_NeuroImage
jupyter notebook Experiments.ipynb
```
