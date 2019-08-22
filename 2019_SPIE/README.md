
This subdirectory contains the guidelines and steps to reproduce the experiments from the specific paper **Reproducible evaluation of methods for predicting progression to Alzheimer’s disease from clinical and neuroimaging data** using data obtained from [ADNI](http://adni.loni.usc.edu/) dataset. It is developed by the [ARAMIS Lab](http://www.aramislab.fr).

We assume that you have installed all the dependencies of [Clinica software platform](http://www.clinica.run) and downloaded the ADNI original data.

# Ensure that you have installed the corresponding version of Clinica
Install Clinica in developer's mode using the commit that corresponds to the version of Clinica used in this work. For that, we suggest to use an isolated Conda environment to avoid conflicts with previous versions of libraries already installed in your computer.

```bash
git clone https://github.com/aramis-lab/clinica.git clinica_for_2019_SPIE
cd clinica_for_2019_SPIE/
git checkout 0bb46b3b8badfacd1632cf0270b096eb270629d1
# Former commit-id : a27de358cd4546ad59595d3c565afd3e9b8524a0
conda env create -f environment.yml --name 2019_SPIE_env
conda activate 2019_SPIE_env
conda install jupyter
pip install .
conda deactivate
```

# Clone this repository, install Jupyter notebook and launch the notebook
One you have installed the right version of Clinica (in order to reproduce these experiments). You can run the notebook presented in this folder.

```bash
conda activate 2019_SPIE_env
git clone https://github.com/aramis-lab/AD-ML.git
git checkout 2019_SPIE
jupyter notebook Experiments.ipynb
```
